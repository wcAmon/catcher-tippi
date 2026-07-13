import Foundation

public protocol ModelDirectoryMigrating: Sendable {
    func migrateIfNeeded() async throws
}

public enum ModelDirectoryMigrationError: Error, LocalizedError {
    case invalidLegacyDirectory(URL)
    case incompleteCopy
    case missingFileSize(URL)

    public var errorDescription: String? {
        switch self {
        case let .invalidLegacyDirectory(url):
            "The legacy model path is not a directory: \(url.path)"
        case .incompleteCopy:
            "The copied model directory did not match the legacy model directory"
        case let .missingFileSize(url):
            "Could not determine the size of copied model file: \(url.path)"
        }
    }
}

public actor ModelDirectoryMigrator: ModelDirectoryMigrating {
    public typealias MoveItem = @Sendable (URL, URL) throws -> Void

    private struct FileFingerprint: Equatable {
        let byteCount: Int
        let sha256: String
    }

    private let source: URL
    private let destination: URL
    private let fileManager: FileManager
    private let moveItem: MoveItem

    public init(
        source: URL = ModelDirectoryMigrator.defaultLegacyRoot(),
        destination: URL = ModelStore.defaultRootDirectory(),
        fileManager: FileManager = .default,
        moveItem: @escaping MoveItem = {
            try FileManager.default.moveItem(at: $0, to: $1)
        }
    ) {
        self.source = source
        self.destination = destination
        self.fileManager = fileManager
        self.moveItem = moveItem
    }

    public func migrateIfNeeded() async throws {
        var sourceIsDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: source.path, isDirectory: &sourceIsDirectory) else {
            return
        }
        guard sourceIsDirectory.boolValue else {
            throw ModelDirectoryMigrationError.invalidLegacyDirectory(source)
        }
        guard try destinationCanBeReplaced() else {
            return
        }

        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        try removeEmptyDestinationIfPresent()

        do {
            try moveItem(source, destination)
            return
        } catch {
            guard fileManager.fileExists(atPath: source.path) else {
                throw error
            }
        }

        try copyThroughVerifiedStaging()
    }

    public static func defaultLegacyRoot(
        fileManager: FileManager = .default
    ) -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appending(
                path: "Library/Containers/com.wcamon.tippi/Data/Library/Application Support/Tippi/Models",
                directoryHint: .isDirectory
            )
    }

    private func destinationCanBeReplaced() throws -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: destination.path, isDirectory: &isDirectory) else {
            return true
        }
        guard isDirectory.boolValue else {
            return false
        }
        return try fileManager.contentsOfDirectory(atPath: destination.path).isEmpty
    }

    private func removeEmptyDestinationIfPresent() throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
    }

    private func copyThroughVerifiedStaging() throws {
        let staging = destination.deletingLastPathComponent()
            .appending(path: ".Models.migration.partial", directoryHint: .isDirectory)
        if fileManager.fileExists(atPath: staging.path) {
            try fileManager.removeItem(at: staging)
        }

        do {
            try fileManager.copyItem(at: source, to: staging)
            let sourceFingerprint = try fingerprints(in: source)
            let stagingFingerprint = try fingerprints(in: staging)
            guard sourceFingerprint == stagingFingerprint else {
                throw ModelDirectoryMigrationError.incompleteCopy
            }
            try fileManager.moveItem(at: staging, to: destination)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }

        // Promotion produced a complete, independently verified copy. Legacy cleanup is
        // best-effort so a cleanup error cannot turn the only usable copy into a failure.
        try? fileManager.removeItem(at: source)
    }

    private func fingerprints(in root: URL) throws -> [String: FileFingerprint] {
        let resourceKeys: Set<URLResourceKey> = [
            .isDirectoryKey,
            .isRegularFileKey,
            .isSymbolicLinkKey,
            .fileSizeKey,
        ]
        let standardizedRoot = root.standardizedFileURL.path
        let rootPrefix = standardizedRoot.hasSuffix("/")
            ? standardizedRoot
            : standardizedRoot + "/"
        var directories = [root]
        var result: [String: FileFingerprint] = [:]

        while let directory = directories.popLast() {
            let children = try fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: Array(resourceKeys),
                options: []
            )
            for child in children {
                let values = try child.resourceValues(forKeys: resourceKeys)
                if values.isSymbolicLink == true {
                    continue
                }
                if values.isDirectory == true {
                    directories.append(child)
                    continue
                }
                guard values.isRegularFile == true else {
                    continue
                }
                guard let byteCount = values.fileSize else {
                    throw ModelDirectoryMigrationError.missingFileSize(child)
                }
                let childPath = child.standardizedFileURL.path
                guard childPath.hasPrefix(rootPrefix) else {
                    throw ModelDirectoryMigrationError.incompleteCopy
                }
                let relativePath = String(childPath.dropFirst(rootPrefix.count))
                result[relativePath] = FileFingerprint(
                    byteCount: byteCount,
                    sha256: try ModelChecksum.sha256(of: child)
                )
            }
        }

        return result
    }
}

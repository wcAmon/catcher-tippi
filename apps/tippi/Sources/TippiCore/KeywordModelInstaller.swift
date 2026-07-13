import Foundation

public protocol ArchiveExtracting: Sendable {
    func extract(archive: URL, to directory: URL) async throws
}

public protocol KeywordModelInstalling: Sendable {
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL
}

public enum KeywordModelInstallerError: Error, LocalizedError {
    case invalidArchiveChecksum
    case missingRuntimeFile(String)
    case invalidRuntimeFileChecksum(String)
    case incompleteInstallation
    case extractionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidArchiveChecksum:
            "The Tippi Go model archive failed SHA-256 verification"
        case let .missingRuntimeFile(file):
            "The Tippi Go model archive is missing \(file)"
        case let .invalidRuntimeFileChecksum(file):
            "The Tippi Go model file failed SHA-256 verification: \(file)"
        case .incompleteInstallation:
            "The installed Tippi Go model is incomplete"
        case let .extractionFailed(message):
            "Could not extract the Tippi Go model: \(message)"
        }
    }
}

public struct TarArchiveExtractor: ArchiveExtracting {
    public init() {}

    public func extract(archive: URL, to directory: URL) async throws {
        let archivePath = archive.path
        let directoryPath = directory.path
        try await Task.detached {
            let process = Process()
            let standardError = Pipe()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
            process.arguments = ["-xjf", archivePath, "-C", directoryPath]
            process.standardError = standardError

            do {
                try process.run()
            } catch {
                throw KeywordModelInstallerError.extractionFailed(error.localizedDescription)
            }

            let errorData = standardError.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationReason == .exit, process.terminationStatus == 0 else {
                let details = String(data: errorData, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                throw KeywordModelInstallerError.extractionFailed(
                    details?.isEmpty == false
                        ? details!
                        : "tar exited with status \(process.terminationStatus)"
                )
            }
        }.value
    }
}

public actor KeywordModelInstaller: KeywordModelInstalling {
    static let thirdPartyNotice = """
        sherpa-onnx and the sherpa-onnx KWS model
        Copyright (c) k2-fsa contributors
        Licensed under the Apache License, Version 2.0.
        Source: https://github.com/k2-fsa/sherpa-onnx

        """

    private let rootDirectory: URL
    private let manifest: KeywordModelArchive
    private let downloader: any ModelDownloading
    private let extractor: any ArchiveExtracting
    private let fileManager: FileManager

    public init(
        rootDirectory: URL? = nil,
        manifest: KeywordModelArchive = KeywordModelManifest.release,
        downloader: any ModelDownloading = URLSessionModelDownloader(),
        extractor: any ArchiveExtracting = TarArchiveExtractor(),
        fileManager: FileManager = .default
    ) {
        self.rootDirectory =
            rootDirectory ?? ModelStore.defaultRootDirectory(fileManager: fileManager)
        self.manifest = manifest
        self.downloader = downloader
        self.extractor = extractor
        self.fileManager = fileManager
    }

    public func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        let destination = rootDirectory.appending(
            path: manifest.directoryName,
            directoryHint: .isDirectory
        )
        let partials = partialURLs()

        try fileManager.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        try removePartials(partials)
        if try verifyInstallation(at: destination) {
            progress(1.0)
            return destination
        }

        do {
            try await downloader.download(from: manifest.url, to: partials.archive) { value in
                progress(min(max(value, 0), 1) * 0.8)
            }
            guard try ModelChecksum.sha256(of: partials.archive) == manifest.sha256 else {
                throw KeywordModelInstallerError.invalidArchiveChecksum
            }
            progress(0.82)

            try fileManager.createDirectory(
                at: partials.unpack,
                withIntermediateDirectories: true
            )
            try await extractor.extract(archive: partials.archive, to: partials.unpack)
            progress(0.86)

            let extractedModel = partials.unpack.appending(
                path: manifest.directoryName,
                directoryHint: .isDirectory
            )
            try fileManager.createDirectory(
                at: partials.install,
                withIntermediateDirectories: true
            )
            for (index, file) in manifest.files.enumerated() {
                let source = extractedModel.appending(path: file.name)
                guard fileManager.fileExists(atPath: source.path) else {
                    throw KeywordModelInstallerError.missingRuntimeFile(file.name)
                }
                guard try ModelChecksum.sha256(of: source) == file.sha256 else {
                    throw KeywordModelInstallerError.invalidRuntimeFileChecksum(file.name)
                }
                try fileManager.copyItem(
                    at: source,
                    to: partials.install.appending(path: file.name)
                )
                let verifiedShare = Double(index + 1) / Double(manifest.files.count)
                progress(0.86 + verifiedShare * 0.12)
            }
            try Data(KeywordModelManifest.keywords.utf8).write(
                to: partials.install.appending(path: "keywords.txt"),
                options: .atomic
            )
            try Data(Self.thirdPartyNotice.utf8).write(
                to: partials.install.appending(path: "THIRD_PARTY_NOTICES.md"),
                options: .atomic
            )
            guard try verifyInstallation(at: partials.install) else {
                throw KeywordModelInstallerError.incompleteInstallation
            }

            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.moveItem(at: partials.install, to: destination)
            try? fileManager.removeItem(at: partials.archive)
            try? fileManager.removeItem(at: partials.unpack)
            progress(1.0)
            return destination
        } catch {
            try? removePartials(partials)
            throw error
        }
    }

    private func verifyInstallation(at directory: URL) throws -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return false
        }

        let expectedNames = Set(
            manifest.files.map(\.name) + ["keywords.txt", "THIRD_PARTY_NOTICES.md"]
        )
        guard Set(try fileManager.contentsOfDirectory(atPath: directory.path)) == expectedNames else {
            return false
        }
        for file in manifest.files {
            let url = directory.appending(path: file.name)
            guard fileManager.fileExists(atPath: url.path),
                  try ModelChecksum.sha256(of: url) == file.sha256
            else {
                return false
            }
        }
        guard try Data(contentsOf: directory.appending(path: "keywords.txt"))
            == Data(KeywordModelManifest.keywords.utf8)
        else {
            return false
        }
        return try Data(contentsOf: directory.appending(path: "THIRD_PARTY_NOTICES.md"))
            == Data(Self.thirdPartyNotice.utf8)
    }

    private func partialURLs() -> (archive: URL, unpack: URL, install: URL) {
        let prefix = ".\(manifest.directoryName)"
        return (
            rootDirectory.appending(path: "\(prefix).archive.partial"),
            rootDirectory.appending(path: "\(prefix).unpack.partial", directoryHint: .isDirectory),
            rootDirectory.appending(path: "\(prefix).install.partial", directoryHint: .isDirectory)
        )
    }

    private func removePartials(_ partials: (archive: URL, unpack: URL, install: URL)) throws {
        for url in [partials.archive, partials.unpack, partials.install]
            where fileManager.fileExists(atPath: url.path)
        {
            try fileManager.removeItem(at: url)
        }
    }
}

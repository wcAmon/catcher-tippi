import Foundation

public protocol ArchiveExtracting: Sendable {
    func extract(archive: URL, to directory: URL) async throws
}

public protocol KeywordModelInstalling: Sendable {
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL
}

public protocol KeywordModelPromoting: Sendable {
    func promote(staging: URL, to destination: URL, fileManager: FileManager) throws
}

public enum KeywordModelInstallerError: Error, LocalizedError {
    case invalidArchiveChecksum
    case missingRuntimeFile(String)
    case invalidRuntimeFileChecksum(String)
    case incompleteInstallation
    case extractionFailed(String)
    case generatedFileRepairFailed

    public var errorDescription: String? {
        switch self {
        case .invalidArchiveChecksum:
            "口令模型封存檔未通過 SHA-256 驗證"
        case let .missingRuntimeFile(file):
            "口令模型封存檔缺少 \(file)"
        case let .invalidRuntimeFileChecksum(file):
            "口令模型檔案未通過 SHA-256 驗證：\(file)"
        case .incompleteInstallation:
            "已安裝的口令模型不完整"
        case let .extractionFailed(message):
            "無法解壓縮口令模型：\(message)"
        case .generatedFileRepairFailed:
            "無法更新口令模型設定"
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

public struct AtomicKeywordModelPromoter: KeywordModelPromoting {
    public init() {}

    public func promote(staging: URL, to destination: URL, fileManager: FileManager) throws {
        if fileManager.fileExists(atPath: destination.path) {
            _ = try fileManager.replaceItemAt(
                destination,
                withItemAt: staging,
                backupItemName: nil,
                options: []
            )
        } else {
            try fileManager.moveItem(at: staging, to: destination)
        }
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
    private let promoter: any KeywordModelPromoting
    private let fileManager: FileManager

    public init(
        rootDirectory: URL? = nil,
        manifest: KeywordModelArchive = KeywordModelManifest.release,
        downloader: any ModelDownloading = URLSessionModelDownloader(),
        extractor: any ArchiveExtracting = TarArchiveExtractor(),
        promoter: any KeywordModelPromoting = AtomicKeywordModelPromoter(),
        fileManager: FileManager = .default
    ) {
        self.rootDirectory =
            rootDirectory ?? ModelStore.defaultRootDirectory(fileManager: fileManager)
        self.manifest = manifest
        self.downloader = downloader
        self.extractor = extractor
        self.promoter = promoter
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
        if try hasExpectedInventory(at: destination), try verifyRuntimeFiles(at: destination) {
            do {
                try writeGeneratedFiles(to: destination)
                guard try verifyInstallation(at: destination) else {
                    throw KeywordModelInstallerError.incompleteInstallation
                }
            } catch {
                throw KeywordModelInstallerError.generatedFileRepairFailed
            }
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
            try writeGeneratedFiles(to: partials.install)
            guard try verifyInstallation(at: partials.install) else {
                throw KeywordModelInstallerError.incompleteInstallation
            }

            try promoter.promote(
                staging: partials.install,
                to: destination,
                fileManager: fileManager
            )
            try? fileManager.removeItem(at: partials.archive)
            try? fileManager.removeItem(at: partials.unpack)
            progress(1.0)
            return destination
        } catch {
            try? removePartials(partials)
            throw error
        }
    }

    private var expectedInstalledNames: Set<String> {
        Set(manifest.files.map(\.name) + ["keywords.txt", "THIRD_PARTY_NOTICES.md"])
    }

    private func hasExpectedInventory(at directory: URL) throws -> Bool {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            return false
        }

        return Set(try fileManager.contentsOfDirectory(atPath: directory.path))
            == expectedInstalledNames
    }

    private func verifyRuntimeFiles(at directory: URL) throws -> Bool {
        for file in manifest.files {
            let url = directory.appending(path: file.name)
            guard fileManager.fileExists(atPath: url.path),
                  try ModelChecksum.sha256(of: url) == file.sha256
            else {
                return false
            }
        }
        return true
    }

    private func verifyGeneratedFiles(at directory: URL) -> Bool {
        guard let keywords = try? Data(contentsOf: directory.appending(path: "keywords.txt")),
              let notice = try? Data(
                  contentsOf: directory.appending(path: "THIRD_PARTY_NOTICES.md")
              )
        else {
            return false
        }
        return keywords == Data(VoiceSubmitCommand.keywordDefinition.utf8)
            && notice == Data(Self.thirdPartyNotice.utf8)
    }

    private func verifyInstallation(at directory: URL) throws -> Bool {
        guard try hasExpectedInventory(at: directory),
              try verifyRuntimeFiles(at: directory)
        else {
            return false
        }
        return verifyGeneratedFiles(at: directory)
    }

    private func writeGeneratedFiles(to directory: URL) throws {
        try Data(VoiceSubmitCommand.keywordDefinition.utf8).write(
            to: directory.appending(path: "keywords.txt"),
            options: .atomic
        )
        try Data(Self.thirdPartyNotice.utf8).write(
            to: directory.appending(path: "THIRD_PARTY_NOTICES.md"),
            options: .atomic
        )
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

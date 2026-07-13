import CryptoKit
import Foundation
import Testing
@testable import TippiCore

private actor KeywordArchiveDownloader: ModelDownloading {
    let payload: Data
    private(set) var calls: [URL] = []

    init(payload: Data) {
        self.payload = payload
    }

    func download(
        from url: URL,
        to destination: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        calls.append(url)
        progress(0.5)
        try payload.write(to: destination)
        progress(1.0)
    }

    func callCount() -> Int { calls.count }
}

private actor KeywordArchiveExtractor: ArchiveExtracting {
    let modelDirectoryName: String
    let payloads: [String: Data]
    private(set) var callCount = 0

    init(modelDirectoryName: String, payloads: [String: Data]) {
        self.modelDirectoryName = modelDirectoryName
        self.payloads = payloads
    }

    func extract(archive _: URL, to directory: URL) async throws {
        callCount += 1
        let modelDirectory = directory.appending(
            path: modelDirectoryName,
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true
        )
        for (name, payload) in payloads {
            try payload.write(to: modelDirectory.appending(path: name))
        }
        try Data("must not be installed".utf8).write(
            to: modelDirectory.appending(path: "extra.bin")
        )
    }

    func calls() -> Int { callCount }
}

private let keywordRuntimeNames = [
    "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "tokens.txt",
]

private let expectedInstalledKeywordFiles = [
    "THIRD_PARTY_NOTICES.md",
    "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "keywords.txt",
    "tokens.txt",
]

private let expectedThirdPartyNotice = """
    sherpa-onnx and the sherpa-onnx KWS model
    Copyright (c) k2-fsa contributors
    Licensed under the Apache License, Version 2.0.
    Source: https://github.com/k2-fsa/sherpa-onnx

    """

@Test
func installsOnlyVerifiedRuntimeFilesAndGeneratedMetadata() async throws {
    let fixture = keywordInstallerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let downloader = KeywordArchiveDownloader(payload: fixture.archive)
    let extractor = KeywordArchiveExtractor(
        modelDirectoryName: fixture.manifest.directoryName,
        payloads: fixture.payloads
    )
    let installer = KeywordModelInstaller(
        rootDirectory: fixture.root,
        manifest: fixture.manifest,
        downloader: downloader,
        extractor: extractor
    )

    let installed = try await installer.installIfNeeded { _ in }

    let installedFiles = try FileManager.default.contentsOfDirectory(atPath: installed.path).sorted()
    #expect(installedFiles == expectedInstalledKeywordFiles)
    #expect(try Data(contentsOf: installed.appending(path: "keywords.txt"))
        == Data("T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO\n".utf8))
    #expect(try String(contentsOf: installed.appending(path: "THIRD_PARTY_NOTICES.md"), encoding: .utf8)
        == expectedThirdPartyNotice)
    #expect(await downloader.callCount() == 1)
    #expect(await extractor.calls() == 1)
    assertNoKeywordPartials(root: fixture.root, directoryName: fixture.manifest.directoryName)
}

@Test
func archiveHashMismatchCleansEveryStagingPath() async throws {
    let fixture = keywordInstallerFixture(archiveHash: sha256(Data("different".utf8)))
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let downloader = KeywordArchiveDownloader(payload: fixture.archive)
    let extractor = KeywordArchiveExtractor(
        modelDirectoryName: fixture.manifest.directoryName,
        payloads: fixture.payloads
    )
    let installer = KeywordModelInstaller(
        rootDirectory: fixture.root,
        manifest: fixture.manifest,
        downloader: downloader,
        extractor: extractor
    )

    await #expect(throws: KeywordModelInstallerError.self) {
        _ = try await installer.installIfNeeded { _ in }
    }

    #expect(await extractor.calls() == 0)
    #expect(!FileManager.default.fileExists(
        atPath: fixture.root.appending(path: fixture.manifest.directoryName).path
    ))
    assertNoKeywordPartials(root: fixture.root, directoryName: fixture.manifest.directoryName)
}

@Test
func extractedFileHashMismatchDoesNotReplaceExistingInstall() async throws {
    let fixture = keywordInstallerFixture(corruptFile: keywordRuntimeNames[0])
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let destination = fixture.root.appending(
        path: fixture.manifest.directoryName,
        directoryHint: .isDirectory
    )
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    let marker = destination.appending(path: "existing-install.marker")
    try Data("keep me".utf8).write(to: marker)
    let downloader = KeywordArchiveDownloader(payload: fixture.archive)
    let extractor = KeywordArchiveExtractor(
        modelDirectoryName: fixture.manifest.directoryName,
        payloads: fixture.payloads
    )
    let installer = KeywordModelInstaller(
        rootDirectory: fixture.root,
        manifest: fixture.manifest,
        downloader: downloader,
        extractor: extractor
    )

    await #expect(throws: KeywordModelInstallerError.self) {
        _ = try await installer.installIfNeeded { _ in }
    }

    #expect(try Data(contentsOf: marker) == Data("keep me".utf8))
    #expect(try FileManager.default.contentsOfDirectory(atPath: destination.path)
        == ["existing-install.marker"])
    assertNoKeywordPartials(root: fixture.root, directoryName: fixture.manifest.directoryName)
}

@Test
func verifiedExistingInstallSkipsDownloadAndExtraction() async throws {
    let fixture = keywordInstallerFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    let destination = fixture.root.appending(
        path: fixture.manifest.directoryName,
        directoryHint: .isDirectory
    )
    try writeVerifiedInstall(
        at: destination,
        payloads: fixture.payloads
    )
    for suffix in ["archive", "unpack", "install"] {
        let stale = fixture.root.appending(
            path: ".\(fixture.manifest.directoryName).\(suffix).partial",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: stale, withIntermediateDirectories: true)
    }
    let downloader = KeywordArchiveDownloader(payload: fixture.archive)
    let extractor = KeywordArchiveExtractor(
        modelDirectoryName: fixture.manifest.directoryName,
        payloads: fixture.payloads
    )
    let installer = KeywordModelInstaller(
        rootDirectory: fixture.root,
        manifest: fixture.manifest,
        downloader: downloader,
        extractor: extractor
    )

    let installed = try await installer.installIfNeeded { _ in }

    #expect(installed == destination)
    #expect(await downloader.callCount() == 0)
    #expect(await extractor.calls() == 0)
    assertNoKeywordPartials(root: fixture.root, directoryName: fixture.manifest.directoryName)
}

@Test
func keywordReleaseManifestPinsOfficialChunk16Artifacts() {
    let release = KeywordModelManifest.release
    #expect(release.url.absoluteString
        == "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2")
    #expect(release.sha256
        == "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6")
    #expect(release.byteCount == 32_885_699)
    #expect(release.files.map(\.name) == keywordRuntimeNames)
    #expect(release.files.map(\.byteCount) == [4_599_656, 759_829, 86_629, 1_928])
    #expect(KeywordModelManifest.keywords
        == "T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO\n")
}

private struct KeywordInstallerFixture {
    let root: URL
    let archive: Data
    let payloads: [String: Data]
    let manifest: KeywordModelArchive
}

private func keywordInstallerFixture(
    archiveHash: String? = nil,
    corruptFile: String? = nil
) -> KeywordInstallerFixture {
    let root = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    let archive = Data("pinned archive".utf8)
    let verifiedPayloads = Dictionary(
        uniqueKeysWithValues: keywordRuntimeNames.enumerated().map { index, name in
            (name, Data("runtime-\(index)".utf8))
        }
    )
    var extractedPayloads = verifiedPayloads
    if let corruptFile {
        extractedPayloads[corruptFile] = Data("corrupt".utf8)
    }
    let manifest = KeywordModelArchive(
        url: URL(string: "https://example.test/keyword-model.tar.bz2")!,
        sha256: archiveHash ?? sha256(archive),
        byteCount: Int64(archive.count),
        directoryName: "test-keyword-model",
        files: keywordRuntimeNames.map { name in
            ModelFile(
                name: name,
                sha256: sha256(verifiedPayloads[name]!),
                required: true,
                byteCount: Int64(verifiedPayloads[name]!.count)
            )
        }
    )
    return KeywordInstallerFixture(
        root: root,
        archive: archive,
        payloads: extractedPayloads,
        manifest: manifest
    )
}

private func writeVerifiedInstall(at destination: URL, payloads: [String: Data]) throws {
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    for (name, payload) in payloads {
        try payload.write(to: destination.appending(path: name))
    }
    try Data(KeywordModelManifest.keywords.utf8).write(
        to: destination.appending(path: "keywords.txt")
    )
    try Data(expectedThirdPartyNotice.utf8).write(
        to: destination.appending(path: "THIRD_PARTY_NOTICES.md")
    )
}

private func assertNoKeywordPartials(root: URL, directoryName: String) {
    for suffix in ["archive", "unpack", "install"] {
        #expect(!FileManager.default.fileExists(
            atPath: root.appending(path: ".\(directoryName).\(suffix).partial").path
        ))
    }
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

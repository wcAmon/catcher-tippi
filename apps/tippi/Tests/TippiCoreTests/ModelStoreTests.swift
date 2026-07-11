import CryptoKit
import Foundation
import Testing
@testable import TippiCore

private actor FakeDownloader: ModelDownloading {
    let payloads: [String: Data]
    private(set) var calls: [URL] = []

    init(payloads: [String: Data]) {
        self.payloads = payloads
    }

    func download(
        from url: URL,
        to destination: URL,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws {
        calls.append(url)
        progress(0.25)
        let data = try #require(payloads[url.lastPathComponent])
        try data.write(to: destination)
        progress(1.0)
    }

    func callCount() -> Int { calls.count }
}

private actor ProgressRecorder {
    private(set) var values: [Double] = []
    func append(_ value: Double) { values.append(value) }
    func snapshot() -> [Double] { values }
}

@Test
func installsVerifiedFilesAtomicallyAndReportsMonotonicProgress() async throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    defer { try? FileManager.default.removeItem(at: root) }
    let weights = Data("weights".utf8)
    let manifest = Data("manifest".utf8)
    let files = [
        ModelFile(name: "weights.safetensors", sha256: sha256(weights), required: true, byteCount: 3),
        ModelFile(name: "manifest.json", sha256: sha256(manifest), required: true, byteCount: 1),
    ]
    let downloader = FakeDownloader(payloads: [
        "weights.safetensors": weights,
        "manifest.json": manifest,
    ])
    let progress = ProgressRecorder()
    let store = ModelStore(
        rootDirectory: root,
        baseURL: URL(string: "https://example.test/model/")!,
        files: files,
        downloader: downloader
    )

    let installed = try await store.installIfNeeded { value in
        Task { await progress.append(value) }
    }

    #expect(FileManager.default.fileExists(atPath: installed.appending(path: "weights.safetensors").path))
    #expect(!FileManager.default.fileExists(atPath: root.appending(path: ".catcher-asr-mlx-int8.partial").path))
    let values = await progress.snapshot()
    #expect(values == values.sorted())
    #expect(values.contains(0.1875))
    #expect(values.last == 1.0)
}

@Test
func checksumMismatchRemovesStagingAndDoesNotPromoteModel() async throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    defer { try? FileManager.default.removeItem(at: root) }
    let downloader = FakeDownloader(payloads: ["weights.safetensors": Data("bad".utf8)])
    let store = ModelStore(
        rootDirectory: root,
        baseURL: URL(string: "https://example.test/model/")!,
        files: [ModelFile(name: "weights.safetensors", sha256: sha256(Data("good".utf8)), required: true)],
        downloader: downloader
    )

    await #expect(throws: ModelStoreError.self) {
        try await store.installIfNeeded { _ in }
    }
    #expect(!FileManager.default.fileExists(atPath: root.appending(path: "catcher-asr-mlx-int8").path))
    #expect(!FileManager.default.fileExists(atPath: root.appending(path: ".catcher-asr-mlx-int8.partial").path))
}

@Test
func existingVerifiedModelSkipsNetwork() async throws {
    let root = FileManager.default.temporaryDirectory
        .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    defer { try? FileManager.default.removeItem(at: root) }
    let data = Data("ready".utf8)
    let destination = root.appending(path: "catcher-asr-mlx-int8", directoryHint: .isDirectory)
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    try data.write(to: destination.appending(path: "weights.safetensors"))
    let downloader = FakeDownloader(payloads: [:])
    let store = ModelStore(
        rootDirectory: root,
        baseURL: URL(string: "https://example.test/model/")!,
        files: [ModelFile(name: "weights.safetensors", sha256: sha256(data), required: true)],
        downloader: downloader
    )

    let installed = try await store.installIfNeeded { _ in }

    #expect(installed == destination)
    #expect(await downloader.callCount() == 0)
}

private func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

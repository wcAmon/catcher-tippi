import Foundation
import Testing
@testable import TippiCore

private actor ScriptedInstaller: ModelInstalling {
    let url: URL
    let steps: [Double]
    let error: (any Error)?
    init(url: URL, steps: [Double], error: (any Error)? = nil) {
        self.url = url
        self.steps = steps
        self.error = error
    }
    func installIfNeeded(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        for step in steps { progress(step) }
        if let error { throw error }
        return url
    }
}

private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [Double] = []
    func append(_ value: Double) {
        lock.lock()
        defer { lock.unlock() }
        values.append(value)
    }
    func snapshot() -> [Double] {
        lock.lock()
        defer { lock.unlock() }
        return values
    }
}

private enum TestFailure: Error { case download }

@Test
func mergesProgressWeightedByBytes() async throws {
    // ASR 600、diar 200 bytes → 權重 0.75 / 0.25。
    let installer = ModelBundleInstaller(
        asr: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/asr"), steps: [0.5, 1.0]),
        asrTotalBytes: 600,
        diar: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/diar"), steps: [0.5, 1.0]),
        diarTotalBytes: 200
    )
    let recorder = ProgressRecorder()

    let bundle = try await installer.installIfNeeded { value in
        recorder.append(value)
    }

    #expect(bundle == ModelBundle(
        asr: URL(fileURLWithPath: "/tmp/asr"),
        diar: URL(fileURLWithPath: "/tmp/diar")
    ))
    let values = recorder.snapshot()
    #expect(values == values.sorted())
    #expect(values.contains(0.375))   // ASR 一半:0.5 × 0.75
    #expect(values.contains(0.75))    // ASR 完成
    #expect(values.contains(0.875))   // diar 一半:0.75 + 0.5 × 0.25
    #expect(values.last == 1.0)
}

@Test
func alreadyInstalledAsrJumpsStraightToItsShare() async throws {
    let installer = ModelBundleInstaller(
        asr: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/asr"), steps: [1.0]),
        asrTotalBytes: 600,
        diar: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/diar"), steps: [1.0]),
        diarTotalBytes: 200
    )
    let recorder = ProgressRecorder()

    _ = try await installer.installIfNeeded { value in
        recorder.append(value)
    }

    let values = recorder.snapshot()
    #expect(values.first == 0.75)
    #expect(values.last == 1.0)
}

@Test
func diarFailurePropagates() async {
    let installer = ModelBundleInstaller(
        asr: ScriptedInstaller(url: URL(fileURLWithPath: "/tmp/asr"), steps: [1.0]),
        asrTotalBytes: 600,
        diar: ScriptedInstaller(
            url: URL(fileURLWithPath: "/tmp/diar"),
            steps: [0.5],
            error: TestFailure.download
        ),
        diarTotalBytes: 200
    )

    await #expect(throws: TestFailure.self) {
        _ = try await installer.installIfNeeded { _ in }
    }
}

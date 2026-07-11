import Foundation
import Testing
@testable import TippiCore

private actor FakeInstaller: ModelInstalling {
    let url: URL
    init(url: URL) { self.url = url }
    func installIfNeeded(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        progress(0.5)
        progress(1.0)
        return url
    }
}

private actor FakeCatcher: CatcherServing {
    private(set) var events: [String] = []
    func start() async throws { events.append("start") }
    func push(_ samples: [Float]) async throws -> String? {
        events.append("push:\(samples.count)")
        return "live words"
    }
    func finish() async throws -> String {
        events.append("finish")
        return "final words"
    }
    func snapshot() -> [String] { events }
}

private actor FakeAudio: AudioRecording {
    private var sink: (@Sendable ([Float]) -> Void)?
    private(set) var events: [String] = []
    var startError: (any Error)?

    func start(onSamples: @escaping @Sendable ([Float]) -> Void) async throws {
        if let startError { throw startError }
        sink = onSamples
        events.append("start")
    }

    func stop() async {
        events.append("stop")
        sink = nil
    }

    func emit(_ samples: [Float]) { sink?(samples) }
    func snapshot() -> [String] { events }
}

private enum TestFailure: Error { case microphone }

@MainActor
@Test
func recordingTogglePublishesPartialThenFinalText() async throws {
    let catcher = FakeCatcher()
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(url: URL(fileURLWithPath: "/tmp/model")),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()
    #expect(controller.state == .ready)

    await controller.toggleRecording()
    #expect(controller.state == .recording)
    await audio.emit([0.1, 0.2, 0.3])
    try await Task.sleep(for: .milliseconds(20))
    #expect(controller.text == "live words")

    await controller.toggleRecording()
    #expect(controller.state == .ready)
    #expect(controller.text == "final words")
    #expect(await audio.snapshot() == ["start", "stop"])
    #expect(await catcher.snapshot() == ["start", "push:3", "finish"])
}

@MainActor
@Test
func microphoneFailureLeavesRecordingOffAndCanBeRetried() async {
    let catcher = FakeCatcher()
    let audio = FakeAudio()
    await audio.setStartError(TestFailure.microphone)
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(url: URL(fileURLWithPath: "/tmp/model")),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()

    await controller.toggleRecording()

    guard case .failed = controller.state else {
        Issue.record("expected failed state")
        return
    }
    #expect(!controller.isRecording)
}

private extension FakeAudio {
    func setStartError(_ error: any Error) {
        startError = error
    }
}

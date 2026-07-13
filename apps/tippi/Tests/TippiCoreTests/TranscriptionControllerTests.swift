import Foundation
import Testing
@testable import TippiCore

private actor FakeInstaller: ModelBundleInstalling {
    let bundle: ModelBundle
    init(bundle: ModelBundle) { self.bundle = bundle }
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelBundle {
        progress(0.5)
        progress(1.0)
        return bundle
    }
}

private actor FakeCatcher: CatcherServing {
    private(set) var events: [String] = []
    private var pushUpdates: [TranscriptUpdate?] = []
    private var finishUpdate = TranscriptUpdate(text: "", segments: [], warning: nil)

    func start() async throws { events.append("start") }

    func push(_ samples: [Float]) async throws -> TranscriptUpdate? {
        events.append("push:\(samples.count)")
        return pushUpdates.isEmpty ? nil : pushUpdates.removeFirst()
    }

    func finish() async throws -> TranscriptUpdate {
        events.append("finish")
        return finishUpdate
    }

    func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate {
        events.append("finishBefore:\(cutoffMs)")
        return finishUpdate
    }

    func script(pushes: [TranscriptUpdate?], finish: TranscriptUpdate) {
        pushUpdates = pushes
        finishUpdate = finish
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
    func setStartError(_ error: any Error) { startError = error }
}

private enum TestFailure: Error { case microphone }

private let testBundle = ModelBundle(
    asr: URL(fileURLWithPath: "/tmp/asr"),
    diar: URL(fileURLWithPath: "/tmp/diar")
)

private func segment(
    _ speaker: Int, _ startMs: UInt64, _ text: String, final isFinal: Bool
) -> SpeakerSegment {
    SpeakerSegment(speaker: speaker, startMs: startMs, endMs: startMs + 80, text: text, isFinal: isFinal)
}

@MainActor
@Test
func recordingPublishesMessagesThenFinalizesOnStop() async throws {
    let catcher = FakeCatcher()
    await catcher.script(
        pushes: [TranscriptUpdate(
            text: "今天先討論這個。",
            segments: [segment(0, 400, "今天先討論這個。", final: false)],
            warning: nil
        )],
        finish: TranscriptUpdate(
            text: "今天先討論這個。好。",
            segments: [
                segment(0, 400, "今天先討論這個。", final: true),
                segment(1, 2080, "好。", final: true),
            ],
            warning: nil
        )
    )
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()
    #expect(controller.state == .ready)

    await controller.toggleRecording()
    #expect(controller.state == .recording)
    await audio.emit([0.1, 0.2, 0.3])
    try await Task.sleep(for: .milliseconds(20))
    #expect(controller.messages == [
        Message(id: 0, speaker: 0, startMs: 400, endMs: 480, text: "今天先討論這個。", isFinal: false)
    ])

    await controller.toggleRecording()
    #expect(controller.state == .ready)
    #expect(controller.messages.count == 2)
    #expect(controller.messages.allSatisfy { $0.isFinal })
    #expect(controller.messages[1].speaker == 1)
    #expect(await catcher.snapshot() == ["start", "push:3", "finish"])
}

@MainActor
@Test
func warningIsPublishedAndClearedOnNextRecording() async throws {
    let catcher = FakeCatcher()
    await catcher.script(
        pushes: [TranscriptUpdate(
            text: "喂?",
            segments: [segment(0, 0, "喂?", final: false)],
            warning: "diarization disabled after a runtime error: injected"
        )],
        finish: TranscriptUpdate(
            text: "喂?",
            segments: [segment(0, 0, "喂?", final: true)],
            warning: nil
        )
    )
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()

    await controller.toggleRecording()
    await audio.emit([0.1])
    try await Task.sleep(for: .milliseconds(20))
    #expect(controller.warningMessage == "diarization disabled after a runtime error: injected")
    await controller.toggleRecording()
    #expect(controller.warningMessage == nil)

    // 新錄音清空訊息、命名與警告。
    controller.speakerNames[0] = "小明"
    await controller.toggleRecording()
    #expect(controller.messages.isEmpty)
    #expect(controller.speakerNames.isEmpty)
    #expect(controller.warningMessage == nil)
    await controller.toggleRecording()
}

@MainActor
@Test
func microphoneFailureLeavesRecordingOffAndCanBeRetried() async {
    let catcher = FakeCatcher()
    let audio = FakeAudio()
    await audio.setStartError(TestFailure.microphone)
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
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

@MainActor
@Test
func clearTranscriptClearsOnlyWhenReady() async throws {
    let catcher = FakeCatcher()
    await catcher.script(
        pushes: [TranscriptUpdate(
            text: "今天先討論這個。",
            segments: [segment(0, 400, "今天先討論這個。", final: false)],
            warning: "diarization disabled after a runtime error: injected"
        )],
        finish: TranscriptUpdate(
            text: "今天先討論這個。",
            segments: [segment(0, 400, "今天先討論這個。", final: true)],
            warning: "diarization disabled after a runtime error: injected"
        )
    )
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()

    await controller.toggleRecording()
    await audio.emit([0.1])
    try await Task.sleep(for: .milliseconds(20))
    controller.speakerNames[0] = "小明"

    // 錄音中呼叫是 no-op。
    controller.clearTranscript()
    #expect(!controller.messages.isEmpty)
    #expect(controller.speakerNames == [0: "小明"])

    await controller.toggleRecording()
    #expect(controller.state == .ready)
    #expect(controller.warningMessage != nil)

    controller.clearTranscript()
    #expect(controller.messages.isEmpty)
    #expect(controller.speakerNames.isEmpty)
    #expect(controller.warningMessage == nil)
    #expect(controller.state == .ready)
}

import Foundation
import Testing
@testable import TippiCore

private final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func append(_ event: String) {
        lock.withLock { storage.append(event) }
    }

    func snapshot() -> [String] {
        lock.withLock { storage }
    }

    func removeAll() {
        lock.withLock { storage.removeAll() }
    }
}

private actor FakeInstaller: ModelBundleInstalling {
    let bundle: ModelBundle
    let log: EventLog?
    var error: TestFailure?

    init(bundle: ModelBundle, log: EventLog? = nil) {
        self.bundle = bundle
        self.log = log
    }

    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ModelBundle {
        log?.append("model.install")
        if let error { throw error }
        progress(0.5)
        progress(1.0)
        return bundle
    }
}

private actor FakeMigrator: ModelDirectoryMigrating {
    let log: EventLog?
    var error: TestFailure?

    init(log: EventLog? = nil) {
        self.log = log
    }

    func migrateIfNeeded() async throws {
        log?.append("migrate")
        if let error { throw error }
    }
}

private actor FakeKeywordInstaller: KeywordModelInstalling {
    let directory: URL
    let log: EventLog?
    var error: TestFailure?

    init(
        directory: URL = URL(fileURLWithPath: "/tmp/kws"),
        log: EventLog? = nil
    ) {
        self.directory = directory
        self.log = log
    }

    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL {
        log?.append("kws.install")
        if let error { throw error }
        progress(0.25)
        progress(1.0)
        return directory
    }
}

private actor FakeCatcher: CatcherServing {
    private let log: EventLog?
    private var events: [String] = []
    private var pushUpdates: [TranscriptUpdate?] = []
    private var stableTexts: [String] = []
    private var finishUpdate = TranscriptUpdate(text: "", segments: [], warning: nil)
    private var finishBeforeUpdate = TranscriptUpdate(text: "", segments: [], warning: nil)
    private var startError: TestFailure?
    private var pushError: TestFailure?
    private var textBeforeError: TestFailure?
    private var finishError: TestFailure?
    private var finishBeforeError: TestFailure?

    init(log: EventLog? = nil) {
        self.log = log
    }

    func start() async throws {
        record("start")
        if let startError { throw startError }
    }

    func push(_ samples: [Float]) async throws -> TranscriptUpdate? {
        record("push", detailed: "push:\(samples.count)")
        if let pushError { throw pushError }
        return pushUpdates.isEmpty ? nil : pushUpdates.removeFirst()
    }

    func text(before cutoffMs: UInt64) async throws -> String {
        record("textBefore:\(cutoffMs)")
        if let textBeforeError { throw textBeforeError }
        return stableTexts.isEmpty ? "" : stableTexts.removeFirst()
    }

    func finish() async throws -> TranscriptUpdate {
        record("finish")
        if let finishError { throw finishError }
        return finishUpdate
    }

    func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate {
        record("finishBefore:\(cutoffMs)")
        if let finishBeforeError { throw finishBeforeError }
        return finishBeforeUpdate
    }

    func script(
        pushes: [TranscriptUpdate?] = [],
        stableTexts: [String] = [],
        finish: TranscriptUpdate = TranscriptUpdate(text: "", segments: [], warning: nil),
        finishBefore: TranscriptUpdate? = nil
    ) {
        pushUpdates = pushes
        self.stableTexts = stableTexts
        finishUpdate = finish
        finishBeforeUpdate = finishBefore ?? finish
    }

    func failStart(with error: TestFailure) { startError = error }
    func failPush(with error: TestFailure) { pushError = error }
    func failTextBefore(with error: TestFailure) { textBeforeError = error }
    func failFinish(with error: TestFailure) { finishError = error }
    func failFinishBefore(with error: TestFailure) { finishBeforeError = error }
    func snapshot() -> [String] { events }

    private func record(_ event: String, detailed: String? = nil) {
        events.append(detailed ?? event)
        log?.append("asr.\(event)")
    }
}

private actor FakeKeywordSpotter: KeywordSpotting {
    private let log: EventLog?
    private var events: [String] = []
    private var detections: [KeywordDetection?] = []
    private var startError: TestFailure?
    private var pushError: TestFailure?
    private var resetError: TestFailure?

    init(log: EventLog? = nil) {
        self.log = log
    }

    func start() async throws {
        record("start")
        if let startError { throw startError }
    }

    func push(_ samples: [Float]) async throws -> KeywordDetection? {
        record("push")
        if let pushError { throw pushError }
        return detections.isEmpty ? nil : detections.removeFirst()
    }

    func reset() async throws {
        record("reset")
        if let resetError { throw resetError }
    }

    func script(_ values: [KeywordDetection?]) { detections = values }
    func failStart(with error: TestFailure) { startError = error }
    func failPush(with error: TestFailure) { pushError = error }
    func clearPushError() { pushError = nil }
    func failReset(with error: TestFailure) { resetError = error }
    func snapshot() -> [String] { events }

    private func record(_ event: String) {
        events.append(event)
        log?.append("kws.\(event)")
    }
}

private actor FakeAudio: AudioRecording {
    private let log: EventLog?
    private var sink: (@Sendable ([Float]) -> Void)?
    private(set) var events: [String] = []
    var startError: TestFailure?

    init(log: EventLog? = nil) {
        self.log = log
    }

    func start(onSamples: @escaping @Sendable ([Float]) -> Void) async throws {
        events.append("start")
        log?.append("audio.start")
        if let startError { throw startError }
        sink = onSamples
    }

    func stop() async {
        events.append("stop")
        log?.append("audio.stop")
        sink = nil
    }

    func emit(_ samples: [Float]) { sink?(samples) }
    func snapshot() -> [String] { events }
    func setStartError(_ error: TestFailure) { startError = error }
}

@MainActor
private final class FakeTextInjector: TextInjecting {
    let log: EventLog?
    var trusted: Bool
    private(set) var injected: [String] = []
    private(set) var submitCount = 0

    init(trusted: Bool = true, log: EventLog? = nil) {
        self.trusted = trusted
        self.log = log
    }

    func isTrusted(prompt: Bool) -> Bool {
        log?.append("permission:\(prompt)")
        return trusted
    }

    func inject(_ text: String) throws {
        injected.append(text)
        log?.append("inject:\(text)")
    }

    func submit() throws {
        submitCount += 1
        log?.append("submit")
    }
}

@MainActor
private final class FakeTargetProvider: FrontmostApplicationProviding {
    var target: TargetApplication?

    init(target: TargetApplication?) {
        self.target = target
    }

    func current() -> TargetApplication? { target }
}

private enum TestFailure: Error, LocalizedError, Sendable {
    case microphone
    case keyword
    case inference
    case migration
    case installation

    var errorDescription: String? {
        switch self {
        case .microphone: "microphone failed"
        case .keyword: "keyword failed"
        case .inference: "inference failed"
        case .migration: "migration failed"
        case .installation: "installation failed"
        }
    }
}

private let testBundle = ModelBundle(
    asr: URL(fileURLWithPath: "/tmp/asr"),
    diar: URL(fileURLWithPath: "/tmp/diar")
)
private let tippiBundleIdentifier = "com.wcamon.tippi"
private let textEdit = TargetApplication(
    name: "TextEdit",
    bundleIdentifier: "com.apple.TextEdit"
)

private func segment(
    _ speaker: Int, _ startMs: UInt64, _ text: String, final isFinal: Bool
) -> SpeakerSegment {
    SpeakerSegment(
        speaker: speaker,
        startMs: startMs,
        endMs: startMs + 80,
        text: text,
        isFinal: isFinal
    )
}

private struct ControllerFixture {
    let controller: TranscriptionController
    let catcher: FakeCatcher
    let keywordSpotter: FakeKeywordSpotter
    let audio: FakeAudio
    let migrator: FakeMigrator
    let keywordInstaller: FakeKeywordInstaller
    let injector: FakeTextInjector
    let targetProvider: FakeTargetProvider
}

@MainActor
private func makeFixture(
    log: EventLog? = nil,
    trusted: Bool = true,
    target: TargetApplication? = textEdit
) -> ControllerFixture {
    let catcher = FakeCatcher(log: log)
    let keywordSpotter = FakeKeywordSpotter(log: log)
    let audio = FakeAudio(log: log)
    let migrator = FakeMigrator(log: log)
    let keywordInstaller = FakeKeywordInstaller(log: log)
    let injector = FakeTextInjector(trusted: trusted, log: log)
    let targetProvider = FakeTargetProvider(target: target)
    let coordinator = TextInjectionCoordinator(
        injector: injector,
        targetProvider: targetProvider,
        ownBundleIdentifier: tippiBundleIdentifier
    )
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle, log: log),
        audio: audio,
        catcherFactory: { _ in catcher },
        modelMigrator: migrator,
        keywordInstaller: keywordInstaller,
        keywordFactory: { directory in
            log?.append("kws.factory:\(directory.lastPathComponent)")
            return keywordSpotter
        },
        injectionCoordinator: coordinator
    )
    return ControllerFixture(
        controller: controller,
        catcher: catcher,
        keywordSpotter: keywordSpotter,
        audio: audio,
        migrator: migrator,
        keywordInstaller: keywordInstaller,
        injector: injector,
        targetProvider: targetProvider
    )
}

@MainActor
private func prepareVoiceFixture(_ fixture: ControllerFixture) async {
    await fixture.controller.prepare()
    await fixture.controller.prepareVoiceInput()
}

@MainActor
private func waitUntil(
    timeout: Duration = .seconds(1),
    _ condition: @escaping @MainActor () async -> Bool
) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while !(await condition()) {
        if clock.now >= deadline {
            Issue.record("timed out waiting for asynchronous controller work")
            return
        }
        try await Task.sleep(for: .milliseconds(5))
    }
}

@Test(arguments: zip(
    [UInt64(0), 23_999, 24_000, 40_000],
    [UInt64(0), 0, 0, 1_000]
))
func stableCutoffUsesSixteenKilohertzSampleClock(
    sampleCount: UInt64,
    expected: UInt64
) {
    #expect(VoiceInputTiming.stableCutoffMs(receivedSampleCount: sampleCount) == expected)
}

@MainActor
@Test
func recordingPublishesMessagesThenFinalizesOnStop() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
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
    await fixture.controller.prepare()
    #expect(fixture.controller.state == .ready)

    await fixture.controller.toggleRecording(mode: .transcription)
    #expect(fixture.controller.state == .recording)
    #expect(fixture.controller.activeMode == .transcription)
    await fixture.audio.emit([0.1, 0.2, 0.3])
    try await waitUntil { await fixture.catcher.snapshot().contains("push:3") }
    #expect(fixture.controller.messages == [
        Message(
            id: 0,
            speaker: 0,
            startMs: 400,
            endMs: 480,
            text: "今天先討論這個。",
            isFinal: false
        )
    ])

    await fixture.controller.toggleRecording(mode: .transcription)
    #expect(fixture.controller.state == .ready)
    #expect(fixture.controller.activeMode == nil)
    #expect(fixture.controller.messages.count == 2)
    #expect(fixture.controller.messages.allSatisfy { $0.isFinal })
    #expect(fixture.controller.messages[1].speaker == 1)
    #expect(await fixture.catcher.snapshot() == ["start", "push:3", "finish"])
}

@MainActor
@Test
func warningIsPublishedAndClearedOnNextRecording() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
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
    await fixture.controller.prepare()

    await fixture.controller.toggleRecording(mode: .transcription)
    await fixture.audio.emit([0.1])
    try await waitUntil { fixture.controller.warningMessage != nil }
    #expect(fixture.controller.warningMessage == "diarization disabled after a runtime error: injected")
    await fixture.controller.toggleRecording(mode: .transcription)
    #expect(fixture.controller.warningMessage == nil)

    fixture.controller.speakerNames[0] = "小明"
    await fixture.controller.toggleRecording(mode: .transcription)
    #expect(fixture.controller.messages.isEmpty)
    #expect(fixture.controller.speakerNames.isEmpty)
    #expect(fixture.controller.warningMessage == nil)
    await fixture.controller.toggleRecording(mode: .transcription)
}

@MainActor
@Test
func microphoneFailureLeavesRecordingOffAndCanBeRetried() async {
    let fixture = makeFixture()
    await fixture.audio.setStartError(.microphone)
    await fixture.controller.prepare()

    await fixture.controller.toggleRecording(mode: .transcription)

    #expect(fixture.controller.state == .failed("microphone failed"))
    #expect(fixture.controller.activeMode == nil)
    #expect(!fixture.controller.isRecording(.transcription))
    #expect(await fixture.audio.snapshot() == ["start", "stop"])
    #expect(await fixture.catcher.snapshot() == ["start", "finish"])
}

@MainActor
@Test
func clearTranscriptClearsOnlyWhenReady() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
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
    await fixture.controller.prepare()

    await fixture.controller.toggleRecording(mode: .transcription)
    await fixture.audio.emit([0.1])
    try await waitUntil { !fixture.controller.messages.isEmpty }
    fixture.controller.speakerNames[0] = "小明"

    fixture.controller.clearTranscript()
    #expect(!fixture.controller.messages.isEmpty)
    #expect(fixture.controller.speakerNames == [0: "小明"])

    await fixture.controller.toggleRecording(mode: .transcription)
    #expect(fixture.controller.state == .ready)
    #expect(fixture.controller.warningMessage != nil)

    fixture.controller.clearTranscript()
    #expect(fixture.controller.messages.isEmpty)
    #expect(fixture.controller.speakerNames.isEmpty)
    #expect(fixture.controller.warningMessage == nil)
    #expect(fixture.controller.state == .ready)
}

@MainActor
@Test
func transcriptionModeNeverStartsOrPushesKeywordSpotter() async throws {
    let fixture = makeFixture()
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .transcription)
    await fixture.audio.emit([0.1, 0.2])
    try await waitUntil { await fixture.catcher.snapshot().contains("push:2") }
    await fixture.controller.toggleRecording(mode: .transcription)

    #expect(await fixture.keywordSpotter.snapshot().isEmpty)
}

@MainActor
@Test
func voiceModeFansEachAudioChunkToAsrAndKws() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([0.1, 0.2])
    try await waitUntil { log.snapshot().contains("kws.push") }

    #expect(log.snapshot() == [
        "asr.start",
        "kws.start",
        "audio.start",
        "asr.push",
        "kws.push",
        "asr.textBefore:0",
    ])
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test(arguments: [UInt64(0), 320, 960])
func commandCutoffUsesSampleClockInsteadOfKwsTimestamp(startMs: UInt64) async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "你好", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: startMs)
    ])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { log.snapshot().contains("kws.reset") }

    #expect(log.snapshot().contains("asr.finishBefore:1000"))
    if startMs != 1_000 {
        #expect(!log.snapshot().contains("asr.finishBefore:\(startMs)"))
    }
    #expect(fixture.injector.injected == ["你好"])
    #expect(fixture.injector.submitCount == 1)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func heldTextBecomesInjectableAsSilenceAdvancesTheClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        pushes: [TranscriptUpdate(text: "你好", segments: [], warning: nil), nil],
        stableTexts: ["", "你好"]
    )
    await fixture.keywordSpotter.script([nil, nil])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:0") }
    #expect(fixture.injector.injected.isEmpty)

    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:500") }
    #expect(fixture.injector.injected == ["你好"])
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func commandResetStartsTheNextTurnAtZeroMilliseconds() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        pushes: [nil],
        stableTexts: [""],
        finishBefore: TranscriptUpdate(text: "第一輪", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 320),
        nil,
    ])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { fixture.injector.submitCount == 1 }
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:0") }

    #expect(log.snapshot().filter { $0 == "asr.finishBefore:1000" }.count == 1)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func commandInsideInitialHoldbackDoesNotPressReturn() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 960)
    ])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { fixture.controller.voiceInputMessage == "沒有可送出的文字" }

    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func streamFailureAndRetryResetTheSampleClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.keywordSpotter.failPush(with: .keyword)
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { fixture.controller.state == .failed("keyword failed") }

    await fixture.keywordSpotter.clearPushError()
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 320)
    ])
    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "", segments: [], warning: nil)
    )
    await fixture.controller.prepare()
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.finishBefore:0") }

    #expect(fixture.injector.submitCount == 0)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func stopAndRestartResetTheSampleClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(pushes: [nil], stableTexts: [""])
    await fixture.keywordSpotter.script([nil])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:1000") }
    await fixture.controller.toggleRecording(mode: .voiceInput)

    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 960)
    ])
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.finishBefore:0") }

    #expect(fixture.injector.submitCount == 0)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func stableSnapshotFailureStopsVoiceInputWithoutSubmitting() async throws {
    let fixture = makeFixture()
    await fixture.catcher.failTextBefore(with: .inference)
    await fixture.keywordSpotter.script([nil])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { fixture.controller.state == .failed("inference failed") }

    #expect(fixture.controller.failedMode == .voiceInput)
    #expect(fixture.controller.activeMode == nil)
    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    #expect(await fixture.audio.snapshot() == ["start", "stop"])
}

@MainActor
@Test
func duplicateCommandDoesNotSubmitTwice() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
        pushes: [
            nil,
            nil,
            TranscriptUpdate(text: "第二輪", segments: [], warning: nil),
        ],
        stableTexts: ["第二輪"],
        finishBefore: TranscriptUpdate(text: "第一輪", segments: [], warning: nil)
    )
    let command = KeywordDetection(keyword: "TIPPI_GO", startMs: 960)
    await fixture.keywordSpotter.script([command, command, nil])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([0.1])
    try await waitUntil { fixture.injector.submitCount == 1 }
    await fixture.audio.emit([0.2])
    try await waitUntil { await fixture.keywordSpotter.snapshot().filter { $0 == "push" }.count == 2 }
    await fixture.audio.emit([0.3])
    try await waitUntil { fixture.injector.injected.count == 2 }

    #expect(fixture.injector.submitCount == 1)
    #expect(fixture.injector.injected == ["第一輪", "第二輪"])
    #expect(await fixture.catcher.snapshot().filter { $0 == "start" }.count == 3)
    #expect(await fixture.keywordSpotter.snapshot().filter { $0 == "reset" }.count == 2)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func voiceStopDoesNotInjectOrSubmitTrailingText() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
        pushes: [nil],
        finish: TranscriptUpdate(text: "尚未穩定的尾段", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([nil])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([0.1])
    try await waitUntil { await fixture.keywordSpotter.snapshot().contains("push") }
    await fixture.controller.toggleRecording(mode: .voiceInput)

    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    #expect(fixture.controller.state == .ready)
    #expect(fixture.controller.activeMode == nil)
    #expect(await fixture.catcher.snapshot() == ["start", "push:1", "textBefore:0", "finish"])
    #expect(await fixture.keywordSpotter.snapshot() == ["start", "push", "reset"])
}

@MainActor
@Test
func kwsFailureStopsAudioAndFailsClosed() async throws {
    let fixture = makeFixture()
    await fixture.keywordSpotter.failPush(with: .keyword)
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([0.1])
    try await waitUntil { fixture.controller.state == .failed("keyword failed") }

    #expect(fixture.controller.activeMode == nil)
    #expect(await fixture.audio.snapshot() == ["start", "stop"])
    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    #expect(!fixture.controller.canToggle(.transcription))
    #expect(!fixture.controller.canToggle(.voiceInput))
}

@MainActor
@Test
func otherTabsRecordingButtonIsDisabledByActiveMode() async {
    let fixture = makeFixture()
    await prepareVoiceFixture(fixture)

    #expect(fixture.controller.canToggle(.transcription))
    #expect(fixture.controller.canToggle(.voiceInput))
    await fixture.controller.toggleRecording(mode: .transcription)

    #expect(fixture.controller.isRecording(.transcription))
    #expect(!fixture.controller.isRecording(.voiceInput))
    #expect(fixture.controller.canToggle(.transcription))
    #expect(!fixture.controller.canToggle(.voiceInput))
    await fixture.controller.toggleRecording(mode: .voiceInput)
    #expect(fixture.controller.activeMode == .transcription)

    await fixture.controller.toggleRecording(mode: .transcription)
    #expect(fixture.controller.activeMode == nil)
}

@MainActor
@Test
func prepareRunsLegacyMigrationBeforeEachModelInstaller() async {
    let log = EventLog()
    let fixture = makeFixture(log: log)

    await fixture.controller.prepare()
    #expect(Array(log.snapshot().prefix(2)) == ["migrate", "model.install"])

    log.removeAll()
    await fixture.controller.prepareVoiceInput()
    #expect(log.snapshot().filter { $0 != "permission:false" } == [
        "migrate",
        "kws.install",
        "kws.factory:kws",
    ])
    #expect(fixture.controller.voiceInputPreparation == .ready)
}

@MainActor
@Test
func voicePreparationRefreshesAccessibilityAndTarget() async {
    let fixture = makeFixture(trusted: false)

    await fixture.controller.prepareVoiceInput()

    #expect(fixture.controller.voiceInputPreparation == .ready)
    #expect(!fixture.controller.accessibilityTrusted)
    #expect(fixture.controller.targetApplicationName == "TextEdit")
    fixture.injector.trusted = true
    fixture.controller.refreshAccessibility(prompt: true)
    #expect(fixture.controller.accessibilityTrusted)
}

@MainActor
@Test
func voiceModeCannotStartWithoutAccessibilityTrust() async {
    let fixture = makeFixture(trusted: false)
    await prepareVoiceFixture(fixture)

    #expect(!fixture.controller.canToggle(.voiceInput))
    await fixture.controller.toggleRecording(mode: .voiceInput)
    #expect(fixture.controller.state == .ready)
    #expect(fixture.controller.activeMode == nil)
    #expect(await fixture.audio.snapshot().isEmpty)
}

@MainActor
@Test
func tippiFrontmostFailsSafeAndRequiresCommandToBeRepeated() async throws {
    let fixture = makeFixture(
        target: TargetApplication(name: "Tippi", bundleIdentifier: tippiBundleIdentifier)
    )
    await fixture.catcher.script(
        pushes: [TranscriptUpdate(text: "不要打回自己", segments: [], warning: nil)],
        finishBefore: TranscriptUpdate(text: "不要送出", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        nil,
        KeywordDetection(keyword: "TIPPI_GO", startMs: 960),
    ])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([0.1])
    try await waitUntil { fixture.controller.voiceInputMessage == "請切到目標輸入框" }
    await fixture.audio.emit([0.2])
    try await waitUntil {
        fixture.controller.voiceInputMessage == "請切到目標輸入框後重說 Tippi Go"
    }

    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    #expect(fixture.controller.targetApplicationName == "Tippi")
    await fixture.controller.toggleRecording(mode: .voiceInput)
}

@MainActor
@Test
func failedKeywordStartCleansUpOwnershipAndStartedAsr() async {
    let fixture = makeFixture()
    await fixture.keywordSpotter.failStart(with: .keyword)
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)

    #expect(fixture.controller.state == .failed("keyword failed"))
    #expect(fixture.controller.activeMode == nil)
    #expect(await fixture.catcher.snapshot() == ["start", "finish"])
    #expect(await fixture.keywordSpotter.snapshot() == ["start", "reset"])
    #expect(await fixture.audio.snapshot() == ["stop"])
}

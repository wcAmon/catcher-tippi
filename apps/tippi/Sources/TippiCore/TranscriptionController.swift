import Foundation
import Observation

public typealias CatcherFactory = @Sendable (ModelBundle) async throws -> any CatcherServing

@MainActor
@Observable
public final class TranscriptionController {
    public private(set) var state: TippiState = .modelMissing
    public private(set) var messages: [Message] = []
    public var speakerNames: [Int: String] = [:]
    public private(set) var warningMessage: String?
    public private(set) var downloadProgress = 0.0
    public private(set) var activeMode: RecordingMode?
    public private(set) var failedMode: RecordingMode?
    public private(set) var voiceInputPreparation: VoiceInputPreparationState = .notPrepared
    public private(set) var accessibilityTrusted = false
    public private(set) var targetApplicationName: String?
    public private(set) var lastInjectedText = ""
    public private(set) var voiceInputMessage = "請先切到目標輸入框"

    public var isRecording: Bool { state == .recording }

    @ObservationIgnored private let modelInstaller: any ModelBundleInstalling
    @ObservationIgnored private let audio: any AudioRecording
    @ObservationIgnored private let catcherFactory: CatcherFactory
    @ObservationIgnored private let modelMigrator: any ModelDirectoryMigrating
    @ObservationIgnored private let keywordInstaller: any KeywordModelInstalling
    @ObservationIgnored private let keywordFactory: KeywordSpotterFactory
    @ObservationIgnored private let injectionCoordinator: TextInjectionCoordinator
    @ObservationIgnored private var catcher: (any CatcherServing)?
    @ObservationIgnored private var keywordSpotter: (any KeywordSpotting)?
    @ObservationIgnored private var audioContinuation: AsyncStream<[Float]>.Continuation?
    @ObservationIgnored private var audioTask: Task<Void, Never>?
    @ObservationIgnored private var suppressImmediateDuplicateCommand = false
    @ObservationIgnored private var receivedSampleCount: UInt64 = 0

    public init(
        modelInstaller: any ModelBundleInstalling,
        audio: any AudioRecording,
        catcherFactory: @escaping CatcherFactory,
        modelMigrator: any ModelDirectoryMigrating,
        keywordInstaller: any KeywordModelInstalling,
        keywordFactory: @escaping KeywordSpotterFactory,
        injectionCoordinator: TextInjectionCoordinator
    ) {
        self.modelInstaller = modelInstaller
        self.audio = audio
        self.catcherFactory = catcherFactory
        self.modelMigrator = modelMigrator
        self.keywordInstaller = keywordInstaller
        self.keywordFactory = keywordFactory
        self.injectionCoordinator = injectionCoordinator
    }

    public func prepare() async {
        failedMode = nil
        if case .failed = state { state = .modelMissing }
        state = .downloading(0)
        do {
            try await modelMigrator.migrateIfNeeded()
            let bundle = try await modelInstaller.installIfNeeded { [weak self] progress in
                Task { @MainActor in
                    self?.updateDownloadProgress(progress)
                }
            }
            state = .loading
            catcher = try await catcherFactory(bundle)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    public func prepareVoiceInput() async {
        switch voiceInputPreparation {
        case .notPrepared, .failed:
            break
        case .downloading, .loading, .ready:
            return
        }

        voiceInputPreparation = .downloading(0)
        do {
            try await modelMigrator.migrateIfNeeded()
            let directory = try await keywordInstaller.installIfNeeded { [weak self] value in
                Task { @MainActor in
                    self?.updateVoiceInputDownloadProgress(value)
                }
            }
            voiceInputPreparation = .loading
            keywordSpotter = try await keywordFactory(directory)
            voiceInputPreparation = .ready
            refreshAccessibility(prompt: false)
        } catch {
            keywordSpotter = nil
            voiceInputPreparation = .failed(error.localizedDescription)
        }
    }

    public func refreshAccessibility(prompt: Bool) {
        accessibilityTrusted = injectionCoordinator.isTrusted(prompt: prompt)
        targetApplicationName = injectionCoordinator.currentTarget()?.name
    }

    public func toggleRecording(mode: RecordingMode) async {
        switch state {
        case .ready:
            guard activeMode == nil else { return }
            if mode == .voiceInput {
                guard voiceInputPreparation == .ready, accessibilityTrusted else { return }
            }
            await startRecording(mode: mode)
        case .recording where activeMode == mode:
            await stopRecording()
        default:
            break
        }
    }

    public func isRecording(_ mode: RecordingMode) -> Bool {
        state == .recording && activeMode == mode
    }

    public func canToggle(_ mode: RecordingMode) -> Bool {
        if state == .recording { return activeMode == mode }
        guard state == .ready, activeMode == nil else { return false }
        return mode == .transcription
            || (voiceInputPreparation == .ready && accessibilityTrusted)
    }

    /// Wipes the finished transcript so the user can start over without
    /// recording. No-op unless idle: recording keeps its live transcript.
    public func clearTranscript() {
        guard state == .ready else { return }
        messages = []
        speakerNames = [:]
        warningMessage = nil
    }

    private func updateDownloadProgress(_ progress: Double) {
        guard case .downloading = state else { return }
        downloadProgress = progress
        state = .downloading(progress)
    }

    private func updateVoiceInputDownloadProgress(_ progress: Double) {
        guard case .downloading = voiceInputPreparation else { return }
        voiceInputPreparation = .downloading(progress)
    }

    private func startRecording(mode: RecordingMode) async {
        guard let catcher else { return }
        failedMode = nil
        let spotter: (any KeywordSpotting)?
        switch mode {
        case .transcription:
            spotter = nil
        case .voiceInput:
            guard let keywordSpotter else { return }
            spotter = keywordSpotter
        }

        activeMode = mode
        suppressImmediateDuplicateCommand = false
        var catcherStarted = false
        do {
            try await catcher.start()
            catcherStarted = true
            if let spotter {
                try await spotter.start()
            }

            switch mode {
            case .transcription:
                messages = []
                speakerNames = [:]
                warningMessage = nil
            case .voiceInput:
                lastInjectedText = ""
                voiceInputMessage = "請切到目標輸入框"
                resetVoiceTurn()
                targetApplicationName = injectionCoordinator.currentTarget()?.name
            }

            let (stream, continuation) = AsyncStream<[Float]>.makeStream()
            audioContinuation = continuation
            audioTask = Task { @MainActor [weak self, catcher, spotter] in
                for await samples in stream {
                    guard !Task.isCancelled else { return }
                    do {
                        switch mode {
                        case .transcription:
                            if let update = try await catcher.push(samples) {
                                self?.apply(update)
                            }
                        case .voiceInput:
                            guard let self, let spotter else { return }
                            try await self.processVoiceInput(
                                samples,
                                catcher: catcher,
                                keywordSpotter: spotter
                            )
                        }
                    } catch {
                        guard let self else { return }
                        await self.handleStreamFailure(
                            error,
                            catcher: catcher,
                            keywordSpotter: spotter
                        )
                        return
                    }
                }
            }
            try await audio.start { samples in continuation.yield(samples) }
            state = .recording
        } catch {
            await cleanupAfterStartFailure(
                error,
                catcher: catcher,
                catcherStarted: catcherStarted,
                keywordSpotter: spotter
            )
        }
    }

    private func processVoiceInput(
        _ samples: [Float],
        catcher: any CatcherServing,
        keywordSpotter: any KeywordSpotting
    ) async throws {
        receivedSampleCount += UInt64(samples.count)
        _ = try await catcher.push(samples)
        let detection = try await keywordSpotter.push(samples)
        let cutoffMs = VoiceInputTiming.stableCutoffMs(
            receivedSampleCount: receivedSampleCount
        )

        if let detection, detection.keyword == VoiceSubmitCommand.eventIdentifier {
            if suppressImmediateDuplicateCommand {
                // A detector can surface the same buffered command immediately after
                // reset. Discard that audio from both engines without another Return.
                try await catcher.start()
                try await keywordSpotter.reset()
                resetVoiceTurn()
                return
            }

            let final = try await catcher.finish(before: cutoffMs)
            let event = try injectionCoordinator.submit(final.text)
            applyInjectionEvent(event)
            if event == .waitingForTarget {
                voiceInputMessage = "請切到目標輸入框後重說「\(VoiceSubmitCommand.displayPhrase)」"
            }
            try await catcher.start()
            try await keywordSpotter.reset()
            resetVoiceTurn()
            suppressImmediateDuplicateCommand = true
            return
        }

        suppressImmediateDuplicateCommand = false
        let stableText = try await catcher.text(before: cutoffMs)
        applyInjectionEvent(try injectionCoordinator.consume(stableText))
    }

    private func stopRecording() async {
        guard let catcher, let mode = activeMode else { return }
        state = .finishing
        await audio.stop()
        audioContinuation?.finish()
        audioContinuation = nil
        let task = audioTask
        audioTask = nil
        await task?.value
        if case .failed = state { return }

        do {
            switch mode {
            case .transcription:
                apply(try await catcher.finish())
            case .voiceInput:
                _ = try await catcher.finish()
                try await keywordSpotter?.reset()
                resetVoiceTurn()
            }
            suppressImmediateDuplicateCommand = false
            activeMode = nil
            state = .ready
        } catch {
            failedMode = mode
            if mode == .voiceInput {
                try? await keywordSpotter?.reset()
                resetVoiceTurn()
            }
            suppressImmediateDuplicateCommand = false
            activeMode = nil
            state = .failed(error.localizedDescription)
        }
    }

    private func apply(_ update: TranscriptUpdate) {
        messages = update.segments.enumerated().map { index, segment in
            Message(id: index, segment: segment)
        }
        warningMessage = update.warning
    }

    private func applyInjectionEvent(_ event: TextInjectionEvent) {
        switch event {
        case .noChange:
            break
        case .waitingForTarget:
            targetApplicationName = "Tippi"
            voiceInputMessage = "請切到目標輸入框"
        case let .injected(text, target):
            targetApplicationName = target
            lastInjectedText = text
            voiceInputMessage = "已嘗試注入至 \(target)"
        case let .submitted(text, target):
            targetApplicationName = target
            if !text.isEmpty { lastInjectedText = text }
            voiceInputMessage = "已送出"
        case .nothingToSubmit:
            lastInjectedText = ""
            voiceInputMessage = "沒有可送出的文字"
        case .duplicateCommandIgnored:
            break
        }
    }

    private func cleanupAfterStartFailure(
        _ error: any Error,
        catcher: any CatcherServing,
        catcherStarted: Bool,
        keywordSpotter: (any KeywordSpotting)?
    ) async {
        let failedRecordingMode = activeMode
        await audio.stop()
        audioContinuation?.finish()
        audioContinuation = nil
        let task = audioTask
        audioTask = nil
        task?.cancel()
        await task?.value
        if catcherStarted {
            _ = try? await catcher.finish()
        }
        try? await keywordSpotter?.reset()
        if failedRecordingMode == .voiceInput {
            resetVoiceTurn()
        } else {
            injectionCoordinator.resetTurn()
        }
        suppressImmediateDuplicateCommand = false
        activeMode = nil
        failedMode = failedRecordingMode
        state = .failed(error.localizedDescription)
    }

    private func handleStreamFailure(
        _ error: any Error,
        catcher: any CatcherServing,
        keywordSpotter: (any KeywordSpotting)?
    ) async {
        let failedRecordingMode = activeMode
        state = .finishing
        await audio.stop()
        audioContinuation?.finish()
        audioContinuation = nil
        audioTask?.cancel()
        audioTask = nil
        _ = try? await catcher.finish()
        try? await keywordSpotter?.reset()
        if failedRecordingMode == .voiceInput {
            resetVoiceTurn()
        } else {
            injectionCoordinator.resetTurn()
        }
        suppressImmediateDuplicateCommand = false
        activeMode = nil
        failedMode = failedRecordingMode
        state = .failed(error.localizedDescription)
    }

    private func resetVoiceTurn() {
        receivedSampleCount = 0
        injectionCoordinator.resetTurn()
    }
}

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
    public var isRecording: Bool { state == .recording }

    @ObservationIgnored private let modelInstaller: any ModelBundleInstalling
    @ObservationIgnored private let audio: any AudioRecording
    @ObservationIgnored private let catcherFactory: CatcherFactory
    @ObservationIgnored private var catcher: (any CatcherServing)?
    @ObservationIgnored private var audioContinuation: AsyncStream<[Float]>.Continuation?
    @ObservationIgnored private var audioTask: Task<Void, Never>?

    public init(
        modelInstaller: any ModelBundleInstalling,
        audio: any AudioRecording,
        catcherFactory: @escaping CatcherFactory
    ) {
        self.modelInstaller = modelInstaller
        self.audio = audio
        self.catcherFactory = catcherFactory
    }

    public func prepare() async {
        if case .failed = state { state = .modelMissing }
        state = .downloading(0)
        do {
            let bundle = try await modelInstaller.installIfNeeded { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    self?.state = .downloading(progress)
                }
            }
            state = .loading
            catcher = try await catcherFactory(bundle)
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    public func toggleRecording() async {
        switch state {
        case .ready: await startRecording()
        case .recording: await stopRecording()
        default: break
        }
    }

    private func startRecording() async {
        guard let catcher else { return }
        do {
            try await catcher.start()
            messages = []
            speakerNames = [:]
            warningMessage = nil
            let (stream, continuation) = AsyncStream<[Float]>.makeStream()
            audioContinuation = continuation
            audioTask = Task { [weak self] in
                for await samples in stream {
                    do {
                        if let update = try await catcher.push(samples) {
                            self?.apply(update)
                        }
                    } catch {
                        await self?.handleStreamFailure(error)
                        return
                    }
                }
            }
            try await audio.start { samples in continuation.yield(samples) }
            state = .recording
        } catch {
            audioContinuation?.finish()
            audioContinuation = nil
            audioTask?.cancel()
            audioTask = nil
            state = .failed(error.localizedDescription)
        }
    }

    private func stopRecording() async {
        guard let catcher else { return }
        state = .finishing
        await audio.stop()
        audioContinuation?.finish()
        audioContinuation = nil
        await audioTask?.value
        audioTask = nil
        if case .failed = state { return }
        do {
            apply(try await catcher.finish())
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func apply(_ update: TranscriptUpdate) {
        messages = update.segments.enumerated().map { index, segment in
            Message(id: index, segment: segment)
        }
        warningMessage = update.warning
    }

    private func handleStreamFailure(_ error: any Error) async {
        await audio.stop()
        audioContinuation?.finish()
        audioContinuation = nil
        state = .failed(error.localizedDescription)
    }
}

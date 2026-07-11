import Foundation
import Observation

public typealias CatcherFactory = @Sendable (URL) async throws -> any CatcherServing

@MainActor
@Observable
public final class TranscriptionController {
    public private(set) var state: TippiState = .modelMissing
    public private(set) var text = ""
    public private(set) var downloadProgress = 0.0
    public var isRecording: Bool { state == .recording }

    @ObservationIgnored private let modelInstaller: any ModelInstalling
    @ObservationIgnored private let audio: any AudioRecording
    @ObservationIgnored private let catcherFactory: CatcherFactory
    @ObservationIgnored private var catcher: (any CatcherServing)?
    @ObservationIgnored private var audioContinuation: AsyncStream<[Float]>.Continuation?
    @ObservationIgnored private var audioTask: Task<Void, Never>?

    public init(
        modelInstaller: any ModelInstalling,
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
            let modelURL = try await modelInstaller.installIfNeeded { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    self?.state = .downloading(progress)
                }
            }
            state = .loading
            catcher = try await catcherFactory(modelURL)
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
            text = ""
            let (stream, continuation) = AsyncStream<[Float]>.makeStream()
            audioContinuation = continuation
            audioTask = Task { [weak self] in
                for await samples in stream {
                    do {
                        if let partial = try await catcher.push(samples) {
                            self?.text = partial
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
            text = try await catcher.finish()
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func handleStreamFailure(_ error: any Error) async {
        await audio.stop()
        audioContinuation?.finish()
        audioContinuation = nil
        state = .failed(error.localizedDescription)
    }
}

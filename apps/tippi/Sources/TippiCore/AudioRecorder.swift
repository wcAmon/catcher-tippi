@preconcurrency import AVFoundation
import Foundation

public protocol AudioRecording: Sendable {
    func start(onSamples: @escaping @Sendable ([Float]) -> Void) async throws
    func stop() async
}

public enum AudioRecorderError: Error, LocalizedError {
    case microphoneDenied
    case unsupportedInputFormat
    case converterCreationFailed
    case conversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .microphoneDenied: "Microphone access is disabled in System Settings."
        case .unsupportedInputFormat: "The selected microphone format is not supported."
        case .converterCreationFailed: "Could not create the 16 kHz audio converter."
        case let .conversionFailed(message): "Microphone conversion failed: \(message)"
        }
    }
}

public final class AudioRecorder: AudioRecording, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?
    private var sink: (@Sendable ([Float]) -> Void)?

    public init() {}

    public func start(onSamples: @escaping @Sendable ([Float]) -> Void) async throws {
        guard await requestMicrophonePermission() else {
            throw AudioRecorderError.microphoneDenied
        }
        let input = engine.inputNode
        let inputFormat = input.inputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0,
              let target = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: 16_000,
                  channels: 1,
                  interleaved: false
              )
        else {
            throw AudioRecorderError.unsupportedInputFormat
        }
        guard let converter = AVAudioConverter(from: inputFormat, to: target) else {
            throw AudioRecorderError.converterCreationFailed
        }

        lock.withLock {
            self.converter = converter
            targetFormat = target
            sink = onSamples
        }
        input.installTap(onBus: 0, bufferSize: 2_048, format: inputFormat) { [weak self] buffer, _ in
            self?.consume(buffer, inputFormat: inputFormat)
        }
        engine.prepare()
        try engine.start()
    }

    public func stop() async {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        lock.withLock {
            converter = nil
            targetFormat = nil
            sink = nil
        }
    }

    private func consume(_ input: AVAudioPCMBuffer, inputFormat: AVAudioFormat) {
        let state = lock.withLock { (converter, targetFormat, sink) }
        guard let converter = state.0, let target = state.1, let sink = state.2 else { return }
        let ratio = target.sampleRate / inputFormat.sampleRate
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * ratio) + 32)
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
        let conversionInput = ConversionInput(buffer: input)
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
            conversionInput.next(status: inputStatus)
        }
        guard status != .error, conversionError == nil,
              let channel = output.floatChannelData?.pointee
        else {
            return
        }
        sink(Array(UnsafeBufferPointer(start: channel, count: Int(output.frameLength))))
    }

    private func requestMicrophonePermission() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: true
        case .notDetermined: await AVCaptureDevice.requestAccess(for: .audio)
        default: false
        }
    }
}

private final class ConversionInput: @unchecked Sendable {
    private let buffer: AVAudioPCMBuffer
    private let lock = NSLock()
    private var supplied = false

    init(buffer: AVAudioPCMBuffer) {
        self.buffer = buffer
    }

    func next(status: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
        lock.withLock {
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
    }
}

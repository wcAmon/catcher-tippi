import CCatcher
import Foundation

public protocol CatcherServing: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> TranscriptUpdate?
    func text(before cutoffMs: UInt64) async throws -> String
    func finish() async throws -> TranscriptUpdate
    func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate
}

public enum CatcherClientError: Error, LocalizedError {
    case creationFailed(String)
    case operationFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .creationFailed(message): "Could not load Catcher: \(message)"
        case let .operationFailed(message): "Catcher inference failed: \(message)"
        }
    }
}

private final class CatcherHandleOwner: @unchecked Sendable {
    let pointer: OpaquePointer
    init(pointer: OpaquePointer) { self.pointer = pointer }
    deinit { catcher_destroy(pointer) }
}

public actor CatcherClient: CatcherServing {
    private let owner: CatcherHandleOwner

    public init(
        modelDirectory: URL,
        diarModelDirectory: URL,
        language: String = "auto",
        lookahead: UInt32 = 3
    ) throws {
        let pointer = modelDirectory.path.withCString { modelPath in
            diarModelDirectory.path.withCString { diarPath in
                language.withCString { languageCode in
                    catcher_create(modelPath, diarPath, languageCode, lookahead)
                }
            }
        }
        guard let pointer else {
            throw CatcherClientError.creationFailed(Self.globalError())
        }
        owner = CatcherHandleOwner(pointer: pointer)
    }

    public func start() async throws {
        try check(catcher_start(owner.pointer), allowNoUpdate: false)
    }

    public func push(_ samples: [Float]) async throws -> TranscriptUpdate? {
        let status = samples.withUnsafeBufferPointer { buffer in
            catcher_push_audio(owner.pointer, buffer.baseAddress, buffer.count)
        }
        if status == CATCHER_NO_UPDATE { return nil }
        try check(status, allowNoUpdate: false)
        return try currentUpdate()
    }

    public func text(before cutoffMs: UInt64) async throws -> String {
        guard let pointer = catcher_text_before(owner.pointer, cutoffMs) else {
            throw CatcherClientError.operationFailed(currentError())
        }
        return String(cString: pointer)
    }

    public func finish() async throws -> TranscriptUpdate {
        try check(catcher_finish(owner.pointer), allowNoUpdate: true)
        return try currentUpdate()
    }

    public func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate {
        try check(catcher_finish_before(owner.pointer, cutoffMs), allowNoUpdate: true)
        return try currentUpdate()
    }

    private func currentUpdate() throws -> TranscriptUpdate {
        let text = catcher_text(owner.pointer).map { String(cString: $0) } ?? ""
        let json = catcher_segments(owner.pointer).map { String(cString: $0) } ?? "[]"
        let segments: [SpeakerSegment]
        do {
            segments = try SpeakerSegment.decodeArray(from: json)
        } catch {
            throw CatcherClientError.operationFailed("segments JSON decode failed: \(error)")
        }
        let warning = catcher_warning(owner.pointer).map { String(cString: $0) }
        return TranscriptUpdate(text: text, segments: segments, warning: warning)
    }

    private func check(_ status: Int32, allowNoUpdate: Bool) throws {
        if status == CATCHER_OK || (allowNoUpdate && status == CATCHER_NO_UPDATE) { return }
        throw CatcherClientError.operationFailed(currentError())
    }

    private func currentError() -> String {
        guard let pointer = catcher_last_error(owner.pointer) else { return "unknown error" }
        return String(cString: pointer)
    }

    private static func globalError() -> String {
        guard let pointer = catcher_last_error(nil) else { return "unknown error" }
        return String(cString: pointer)
    }
}

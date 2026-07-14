import CCatcher
import Foundation

public struct KeywordDetection: Equatable, Sendable {
    public let keyword: String
    public let startMs: UInt64

    public init(keyword: String, startMs: UInt64) {
        self.keyword = keyword
        self.startMs = startMs
    }
}

public protocol KeywordSpotting: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> KeywordDetection?
    func reset() async throws
}

public typealias KeywordSpotterFactory =
    @Sendable (URL) async throws -> any KeywordSpotting

public enum KeywordSpotterClientError: Error, LocalizedError {
    case creationFailed(String)
    case operationFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .creationFailed(message):
            "無法載入「\(VoiceSubmitCommand.displayPhrase)」口令模型：\(message)"
        case let .operationFailed(message):
            "「\(VoiceSubmitCommand.displayPhrase)」口令偵測失敗：\(message)"
        }
    }
}

private final class KeywordHandleOwner: @unchecked Sendable {
    let pointer: OpaquePointer

    init(pointer: OpaquePointer) {
        self.pointer = pointer
    }

    deinit {
        catcher_kws_destroy(pointer)
    }
}

public actor KeywordSpotterClient: KeywordSpotting {
    private let owner: KeywordHandleOwner

    public init(modelDirectory: URL) throws {
        let pointer = modelDirectory.path.withCString { catcher_kws_create($0) }
        guard let pointer else {
            throw KeywordSpotterClientError.creationFailed(Self.globalError())
        }
        owner = KeywordHandleOwner(pointer: pointer)
    }

    public func start() async throws {
        try check(catcher_kws_start(owner.pointer))
    }

    public func push(_ samples: [Float]) async throws -> KeywordDetection? {
        let status = samples.withUnsafeBufferPointer { buffer in
            catcher_kws_push_audio(owner.pointer, buffer.baseAddress, buffer.count)
        }
        if status == CATCHER_NO_UPDATE { return nil }
        guard status == CATCHER_COMMAND_DETECTED else {
            throw KeywordSpotterClientError.operationFailed(currentError())
        }
        guard let keyword = catcher_kws_keyword(owner.pointer) else {
            throw KeywordSpotterClientError.operationFailed(
                "detected a command without a keyword"
            )
        }
        return KeywordDetection(
            keyword: String(cString: keyword),
            startMs: catcher_kws_start_ms(owner.pointer)
        )
    }

    public func reset() async throws {
        try check(catcher_kws_start(owner.pointer))
    }

    private func check(_ status: Int32) throws {
        guard status == CATCHER_OK else {
            throw KeywordSpotterClientError.operationFailed(currentError())
        }
    }

    private func currentError() -> String {
        guard let pointer = catcher_kws_last_error(owner.pointer) else {
            return "unknown KWS error"
        }
        return String(cString: pointer)
    }

    private nonisolated static func globalError() -> String {
        guard let pointer = catcher_kws_last_error(nil) else {
            return "unknown KWS error"
        }
        return String(cString: pointer)
    }
}

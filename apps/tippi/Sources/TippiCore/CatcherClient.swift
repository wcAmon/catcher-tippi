import CCatcher
import Foundation

public protocol CatcherServing: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> String?
    func finish() async throws -> String
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

    public init(modelDirectory: URL, language: String = "auto", lookahead: UInt32 = 3) throws {
        let pointer = modelDirectory.path.withCString { modelPath in
            language.withCString { languageCode in
                catcher_create(modelPath, languageCode, lookahead)
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

    public func push(_ samples: [Float]) async throws -> String? {
        let status = samples.withUnsafeBufferPointer { buffer in
            catcher_push_audio(owner.pointer, buffer.baseAddress, buffer.count)
        }
        if status == CATCHER_NO_UPDATE { return nil }
        try check(status, allowNoUpdate: false)
        return currentText()
    }

    public func finish() async throws -> String {
        try check(catcher_finish(owner.pointer), allowNoUpdate: true)
        return currentText()
    }

    private func check(_ status: Int32, allowNoUpdate: Bool) throws {
        if status == CATCHER_OK || (allowNoUpdate && status == CATCHER_NO_UPDATE) { return }
        throw CatcherClientError.operationFailed(currentError())
    }

    private func currentText() -> String {
        guard let pointer = catcher_text(owner.pointer) else { return "" }
        return String(cString: pointer)
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

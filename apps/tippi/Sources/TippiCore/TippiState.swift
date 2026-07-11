public enum TippiState: Equatable, Sendable {
    case modelMissing
    case downloading(Double)
    case loading
    case ready
    case recording
    case finishing
    case failed(String)
}

public enum TippiStateError: Error, Equatable {
    case invalidTransition(from: TippiState, to: TippiState)
}

public struct TippiStateMachine: Sendable {
    public private(set) var state: TippiState

    public init(initial: TippiState = .modelMissing) {
        state = initial
    }

    public mutating func transition(to next: TippiState) throws {
        guard Self.allows(state, next) else {
            throw TippiStateError.invalidTransition(from: state, to: next)
        }
        state = next
    }

    private static func allows(_ current: TippiState, _ next: TippiState) -> Bool {
        switch (current, next) {
        case (.modelMissing, .downloading),
             (.downloading, .downloading),
             (.downloading, .loading),
             (.loading, .ready),
             (.ready, .recording),
             (.recording, .finishing),
             (.finishing, .ready),
             (.failed, .modelMissing),
             (.failed, .ready):
            true
        case (_, .failed):
            true
        default:
            false
        }
    }
}

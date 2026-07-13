import Foundation

public enum TextInjectionEvent: Equatable, Sendable {
    case noChange
    case waitingForTarget
    case injected(text: String, target: String)
    case submitted(text: String, target: String)
    case duplicateCommandIgnored
}

public enum TextInjectionError: Error, LocalizedError, Equatable, Sendable {
    case divergentPrefix(previous: String, current: String)
    case eventCreationFailed

    public var errorDescription: String? {
        switch self {
        case let .divergentPrefix(previous, current):
            "Transcription stopped being append-only (previous: \(previous), current: \(current))."
        case .eventCreationFailed:
            "Could not create a keyboard event."
        }
    }
}

@MainActor
public final class TextInjectionCoordinator {
    private let injector: any TextInjecting
    private let targetProvider: any FrontmostApplicationProviding
    private let ownBundleIdentifier: String

    private var injectedPrefix = ""
    private var commandInFlight = false

    public init(
        injector: any TextInjecting,
        targetProvider: any FrontmostApplicationProviding,
        ownBundleIdentifier: String
    ) {
        self.injector = injector
        self.targetProvider = targetProvider
        self.ownBundleIdentifier = ownBundleIdentifier
    }

    public func isTrusted(prompt: Bool) -> Bool {
        injector.isTrusted(prompt: prompt)
    }

    public func currentTarget() -> TargetApplication? {
        targetProvider.current()
    }

    public func consume(_ fullText: String) throws -> TextInjectionEvent {
        guard let target = injectableTarget() else {
            return .waitingForTarget
        }
        let suffix = try pendingSuffix(in: fullText)
        guard !suffix.isEmpty else {
            return .noChange
        }

        try injector.inject(suffix)
        injectedPrefix = fullText
        return .injected(text: suffix, target: target.name)
    }

    public func submit(_ fullText: String) throws -> TextInjectionEvent {
        guard !commandInFlight else {
            return .duplicateCommandIgnored
        }
        guard let target = injectableTarget() else {
            return .waitingForTarget
        }

        let suffix = try pendingSuffix(in: fullText)
        if !suffix.isEmpty {
            try injector.inject(suffix)
            injectedPrefix = fullText
        }
        try injector.submit()
        commandInFlight = true
        return .submitted(text: fullText, target: target.name)
    }

    public func resetTurn() {
        injectedPrefix = ""
        commandInFlight = false
    }

    private func injectableTarget() -> TargetApplication? {
        guard let target = targetProvider.current() else {
            return nil
        }
        guard target.bundleIdentifier != ownBundleIdentifier else {
            return nil
        }
        return target
    }

    private func pendingSuffix(in fullText: String) throws -> String {
        guard fullText.hasPrefix(injectedPrefix) else {
            throw TextInjectionError.divergentPrefix(
                previous: injectedPrefix,
                current: fullText
            )
        }
        return String(fullText.dropFirst(injectedPrefix.count))
    }
}

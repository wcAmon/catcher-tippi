import AppKit
@preconcurrency import ApplicationServices
import CoreGraphics
import Foundation

public struct TargetApplication: Equatable, Sendable {
    public let name: String
    public let bundleIdentifier: String?

    public init(name: String, bundleIdentifier: String?) {
        self.name = name
        self.bundleIdentifier = bundleIdentifier
    }
}

@MainActor
public protocol FrontmostApplicationProviding: AnyObject {
    func current() -> TargetApplication?
}

@MainActor
public protocol TextInjecting: AnyObject {
    func isTrusted(prompt: Bool) -> Bool
    func inject(_ text: String) throws
    func submit() throws
}

@MainActor
public final class CGEventTextInjector: TextInjecting {
    public init() {}

    public func isTrusted(prompt: Bool) -> Bool {
        let options = [
            kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: prompt
        ] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    public func inject(_ text: String) throws {
        guard !text.isEmpty else { return }
        guard
            let down = CGEvent(
                keyboardEventSource: nil,
                virtualKey: CGKeyCode(0),
                keyDown: true
            ),
            let up = CGEvent(
                keyboardEventSource: nil,
                virtualKey: CGKeyCode(0),
                keyDown: false
            )
        else {
            throw TextInjectionError.eventCreationFailed
        }

        let units = Array(text.utf16)
        units.withUnsafeBufferPointer { buffer in
            down.keyboardSetUnicodeString(
                stringLength: buffer.count,
                unicodeString: buffer.baseAddress
            )
            up.keyboardSetUnicodeString(
                stringLength: buffer.count,
                unicodeString: buffer.baseAddress
            )
        }
        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }

    public func submit() throws {
        guard
            let down = CGEvent(
                keyboardEventSource: nil,
                virtualKey: CGKeyCode(36),
                keyDown: true
            ),
            let up = CGEvent(
                keyboardEventSource: nil,
                virtualKey: CGKeyCode(36),
                keyDown: false
            )
        else {
            throw TextInjectionError.eventCreationFailed
        }

        down.post(tap: .cghidEventTap)
        up.post(tap: .cghidEventTap)
    }
}

@MainActor
public final class FrontmostApplicationProvider: FrontmostApplicationProviding {
    public init() {}

    public func current() -> TargetApplication? {
        guard let application = NSWorkspace.shared.frontmostApplication else {
            return nil
        }
        let bundleIdentifier = application.bundleIdentifier
        return TargetApplication(
            name: application.localizedName ?? bundleIdentifier ?? "Unknown App",
            bundleIdentifier: bundleIdentifier
        )
    }
}

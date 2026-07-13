import Foundation

/// One speaker-attributed run of transcript text, decoded from the
/// `catcher_segments` JSON produced by the Rust fusion module. Field names
/// mirror the Rust `SpeakerSegment` serde output exactly.
public struct SpeakerSegment: Codable, Equatable, Sendable {
    public let speaker: Int
    public let startMs: UInt64
    public let endMs: UInt64
    public let text: String
    public let isFinal: Bool

    enum CodingKeys: String, CodingKey {
        case speaker
        case startMs = "start_ms"
        case endMs = "end_ms"
        case text
        case isFinal = "final"
    }

    public init(speaker: Int, startMs: UInt64, endMs: UInt64, text: String, isFinal: Bool) {
        self.speaker = speaker
        self.startMs = startMs
        self.endMs = endMs
        self.text = text
        self.isFinal = isFinal
    }

    public static func decodeArray(from json: String) throws -> [SpeakerSegment] {
        try JSONDecoder().decode([SpeakerSegment].self, from: Data(json.utf8))
    }
}

/// What one successful push/finish call reports back to the UI layer.
public struct TranscriptUpdate: Equatable, Sendable {
    public let text: String
    public let segments: [SpeakerSegment]
    public let warning: String?

    public init(text: String, segments: [SpeakerSegment], warning: String?) {
        self.text = text
        self.segments = segments
        self.warning = warning
    }
}

/// One row in Tippi's message list. `id` is the row's index; the whole list
/// is rebuilt from segments on every update.
public struct Message: Identifiable, Equatable, Sendable {
    public let id: Int
    public let speaker: Int
    public let startMs: UInt64
    public let endMs: UInt64
    public let text: String
    public let isFinal: Bool

    public init(id: Int, speaker: Int, startMs: UInt64, endMs: UInt64, text: String, isFinal: Bool) {
        self.id = id
        self.speaker = speaker
        self.startMs = startMs
        self.endMs = endMs
        self.text = text
        self.isFinal = isFinal
    }

    public init(id: Int, segment: SpeakerSegment) {
        self.init(
            id: id,
            speaker: segment.speaker,
            startMs: segment.startMs,
            endMs: segment.endMs,
            text: segment.text,
            isFinal: segment.isFinal
        )
    }
}

/// Shared line formatting for on-screen copy actions and file export, so the
/// two never drift apart. Line shape: `[mm:ss] 顯示名:內文` (fullwidth
/// colon; minutes grow past two digits naturally).
public enum TranscriptFormatter {
    public static func displayName(for speaker: Int, names: [Int: String]) -> String {
        names[speaker] ?? "說話者 \(speaker + 1)"
    }

    public static func timestamp(forMs milliseconds: UInt64) -> String {
        let totalSeconds = milliseconds / 1000
        return String(format: "[%02d:%02d]", Int(totalSeconds / 60), Int(totalSeconds % 60))
    }

    public static func line(for message: Message, names: [Int: String]) -> String {
        "\(timestamp(forMs: message.startMs)) \(displayName(for: message.speaker, names: names))：\(message.text)"
    }

    public static func transcript(messages: [Message], names: [Int: String]) -> String {
        messages.map { line(for: $0, names: names) }.joined(separator: "\n")
    }
}

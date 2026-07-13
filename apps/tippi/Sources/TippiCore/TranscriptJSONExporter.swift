import Foundation

/// Serialises the message list for the `.json` export choice. Keys are
/// snake_case to match the FFI segments contract; output is pretty-printed
/// with sorted keys so golden tests can compare byte-for-byte.
public enum TranscriptJSONExporter {
    private struct ExportMessage: Encodable {
        let speaker: Int
        let name: String
        let startMs: UInt64
        let endMs: UInt64
        let text: String
        let isFinal: Bool

        enum CodingKeys: String, CodingKey {
            case speaker
            case name
            case startMs = "start_ms"
            case endMs = "end_ms"
            case text
            case isFinal = "final"
        }
    }

    private struct ExportDocument: Encodable {
        let messages: [ExportMessage]
    }

    public static func data(messages: [Message], names: [Int: String]) throws -> Data {
        let document = ExportDocument(messages: messages.map { message in
            ExportMessage(
                speaker: message.speaker,
                name: TranscriptFormatter.displayName(for: message.speaker, names: names),
                startMs: message.startMs,
                endMs: message.endMs,
                text: message.text,
                isFinal: message.isFinal
            )
        })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(document)
    }
}

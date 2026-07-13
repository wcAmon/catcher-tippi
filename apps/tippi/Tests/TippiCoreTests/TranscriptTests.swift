import Foundation
import Testing
@testable import TippiCore

@Test
func decodesRustSegmentJSON() throws {
    let json = """
    [{"speaker":0,"start_ms":400,"end_ms":2000,"text":"今天先討論這個。","final":true},
     {"speaker":1,"start_ms":2080,"end_ms":2400,"text":"好。","final":false}]
    """
    let segments = try SpeakerSegment.decodeArray(from: json)
    #expect(segments == [
        SpeakerSegment(speaker: 0, startMs: 400, endMs: 2000, text: "今天先討論這個。", isFinal: true),
        SpeakerSegment(speaker: 1, startMs: 2080, endMs: 2400, text: "好。", isFinal: false),
    ])
}

@Test
func decodeFailureThrows() {
    #expect(throws: (any Error).self) {
        _ = try SpeakerSegment.decodeArray(from: "not json")
    }
}

@Test
func formatsLinesWithNamesAndDefaults() {
    let named = Message(id: 0, speaker: 0, startMs: 204_000, text: "今天先討論這個。", isFinal: true)
    let unnamed = Message(id: 1, speaker: 1, startMs: 6_132_000, text: "好。", isFinal: true)
    let names = [0: "小明"]

    #expect(TranscriptFormatter.line(for: named, names: names) == "[03:24] 小明:今天先討論這個。")
    #expect(TranscriptFormatter.line(for: unnamed, names: names) == "[102:12] 說話者 2:好。")
    #expect(TranscriptFormatter.transcript(messages: [named, unnamed], names: names)
        == "[03:24] 小明:今天先討論這個。\n[102:12] 說話者 2:好。")
}

@Test
func messageIsBuiltFromSegment() {
    let segment = SpeakerSegment(speaker: 2, startMs: 80, endMs: 160, text: "喂?", isFinal: false)
    let message = Message(id: 5, segment: segment)
    #expect(message == Message(id: 5, speaker: 2, startMs: 80, text: "喂?", isFinal: false))
}

import Foundation
import Testing
@testable import TippiCore

@Test
func exportsMessagesAsSortedPrettyJSON() throws {
    let messages = [
        Message(id: 0, speaker: 0, startMs: 1234, endMs: 5678, text: "今天先討論這個。", isFinal: true),
        Message(id: 1, speaker: 1, startMs: 6000, endMs: 6400, text: "好。", isFinal: false),
    ]
    let data = try TranscriptJSONExporter.data(messages: messages, names: [0: "小明"])
    let expected = """
    {
      "messages" : [
        {
          "end_ms" : 5678,
          "final" : true,
          "name" : "小明",
          "speaker" : 0,
          "start_ms" : 1234,
          "text" : "今天先討論這個。"
        },
        {
          "end_ms" : 6400,
          "final" : false,
          "name" : "說話者 2",
          "speaker" : 1,
          "start_ms" : 6000,
          "text" : "好。"
        }
      ]
    }
    """
    #expect(String(decoding: data, as: UTF8.self) == expected)
}

@Test
func exportsEmptyMessageListAsEmptyArray() throws {
    let data = try TranscriptJSONExporter.data(messages: [], names: [:])
    let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    let messages = object?["messages"] as? [Any]
    #expect(messages?.isEmpty == true)
}

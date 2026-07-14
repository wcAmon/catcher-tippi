import Foundation
import Testing
@testable import TippiCore

private actor FakeKeywordSpotter: KeywordSpotting {
    func start() async throws {}

    func push(_ samples: [Float]) async throws -> KeywordDetection? {
        KeywordDetection(
            keyword: VoiceSubmitCommand.eventIdentifier,
            startMs: UInt64(samples.count)
        )
    }

    func reset() async throws {}
}

@Test
func chineseSubmitCommandContractIsExact() {
    #expect(VoiceSubmitCommand.displayPhrase == "幫我送出")
    #expect(VoiceSubmitCommand.eventIdentifier == "SUBMIT_ZH")
    #expect(VoiceSubmitCommand.tokenSequence == "b āng w ǒ s òng ch ū")
    #expect(VoiceSubmitCommand.keywordBoost == 1.5)
    #expect(VoiceSubmitCommand.triggerThreshold == 0.25)
    #expect(VoiceSubmitCommand.keywordDefinition
        == "b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH\n")
}

@Test
func keywordDetectionCarriesCommandAndCutoff() async throws {
    let service: any KeywordSpotting = FakeKeywordSpotter()
    let detection = try await service.push([0, 0, 0])
    #expect(detection == KeywordDetection(keyword: "SUBMIT_ZH", startMs: 3))
}

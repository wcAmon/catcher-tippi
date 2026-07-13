import Foundation
import Testing
@testable import TippiCore

private actor FakeKeywordSpotter: KeywordSpotting {
    func start() async throws {}

    func push(_ samples: [Float]) async throws -> KeywordDetection? {
        KeywordDetection(keyword: "TIPPI_GO", startMs: UInt64(samples.count))
    }

    func reset() async throws {}
}

@Test
func keywordDetectionCarriesCommandAndCutoff() async throws {
    let service: any KeywordSpotting = FakeKeywordSpotter()
    let detection = try await service.push([0, 0, 0])
    #expect(detection == KeywordDetection(keyword: "TIPPI_GO", startMs: 3))
}

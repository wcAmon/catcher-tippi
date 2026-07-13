import Testing
@testable import TippiCore

private actor SnapshotCatcher: CatcherServing {
    func start() async throws {}
    func push(_ samples: [Float]) async throws -> TranscriptUpdate? { nil }
    func finish() async throws -> TranscriptUpdate {
        TranscriptUpdate(text: "", segments: [], warning: nil)
    }
    func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate {
        TranscriptUpdate(text: "", segments: [], warning: nil)
    }
    func text(before cutoffMs: UInt64) async throws -> String {
        "stable:\(cutoffMs)"
    }
}

@Test
func catcherServingExposesStableSnapshot() async throws {
    let service: any CatcherServing = SnapshotCatcher()
    #expect(try await service.text(before: 1_500) == "stable:1500")
}

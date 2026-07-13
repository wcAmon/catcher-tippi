public enum VoiceInputTiming {
    public static let sampleRate: UInt64 = 16_000
    public static let holdbackMs: UInt64 = 1_500

    public static func stableCutoffMs(receivedSampleCount: UInt64) -> UInt64 {
        let audioEndMs = receivedSampleCount * 1_000 / sampleRate
        return audioEndMs > holdbackMs ? audioEndMs - holdbackMs : 0
    }
}

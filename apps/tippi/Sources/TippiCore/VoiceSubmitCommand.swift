public enum VoiceSubmitCommand {
    public static let displayPhrase = "幫我送出"
    public static let eventIdentifier = "SUBMIT_ZH"
    public static let tokenSequence = "b āng w ǒ s òng ch ū"
    public static let keywordBoost = 1.5
    public static let triggerThreshold = 0.25

    public static var keywordDefinition: String {
        "\(tokenSequence) :\(keywordBoost) #\(triggerThreshold) @\(eventIdentifier)\n"
    }
}

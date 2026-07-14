import Foundation

public struct KeywordModelArchive: Sendable {
    public let url: URL
    public let sha256: String
    public let byteCount: Int64
    public let directoryName: String
    public let files: [ModelFile]

    public init(
        url: URL,
        sha256: String,
        byteCount: Int64,
        directoryName: String,
        files: [ModelFile]
    ) {
        self.url = url
        self.sha256 = sha256
        self.byteCount = byteCount
        self.directoryName = directoryName
        self.files = files
    }
}

public enum KeywordModelManifest {
    public static let keywords = VoiceSubmitCommand.keywordDefinition

    public static let release = KeywordModelArchive(
        url: URL(
            string: "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2"
        )!,
        sha256: "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6",
        byteCount: 32_885_699,
        directoryName: "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
        files: [
            ModelFile(
                name: "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
                sha256: "408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48",
                required: true,
                byteCount: 4_599_656
            ),
            ModelFile(
                name: "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
                sha256: "63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21",
                required: true,
                byteCount: 759_829
            ),
            ModelFile(
                name: "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
                sha256: "190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1",
                required: true,
                byteCount: 86_629
            ),
            ModelFile(
                name: "tokens.txt",
                sha256: "2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0",
                required: true,
                byteCount: 1_928
            ),
        ]
    )
}

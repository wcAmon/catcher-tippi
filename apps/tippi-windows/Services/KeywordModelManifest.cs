using System.IO;

namespace Tippi.Windows.Services;

public static class KeywordModelManifest
{
    public const string ArchiveName = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2";
    public const string DirectoryName = "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20";
    public const string ArchiveUrl =
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/" + ArchiveName;
    public const long ArchiveBytes = 32_885_699;
    public const string ArchiveSha256 =
        "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6";
    public const string KeywordDefinition = "b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH\n";

    public const string EncoderName = "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx";
    public const string DecoderName = "decoder-epoch-13-avg-2-chunk-16-left-64.onnx";
    public const string JoinerName = "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx";
    public const string TokensName = "tokens.txt";
    public const string KeywordsName = "keywords.txt";

    public static readonly IReadOnlyList<ModelFile> Files =
    [
        new(EncoderName, 4_599_656, "408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48"),
        new(DecoderName, 759_829, "63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21"),
        new(JoinerName, 86_629, "190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1"),
        new(TokensName, 1_928, "2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0"),
    ];

    public static string DefaultDirectory
    {
        get
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "Tippi", "Models", DirectoryName);
        }
    }
}

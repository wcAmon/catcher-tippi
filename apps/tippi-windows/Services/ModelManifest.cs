using System.IO;

namespace Tippi.Windows.Services;

public sealed record ModelFile(string Name, long Size, string Sha256);

public static class ModelManifest
{
    public const string Repository = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4";
    public const string Revision = "8364d9e2dd9da23789b480bdbba9e423717e42ee";

    public static readonly IReadOnlyList<ModelFile> Files =
    [
        new("audio_processor_config.json", 413, "ab28d41eb87ce3922006edeb9c3fad4d5ce451f9a56a12d84f470f02a5ec157b"),
        new("decoder.onnx", 4_696, "6a9f608dcbab71ebd81ffa4c198e82a5b6bb10f1c1830a94c752c5f543454df3"),
        new("decoder.onnx.data", 59_785_216, "e5fd55cbeeb268f9d383e2ee72735b9fbbb13aea4bc7cd38cb73b8e16f1366c7"),
        new("encoder.onnx", 2_677_548, "0b05217594ec0bda442e43a90a298ac2471a3bdcea9b169de34214e61a730e17"),
        new("encoder.onnx.data", 690_089_984, "2f27295855aeb99ab1f8cd2254418d9ad7a087ea8dbe85f5596b4d887ea7d630"),
        new("genai_config.json", 1_892, "39568fbeebbe848696a1e2a01c7f33df000f72c29f2285509fd12442bda9571e"),
        new("joint.onnx", 2_136, "e2c7d2fa40a243bf82eaca36c15698c52129de9361d2875d7f223f67fcd9482d"),
        new("joint.onnx.data", 37_830_656, "2e0fb1c060f3777a1a76e78d5589dd54f01505a06dffbd2588e315508b402c12"),
        new("model_config.json", 365, "f41f943eeb1310a89dd58cf3e11e654a8ae1a788fceeb6cd1eacce3a6d081965"),
        new("silero_vad.onnx", 2_243_022, "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808"),
        new("tokenizer.json", 642_525, "24e1e8335c8396884a86f06880271376ae46a29381cfc35c82c6295d407acec7"),
        new("tokenizer_config.json", 183, "ea4b35353f468fea11f436f837d9621a29b4ba9d1c73c1ed0aa5743f5a53919e"),
        new("vocab.txt", 64_024, "ca88922ac5a92c911b79985b69634d7a4c2ef604d61b71bbe2982210dd77cd43"),
    ];

    public static long TotalBytes => Files.Sum(file => file.Size);

    public static string DefaultDirectory
    {
        get
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "Tippi", "Models", $"nemotron-3.5-asr-onnx-int4-{Revision[..8]}");
        }
    }
}

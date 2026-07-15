using System.IO;

namespace Tippi.Windows.Services;

public static class DiarizationModelManifest
{
    public const string SegmentationRepository = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0";
    public const string SegmentationRevision = "9403a6902bb58e3d5ae8c7e77c3422de279db2e0";
    public const string EmbeddingRepository = "csukuangfj/speaker-embedding-models";
    public const string EmbeddingRevision = "0743f301363dec56491a490f6d6cbc9d67f9a3bf";

    public static readonly IReadOnlyList<ModelFile> Files =
    [
        new("segmentation.int8.onnx", 1_540_506, "d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d"),
        new("nemo_en_titanet_small.onnx", 40_257_283, "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e"),
    ];

    public static long TotalBytes => Files.Sum(file => file.Size);

    public static string DefaultDirectory
    {
        get
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "Tippi", "Models", "speaker-diarization-onnx-v1");
        }
    }

    public static string UrlFor(ModelFile file)
    {
        return file.Name switch
        {
            "segmentation.int8.onnx" =>
                $"https://huggingface.co/{SegmentationRepository}/resolve/{SegmentationRevision}/model.int8.onnx",
            "nemo_en_titanet_small.onnx" =>
                $"https://huggingface.co/{EmbeddingRepository}/resolve/{EmbeddingRevision}/nemo_en_titanet_small.onnx",
            _ => throw new ArgumentOutOfRangeException(nameof(file), file.Name, "未知的說話者模型檔案。"),
        };
    }
}

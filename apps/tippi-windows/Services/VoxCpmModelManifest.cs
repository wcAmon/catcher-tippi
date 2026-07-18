using System.IO;

namespace Tippi.Windows.Services;

public static class VoxCpmModelManifest
{
    public const string Repository = "DennisHuang648/VoxCPM2-GGUF";
    public const string Revision = "169f64d8b98bbaab1761e4ca3a83e6af653456cc";
    public const string BaseModelName = "VoxCPM2-BaseLM-Q8_0.gguf";
    public const string AcousticModelName = "VoxCPM2-Acoustic-F16.gguf";

    public static readonly IReadOnlyList<ModelFile> Files =
    [
        new(BaseModelName, 1_727_309_920, "0113177abd11303503bf0b705e1613ec5f0a8508cc74a7dfd0f99312b962a962"),
        new(AcousticModelName, 1_825_096_352, "5bde898488ad635ff55d24da53543768fa33d5e5cdc538ce190e5ef831038e85"),
    ];

    public static long TotalBytes => Files.Sum(file => file.Size);

    public static string DefaultDirectory
    {
        get
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(local, "Tippi", "Models", $"voxcpm2-gguf-q8-{Revision[..8]}");
        }
    }

    public static string UrlFor(ModelFile file)
    {
        string escapedName = Uri.EscapeDataString(file.Name)
            .Replace("%2F", "/", StringComparison.OrdinalIgnoreCase);
        return $"https://huggingface.co/{Repository}/resolve/{Revision}/{escapedName}";
    }
}

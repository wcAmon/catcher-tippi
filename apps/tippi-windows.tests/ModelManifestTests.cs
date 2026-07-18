using Tippi.Windows.Services;
using System.Security.Cryptography;

namespace Tippi.Windows.Tests;

public sealed class ModelManifestTests
{
    [Fact]
    public void ManifestPinsEveryRequiredFile()
    {
        Assert.Equal(13, ModelManifest.Files.Count);
        Assert.Equal(ModelManifest.Files.Count, ModelManifest.Files.Select(file => file.Name).Distinct().Count());
        Assert.All(ModelManifest.Files, file =>
        {
            Assert.True(file.Size > 0);
            Assert.Matches("^[0-9a-f]{64}$", file.Sha256);
        });
        Assert.InRange(ModelManifest.TotalBytes, 790_000_000, 800_000_000);
        Assert.Equal(40, ModelManifest.Revision.Length);
    }

    [Fact]
    public void DiarizationManifestPinsBothCpuModels()
    {
        Assert.Equal(2, DiarizationModelManifest.Files.Count);
        Assert.All(DiarizationModelManifest.Files, file =>
        {
            Assert.True(file.Size > 0);
            Assert.Matches("^[0-9a-f]{64}$", file.Sha256);
            Assert.StartsWith("https://", DiarizationModelManifest.UrlFor(file));
        });
        Assert.InRange(DiarizationModelManifest.TotalBytes, 41_000_000, 42_000_000);
        Assert.Equal(40, DiarizationModelManifest.SegmentationRevision.Length);
        Assert.Equal(40, DiarizationModelManifest.EmbeddingRevision.Length);
    }

    [Fact]
    public void KeywordManifestPinsArchiveRuntimeAndChineseCommand()
    {
        Assert.Equal(4, KeywordModelManifest.Files.Count);
        Assert.Equal(32_885_699, KeywordModelManifest.ArchiveBytes);
        Assert.Matches("^[0-9a-f]{64}$", KeywordModelManifest.ArchiveSha256);
        Assert.Equal("b āng w ǒ s òng ch ū :1.5 #0.25 @SUBMIT_ZH\n", KeywordModelManifest.KeywordDefinition);
        Assert.All(KeywordModelManifest.Files, file =>
        {
            Assert.True(file.Size > 0);
            Assert.Matches("^[0-9a-f]{64}$", file.Sha256);
        });
    }

    [Fact]
    public void VoxCpmManifestPinsOnlyQ8BaseAndF16AcousticModels()
    {
        Assert.Equal(2, VoxCpmModelManifest.Files.Count);
        Assert.Equal(40, VoxCpmModelManifest.Revision.Length);
        Assert.Equal(3_552_406_272, VoxCpmModelManifest.TotalBytes);
        Assert.Contains(VoxCpmModelManifest.Files, file => file.Name.EndsWith("Q8_0.gguf", StringComparison.Ordinal));
        Assert.Contains(VoxCpmModelManifest.Files, file => file.Name.EndsWith("Acoustic-F16.gguf", StringComparison.Ordinal));
        Assert.All(VoxCpmModelManifest.Files, file =>
        {
            Assert.Matches("^[0-9a-f]{64}$", file.Sha256);
            Assert.StartsWith("https://huggingface.co/", VoxCpmModelManifest.UrlFor(file));
        });
    }

    [Theory]
    [InlineData("D3D12Core.dll", "8a23d826b25b4329522ff451cb52b7f2b34d7f2913cfeb878371ce8bd765fe2d")]
    [InlineData("DirectML.dll", "9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1")]
    [InlineData("onnxruntime-genai.dll", "7b34b5856b1b0b5d8590be37300fe6224169f220a6708e51018b1f90b1dfc3b7")]
    [InlineData("onnxruntime.dll", "cb0380c4072a32d1e2a1aeda9d54b94c4f645df9f81e9b37535559e57938c908")]
    [InlineData("sherpa-onnx.dll", "9cef5904ac912106dfa8aaf0c70a4e5a86370fe08781f981d37cbd49e98fd37b")]
    [InlineData("sherpa-onnx-c-api.dll", "614878147c05121aeb1514ec4fb3e48b89751591532eca9208235b9ab868306a")]
    public void BuildUsesPinnedWindowsRuntime(string fileName, string expectedSha256)
    {
        string path = Path.Combine(AppContext.BaseDirectory, fileName);
        Assert.True(File.Exists(path), $"Missing native runtime: {path}");
        using FileStream stream = File.OpenRead(path);
        string actual = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        Assert.Equal(expectedSha256, actual);
    }

    [Theory]
    [InlineData("cpu/llama-tts-server.exe", "c6d561956f4c20f4c486533aa62cc0a39935aea7dfa8964f1922104620cebd57")]
    [InlineData("cpu/ggml-cpu.dll", "ebb1c9bf0ac1e264b6663abf54e9675b7770429a8ff1181f59732b1c8011525d")]
    [InlineData("vulkan/llama-tts-server.exe", "5287998670c798509d14f69a10000859aaa9cc2c9e2b4e1e87e00405173a6993")]
    [InlineData("vulkan/ggml-vulkan.dll", "9d7c26c47fb375e473bc9b72df5b6cd52196915cfc24e153feb5b2a972b39017")]
    public void BuildUsesPinnedTtsRuntime(string relativePath, string expectedSha256)
    {
        string path = Path.Combine(
            AppContext.BaseDirectory,
            "TtsRuntime",
            relativePath.Replace('/', Path.DirectorySeparatorChar));
        Assert.True(File.Exists(path), $"Missing TTS runtime: {path}");
        using FileStream stream = File.OpenRead(path);
        string actual = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        Assert.Equal(expectedSha256, actual);
    }
}

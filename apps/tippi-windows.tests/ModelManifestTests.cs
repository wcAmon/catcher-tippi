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

    [Theory]
    [InlineData("onnxruntime-genai.dll", "97ee417fa958a7607c1ba57e14b5e3febc383b3f74d394d5e8b636c165384209")]
    [InlineData("onnxruntime.dll", "daa77083a45bf525da0dde9e87f85d8eb146f58f9c9aa7124ca84545e1c0f148")]
    [InlineData("sherpa-onnx.dll", "9cef5904ac912106dfa8aaf0c70a4e5a86370fe08781f981d37cbd49e98fd37b")]
    [InlineData("sherpa-onnx-c-api.dll", "614878147c05121aeb1514ec4fb3e48b89751591532eca9208235b9ab868306a")]
    public void BuildUsesPinnedCpuRuntime(string fileName, string expectedSha256)
    {
        string path = Path.Combine(AppContext.BaseDirectory, fileName);
        Assert.True(File.Exists(path), $"Missing native runtime: {path}");
        using FileStream stream = File.OpenRead(path);
        string actual = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        Assert.Equal(expectedSha256, actual);
    }
}

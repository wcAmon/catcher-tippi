using System.IO;

namespace Tippi.Windows.Services;

public sealed class DiarizationModelInstaller : IDisposable
{
    private readonly VerifiedModelInstaller _installer = new(
        DiarizationModelManifest.DefaultDirectory,
        DiarizationModelManifest.Files,
        DiarizationModelManifest.UrlFor);

    public string ModelDirectory => _installer.ModelDirectory;
    public string SegmentationModelPath => Path.Combine(ModelDirectory, "segmentation.int8.onnx");
    public string EmbeddingModelPath => Path.Combine(ModelDirectory, "nemo_en_titanet_small.onnx");
    public bool IsInstalled() => _installer.IsInstalled();
    public Task InstallAsync(IProgress<ModelInstallProgress>? progress, CancellationToken cancellationToken) =>
        _installer.InstallAsync(progress, cancellationToken);
    public Task<bool> VerifyAsync(CancellationToken cancellationToken) =>
        _installer.VerifyAsync(cancellationToken);
    public Task RepairAsync(IProgress<ModelInstallProgress>? progress, CancellationToken cancellationToken) =>
        _installer.RepairAsync(progress, cancellationToken);
    public void Dispose() => _installer.Dispose();
}

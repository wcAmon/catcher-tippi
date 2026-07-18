using System.IO;

namespace Tippi.Windows.Services;

public sealed class VoxCpmModelInstaller : IDisposable
{
    private readonly VerifiedModelInstaller _installer = new(
        VoxCpmModelManifest.DefaultDirectory,
        VoxCpmModelManifest.Files,
        VoxCpmModelManifest.UrlFor);

    public string ModelDirectory => _installer.ModelDirectory;
    public string BaseModelPath => Path.Combine(ModelDirectory, VoxCpmModelManifest.BaseModelName);
    public string AcousticModelPath => Path.Combine(ModelDirectory, VoxCpmModelManifest.AcousticModelName);
    public bool IsInstalled() => _installer.IsInstalled();
    public Task InstallAsync(IProgress<ModelInstallProgress>? progress, CancellationToken cancellationToken) =>
        _installer.InstallAsync(progress, cancellationToken);
    public Task<bool> VerifyAsync(CancellationToken cancellationToken) =>
        _installer.VerifyAsync(cancellationToken);
    public Task RepairAsync(IProgress<ModelInstallProgress>? progress, CancellationToken cancellationToken) =>
        _installer.RepairAsync(progress, cancellationToken);
    public void Dispose() => _installer.Dispose();
}

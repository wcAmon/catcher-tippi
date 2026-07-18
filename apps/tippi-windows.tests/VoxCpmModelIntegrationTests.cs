using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class VoxCpmModelIntegrationTests
{
    [Fact]
    public async Task DownloaderInstallsAndVerifiesPinnedQ8ModelPair()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_TTS_DOWNLOAD_TEST") != "1")
        {
            return;
        }

        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(30));
        using var installer = new VoxCpmModelInstaller();
        await installer.InstallAsync(progress: null, timeout.Token);

        Assert.True(installer.IsInstalled());
        Assert.True(await installer.VerifyAsync(timeout.Token));
    }
}

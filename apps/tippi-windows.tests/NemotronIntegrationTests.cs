using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class NemotronIntegrationTests
{
    [Fact]
    public async Task OriginalModelTranscribesReferenceAudioOnCpu()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_MODEL_TEST") != "1")
        {
            return;
        }

        using var installer = new ModelInstaller();
        await installer.InstallAsync(progress: null, CancellationToken.None);
        using var engine = new NemotronEngine(installer.ModelDirectory);
        engine.BeginSession("en-US", useVad: false, traditionalChinese: false);

        TranscriptionUpdate? latest = Transcribe(engine, "hello-streaming.wav");

        Assert.NotNull(latest);
        Assert.Contains("streaming speech recognition test", latest.DisplayText, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task OriginalModelTranscribesMandarinAsTaiwanTraditionalOnCpu()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_MODEL_TEST") != "1")
        {
            return;
        }

        using var installer = new ModelInstaller();
        await installer.InstallAsync(progress: null, CancellationToken.None);
        using var engine = new NemotronEngine(installer.ModelDirectory);
        engine.BeginSession("zh-CN", useVad: false, traditionalChinese: true);

        TranscriptionUpdate? latest = Transcribe(engine, "bang-wo-song-chu-zh-cn.wav");

        Assert.NotNull(latest);
        Assert.Contains("幫我送出", latest.DisplayText, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AutoBackendFallsBackToCpuWhenCurrentInt4CannotDecodeOnDirectMl()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_DML_MODEL_TEST") != "1")
        {
            return;
        }

        using var installer = new ModelInstaller();
        await installer.InstallAsync(progress: null, CancellationToken.None);
        string profilePath = Path.Combine(Path.GetTempPath(), $"tippi-backend-{Guid.NewGuid():N}.json");
        try
        {
            var loader = new InferenceBackendLoader(profilePath);
            InferenceLoadResult loaded = await loader.LoadAsync(
                installer.ModelDirectory,
                InferenceBackendPreference.Auto,
                forceProbe: true,
                progress: null,
                CancellationToken.None);
            using NemotronEngine engine = loaded.Engine;

            Assert.Equal(InferenceBackend.Cpu, loaded.SelectedBackend);
            engine.BeginSession("en-US", useVad: false, traditionalChinese: false);

            TranscriptionUpdate? latest = Transcribe(engine, "hello-streaming.wav");

            Assert.NotNull(latest);
            Assert.Contains("streaming speech recognition test", latest.DisplayText, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(profilePath);
        }
    }

    private static TranscriptionUpdate? Transcribe(NemotronEngine engine, string fixtureName)
    {
        string fixture = Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory,
            "..", "..", "..", "..", "..", "tests", "fixtures", fixtureName));
        float[] audio = AudioFileLoader.LoadMono16Khz(fixture);
        TranscriptionUpdate? latest = null;
        foreach (float[] chunk in audio.Chunk(8_960))
        {
            latest = engine.Process(chunk) ?? latest;
        }
        latest = engine.Flush() ?? latest;

        return latest;
    }
}

using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class FourModelCoexistenceIntegrationTests
{
    [Fact]
    public async Task AsrKeywordSpotterDiarizerAndVoxCpmCanRunTogether()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_TTS_TEST") != "1")
        {
            return;
        }

        using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(15));
        using var asrInstaller = new ModelInstaller();
        using var keywordInstaller = new KeywordModelInstaller();
        using var diarizationInstaller = new DiarizationModelInstaller();
        using var ttsInstaller = new VoxCpmModelInstaller();
        Assert.True(asrInstaller.IsInstalled(), $"ASR model is not installed at {asrInstaller.ModelDirectory}");
        Assert.True(keywordInstaller.IsInstalled(), $"KWS model is not installed at {keywordInstaller.ModelDirectory}");
        Assert.True(diarizationInstaller.IsInstalled(), $"Diarization models are not installed at {diarizationInstaller.ModelDirectory}");
        Assert.True(ttsInstaller.IsInstalled(), $"VoxCPM2 model is not installed at {ttsInstaller.ModelDirectory}");

        using var asr = new NemotronEngine(asrInstaller.ModelDirectory, InferenceBackend.Cpu);
        using var keyword = new VoiceCommandSpotter(keywordInstaller);
        using var diarizer = new SpeakerDiarizer(
            diarizationInstaller.SegmentationModelPath,
            diarizationInstaller.EmbeddingModelPath);
        await using var tts = new VoxCpmTtsService();
        asr.BeginSession("zh-CN", useVad: true, traditionalChinese: true);
        keyword.BeginSession();

        TtsBackendPreference preference = Environment.GetEnvironmentVariable("TIPPI_TTS_BACKEND") switch
        {
            "vulkan" => TtsBackendPreference.Vulkan,
            "cpu" => TtsBackendPreference.Cpu,
            _ => TtsBackendPreference.Auto,
        };
        TtsBackend startedBackend = await tts.EnsureStartedAsync(
            ttsInstaller,
            preference,
            progress: null,
            timeout.Token);
        float[] completeConversation = AudioFileLoader.LoadMono16Khz(
            Path.Combine(FindFixturesDirectory(), "conversation.wav"));
        float[] diarizationAudio = completeConversation[..Math.Min(completeConversation.Length, 17 * 16_000)];
        Task<IReadOnlyList<SpeakerTimeSegment>> diarization = Task.Run(
            () => diarizer.Process(diarizationAudio),
            timeout.Token);
        Task<TtsSynthesisResult> synthesis = tts.SynthesizeWithFallbackAsync(
            ttsInstaller,
            preference,
            "你好，這是 Tippi 的四模型共存測試。",
            inferenceTimesteps: 4,
            progress: null,
            timeout.Token);

        float[] commandAudio = AudioFileLoader.LoadMono16Khz(
            Path.Combine(FindFixturesDirectory(), "bang-wo-song-chu-zh-tw.wav"));
        bool commandDetected = false;
        TranscriptionUpdate? latest = null;
        foreach (float[] chunk in commandAudio.Chunk(1_600))
        {
            commandDetected |= keyword.Process(chunk);
            latest = asr.Process(chunk) ?? latest;
        }
        commandDetected |= keyword.Finish();
        latest = asr.Flush() ?? latest;

        TtsSynthesisResult result = await synthesis;
        IReadOnlyList<SpeakerTimeSegment> segments = await diarization;
        Console.WriteLine(
            $"VoxCPM2 started on {startedBackend}, completed on {result.Backend}; " +
            $"diarization segments: {segments.Count}");

        Assert.True(commandDetected);
        Assert.False(string.IsNullOrWhiteSpace(latest?.DisplayText));
        Assert.NotEmpty(segments);
        Assert.True(segments.Select(segment => segment.Speaker).Distinct().Count() >= 2);
        Assert.True(result.WaveData.Length > 44);
        Assert.Equal("RIFF", System.Text.Encoding.ASCII.GetString(result.WaveData, 0, 4));
    }

    private static string FindFixturesDirectory()
    {
        DirectoryInfo? directory = new(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Combine(directory.FullName, "tests", "fixtures");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
            directory = directory.Parent;
        }
        throw new DirectoryNotFoundException("Could not find tests/fixtures.");
    }
}

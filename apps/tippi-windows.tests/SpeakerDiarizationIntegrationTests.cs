using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class SpeakerDiarizationIntegrationTests
{
    [Fact]
    public async Task DownloaderInstallsAndVerifiesPinnedDiarizationModels()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_DIAR_DOWNLOAD_TEST") != "1")
        {
            return;
        }

        using var installer = new DiarizationModelInstaller();
        await installer.InstallAsync(progress: null, CancellationToken.None);

        Assert.True(installer.IsInstalled());
        Assert.True(await installer.VerifyAsync(CancellationToken.None));
    }

    [Fact]
    public void CpuDiarizerFindsBothSpeakersInConversationFixture()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_DIAR_TEST") != "1")
        {
            return;
        }

        string modelDirectory = Environment.GetEnvironmentVariable("TIPPI_DIAR_MODEL_DIR")
            ?? throw new InvalidOperationException("Set TIPPI_DIAR_MODEL_DIR to the two ONNX diarization models.");
        using var diarizer = new SpeakerDiarizer(
            Path.Combine(modelDirectory, "segmentation.int8.onnx"),
            Path.Combine(modelDirectory, "nemo_en_titanet_small.onnx"));
        float[] audio = AudioFileLoader.LoadMono16Khz(Fixture("conversation.wav"));

        IReadOnlyList<SpeakerTimeSegment> segments = diarizer.Process(audio);

        Assert.NotEmpty(segments);
        Assert.True(
            segments.Select(segment => segment.Speaker).Distinct().Count() >= 2,
            $"Expected at least two speakers, got: {string.Join(", ", segments)}");
        Assert.Contains(
            segments.Zip(segments.Skip(1)),
            pair => pair.First.Speaker != pair.Second.Speaker);
        int[] speakerIds = segments.Select(segment => segment.Speaker).Distinct().Order().ToArray();
        Assert.Equal(Enumerable.Range(0, speakerIds.Length), speakerIds);
    }

    [Fact]
    public async Task EndToEndCpuPipelineProducesSpeakerAttributedTranscript()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_END_TO_END_DIAR_TEST") != "1")
        {
            return;
        }

        using var asrInstaller = new ModelInstaller();
        await asrInstaller.InstallAsync(progress: null, CancellationToken.None);
        using var diarInstaller = new DiarizationModelInstaller();
        await diarInstaller.InstallAsync(progress: null, CancellationToken.None);
        float[] completeAudio = AudioFileLoader.LoadMono16Khz(Fixture("conversation.wav"));
        // The first 17 seconds contain both fixture speakers and keep this
        // deliberately expensive CPU-only end-to-end check manageable.
        float[] audio = completeAudio[..Math.Min(completeAudio.Length, 17 * 16_000)];

        var collector = new TimedTranscriptCollector();
        using var engine = new NemotronEngine(asrInstaller.ModelDirectory);
        engine.BeginSession("zh-CN", useVad: true, traditionalChinese: true);
        TranscriptionUpdate? latest = null;
        int processed = 0;
        foreach (float[] chunk in audio.Chunk(8_960))
        {
            processed += chunk.Length;
            latest = engine.Process(chunk) ?? latest;
            if (latest is not null)
            {
                collector.Update(latest.RawText, processed / 16_000d);
            }
        }
        latest = engine.Flush() ?? latest;
        Assert.NotNull(latest);
        collector.Update(latest.RawText, audio.Length / 16_000d);

        using var diarizer = new SpeakerDiarizer(
            diarInstaller.SegmentationModelPath,
            diarInstaller.EmbeddingModelPath);
        IReadOnlyList<SpeakerTimeSegment> segments = diarizer.Process(audio);
        Assert.True(segments.Select(segment => segment.Speaker).Distinct().Count() >= 2);
        string transcript = SpeakerTranscriptFormatter.Format(
            collector.Chunks,
            segments,
            latest.RawText,
            traditionalChinese: true);

        Assert.Contains("說話者 1：", transcript);
        Assert.DoesNotContain("fallback", transcript);
        Assert.DoesNotContain("說話者 4：", transcript);
    }

    private static string Fixture(string name) => Path.GetFullPath(Path.Combine(
        AppContext.BaseDirectory,
        "..", "..", "..", "..", "..", "tests", "fixtures", name));
}

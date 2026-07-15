using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class SpeakerTranscriptFormatterTests
{
    [Fact]
    public void CollectorStoresOnlyNewStreamingText()
    {
        var collector = new TimedTranscriptCollector();

        collector.Update("Hello", 0.56);
        collector.Update("Hello world", 1.12);
        collector.Update("Hello world", 1.68);

        Assert.Collection(
            collector.Chunks,
            first =>
            {
                Assert.Equal("Hello", first.Text);
                Assert.Equal(0, first.StartSeconds, 2);
                Assert.Equal(0.56, first.EndSeconds, 2);
            },
            second =>
            {
                Assert.Equal(" world", second.Text);
                Assert.Equal(0.56, second.StartSeconds, 2);
                Assert.Equal(1.12, second.EndSeconds, 2);
            });
    }

    [Fact]
    public void FormatterGroupsAdjacentChunksBySpeaker()
    {
        TimedTextChunk[] chunks =
        [
            new("Good ", 0, 1),
            new("morning.", 1, 2),
            new("Hello!", 3, 4),
        ];
        SpeakerTimeSegment[] segments =
        [
            new(0, 2, 0),
            new(3, 4, 1),
        ];

        string result = SpeakerTranscriptFormatter.Format(chunks, segments, "fallback", false);

        Assert.Equal(
            "[00:00] 說話者 1：Good morning." + Environment.NewLine + Environment.NewLine +
            "[00:03] 說話者 2：Hello!",
            result);
    }

    [Fact]
    public void FormatterUsesNearestSpeakerAcrossSmallGaps()
    {
        TimedTextChunk[] chunks = [new("gap text", 2.1, 2.3)];
        SpeakerTimeSegment[] segments = [new(0, 2, 0), new(3, 4, 1)];

        string result = SpeakerTranscriptFormatter.Format(chunks, segments, "fallback", false);

        Assert.Contains("說話者 1：gap text", result);
    }

    [Fact]
    public void FormatterFallsBackToFlatTranscriptWithoutDiarization()
    {
        string result = SpeakerTranscriptFormatter.Format([], [], "软件和鼠标", true);

        Assert.Equal("軟體和滑鼠", result);
    }

    [Fact]
    public void FormatterSplitsDelayedAsrUpdateAcrossOverlappingSpeakers()
    {
        TimedTextChunk[] chunks = [new("First speaker. Second speaker.", 0, 4)];
        SpeakerTimeSegment[] segments = [new(0, 2, 0), new(2, 4, 1)];

        string result = SpeakerTranscriptFormatter.Format(chunks, segments, "fallback", false);

        Assert.Contains("說話者 1：", result);
        Assert.Contains("說話者 2：", result);
        Assert.Contains("First speaker.", result);
        Assert.Contains("Second speaker.", result);
    }
}

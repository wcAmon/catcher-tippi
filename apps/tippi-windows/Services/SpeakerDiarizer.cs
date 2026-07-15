using SherpaOnnx;
using System.IO;

namespace Tippi.Windows.Services;

public sealed record SpeakerTimeSegment(double StartSeconds, double EndSeconds, int Speaker);

public sealed class SpeakerDiarizer : IDisposable
{
    private readonly OfflineSpeakerDiarization _diarizer;

    public SpeakerDiarizer(string segmentationModelPath, string embeddingModelPath)
    {
        if (!File.Exists(segmentationModelPath))
        {
            throw new FileNotFoundException("找不到說話者分段模型。", segmentationModelPath);
        }
        if (!File.Exists(embeddingModelPath))
        {
            throw new FileNotFoundException("找不到說話者特徵模型。", embeddingModelPath);
        }

        int threads = Math.Clamp(Environment.ProcessorCount / 2, 1, 4);
        var config = new OfflineSpeakerDiarizationConfig();
        config.Segmentation.Pyannote.Model = segmentationModelPath;
        config.Segmentation.NumThreads = threads;
        config.Segmentation.Provider = "cpu";
        config.Embedding.Model = embeddingModelPath;
        config.Embedding.NumThreads = threads;
        config.Embedding.Provider = "cpu";
        config.Clustering.NumClusters = -1;
        config.Clustering.Threshold = 0.5f;
        config.MinDurationOn = 0.3f;
        config.MinDurationOff = 0.5f;

        _diarizer = new OfflineSpeakerDiarization(config);
        int sampleRate = _diarizer.SampleRate;
        if (sampleRate != 16_000)
        {
            _diarizer.Dispose();
            throw new InvalidOperationException($"說話者模型取樣率不是 16 kHz：{sampleRate}");
        }
    }

    public IReadOnlyList<SpeakerTimeSegment> Process(float[] mono16KhzSamples)
    {
        OfflineSpeakerDiarizationSegment[] segments = _diarizer.Process(mono16KhzSamples);
        SpeakerTimeSegment[] ordered = segments
            .Where(segment => segment.End > segment.Start)
            .Select(segment => new SpeakerTimeSegment(segment.Start, segment.End, segment.Speaker))
            .OrderBy(segment => segment.StartSeconds)
            .ToArray();

        // Clustering IDs are internal labels and may be sparse (for example,
        // 0 and 3). Present stable, first-appearance IDs so the UI always says
        // Speaker 1, Speaker 2, ... without gaps.
        var normalizedIds = new Dictionary<int, int>();
        return ordered.Select(segment =>
        {
            if (!normalizedIds.TryGetValue(segment.Speaker, out int normalized))
            {
                normalized = normalizedIds.Count;
                normalizedIds.Add(segment.Speaker, normalized);
            }
            return segment with { Speaker = normalized };
        }).ToArray();
    }

    public void Dispose() => _diarizer.Dispose();
}

using OpenccNetLib;

namespace Tippi.Windows.Services;

public sealed record TimedTextChunk(string Text, double StartSeconds, double EndSeconds);

public sealed class TimedTranscriptCollector
{
    private readonly List<TimedTextChunk> _chunks = [];
    private string _previousRawText = string.Empty;
    private double _previousTextAudioEndSeconds;

    public IReadOnlyList<TimedTextChunk> Chunks => _chunks;

    public void Update(string rawText, double audioEndSeconds)
    {
        if (string.IsNullOrEmpty(rawText) || rawText == _previousRawText)
        {
            return;
        }

        int commonLength = CommonPrefixLength(_previousRawText, rawText);
        if (commonLength < _previousRawText.Length)
        {
            // Nemotron normally appends tokens. If a future runtime revises an
            // earlier token, retain only the newly returned suffix instead of
            // duplicating the complete transcript.
            commonLength = Math.Min(commonLength, rawText.Length);
        }

        string appended = rawText[commonLength..];
        if (appended.Length > 0)
        {
            // VAD can delay a whole utterance until a later audio block. Keep
            // the complete interval since the previous text emission so that
            // a long update can be distributed across every overlapping
            // speaker segment instead of being assigned to one instant.
            _chunks.Add(new(
                appended,
                _previousTextAudioEndSeconds,
                Math.Max(_previousTextAudioEndSeconds, audioEndSeconds)));
            _previousTextAudioEndSeconds = Math.Max(_previousTextAudioEndSeconds, audioEndSeconds);
        }
        _previousRawText = rawText;
    }

    private static int CommonPrefixLength(string left, string right)
    {
        int length = Math.Min(left.Length, right.Length);
        int index = 0;
        while (index < length && left[index] == right[index])
        {
            index++;
        }
        return index;
    }
}

public static class SpeakerTranscriptFormatter
{
    public static string Format(
        IReadOnlyList<TimedTextChunk> chunks,
        IReadOnlyList<SpeakerTimeSegment> speakerSegments,
        string fallbackRawText,
        bool traditionalChinese)
    {
        if (chunks.Count == 0 || speakerSegments.Count == 0)
        {
            return Convert(fallbackRawText, traditionalChinese);
        }

        var attributed = new List<AttributedText>();
        foreach (TimedTextChunk chunk in chunks.Where(chunk => chunk.Text.Length > 0))
        {
            foreach (AttributedText piece in AttributeChunk(chunk, speakerSegments))
            {
                if (string.IsNullOrEmpty(piece.Text))
                {
                    continue;
                }
                if (attributed.Count > 0 && attributed[^1].Speaker == piece.Speaker)
                {
                    attributed[^1].Text += piece.Text;
                    attributed[^1].EndSeconds = Math.Max(attributed[^1].EndSeconds, piece.EndSeconds);
                }
                else
                {
                    attributed.Add(piece);
                }
            }
        }

        if (attributed.Count == 0)
        {
            return Convert(fallbackRawText, traditionalChinese);
        }

        return string.Join(
            Environment.NewLine + Environment.NewLine,
            attributed.Select(item =>
                $"[{FormatTimestamp(item.StartSeconds)}] 說話者 {item.Speaker + 1}：{Convert(item.Text.Trim(), traditionalChinese)}"));
    }

    private static IReadOnlyList<AttributedText> AttributeChunk(
        TimedTextChunk chunk,
        IReadOnlyList<SpeakerTimeSegment> segments)
    {
        SpeakerTimeSegment[] overlaps = segments
            .Select(segment => new SpeakerTimeSegment(
                Math.Max(segment.StartSeconds, chunk.StartSeconds),
                Math.Min(segment.EndSeconds, chunk.EndSeconds),
                segment.Speaker))
            .Where(segment => segment.EndSeconds > segment.StartSeconds)
            .OrderBy(segment => segment.StartSeconds)
            .ToArray();

        if (overlaps.Length == 0)
        {
            double midpoint = (chunk.StartSeconds + chunk.EndSeconds) / 2;
            SpeakerTimeSegment closest = FindClosestSegment(midpoint, segments);
            return [new AttributedText(
                closest.Speaker,
                closest.StartSeconds,
                closest.EndSeconds,
                chunk.Text)];
        }

        var merged = new List<SpeakerTimeSegment>();
        foreach (SpeakerTimeSegment overlap in overlaps)
        {
            if (merged.Count > 0 && merged[^1].Speaker == overlap.Speaker)
            {
                SpeakerTimeSegment previous = merged[^1];
                merged[^1] = previous with { EndSeconds = Math.Max(previous.EndSeconds, overlap.EndSeconds) };
            }
            else
            {
                merged.Add(overlap);
            }
        }

        if (merged.Count == 1)
        {
            SpeakerTimeSegment only = merged[0];
            return [new AttributedText(only.Speaker, only.StartSeconds, only.EndSeconds, chunk.Text)];
        }

        double totalDuration = merged.Sum(segment => segment.EndSeconds - segment.StartSeconds);
        if (totalDuration <= 0)
        {
            SpeakerTimeSegment first = merged[0];
            return [new AttributedText(first.Speaker, first.StartSeconds, first.EndSeconds, chunk.Text)];
        }

        var result = new List<AttributedText>();
        double elapsedDuration = 0;
        int textStart = 0;
        for (int index = 0; index < merged.Count; index++)
        {
            SpeakerTimeSegment segment = merged[index];
            elapsedDuration += segment.EndSeconds - segment.StartSeconds;
            int textEnd = index == merged.Count - 1
                ? chunk.Text.Length
                : (int)Math.Round(chunk.Text.Length * elapsedDuration / totalDuration);
            textEnd = Math.Clamp(textEnd, textStart, chunk.Text.Length);
            string part = chunk.Text[textStart..textEnd];
            if (part.Length > 0)
            {
                result.Add(new AttributedText(
                    segment.Speaker,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    part));
            }
            textStart = textEnd;
        }
        return result;
    }

    private static SpeakerTimeSegment FindClosestSegment(
        double atSeconds,
        IReadOnlyList<SpeakerTimeSegment> segments)
    {
        SpeakerTimeSegment? containing = segments
            .Where(segment => segment.StartSeconds <= atSeconds && segment.EndSeconds >= atSeconds)
            .OrderBy(segment => Math.Abs(((segment.StartSeconds + segment.EndSeconds) / 2) - atSeconds))
            .FirstOrDefault();
        if (containing is not null)
        {
            return containing;
        }

        return segments
            .OrderBy(segment => atSeconds < segment.StartSeconds
                ? segment.StartSeconds - atSeconds
                : atSeconds - segment.EndSeconds)
            .First();
    }

    private static string FormatTimestamp(double seconds)
    {
        var timestamp = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return timestamp.TotalHours >= 1
            ? $"{(int)timestamp.TotalHours:00}:{timestamp.Minutes:00}:{timestamp.Seconds:00}"
            : $"{timestamp.Minutes:00}:{timestamp.Seconds:00}";
    }

    private static string Convert(string text, bool traditionalChinese)
    {
        if (!traditionalChinese || string.IsNullOrEmpty(text))
        {
            return text;
        }
        var opencc = new Opencc("s2twp");
        return opencc.Convert(text);
    }

    private sealed class AttributedText(int speaker, double startSeconds, double endSeconds, string text)
    {
        public int Speaker { get; } = speaker;
        public double StartSeconds { get; } = startSeconds;
        public double EndSeconds { get; set; } = endSeconds;
        public string Text { get; set; } = text;
    }
}

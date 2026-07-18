using Tippi.Windows.Services;

namespace Tippi.Windows.Tests;

public sealed class VoiceCommandIntegrationTests
{
    [Fact]
    public void RealKeywordModelMatchesPositiveAndNegativeMatrix()
    {
        if (Environment.GetEnvironmentVariable("TIPPI_RUN_KWS_TEST") != "1")
        {
            return;
        }

        using var installer = new KeywordModelInstaller();
        Assert.True(installer.IsInstalled(), $"KWS model is not installed at {installer.ModelDirectory}");
        using var spotter = new VoiceCommandSpotter(installer);
        string fixtures = FindFixturesDirectory();
        string[] positives = ["bang-wo-song-chu-zh-cn.wav", "bang-wo-song-chu-zh-tw.wav"];
        string[] negatives = ["bang-wo-zh-tw.wav", "song-chu-zh-tw.wav", "hello-streaming.wav", "conversation.wav"];

        foreach (string fixture in positives.Concat(negatives))
        {
            float[] audio = AudioFileLoader.LoadMono16Khz(Path.Combine(fixtures, fixture));
            spotter.BeginSession();
            bool detected = false;
            foreach (float[] chunk in audio.Chunk(1_600))
            {
                detected |= spotter.Process(chunk);
            }
            detected |= spotter.Finish();
            Assert.Equal(positives.Contains(fixture), detected);
        }
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

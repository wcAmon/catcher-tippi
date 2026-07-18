// 對應 mac crates/catcher-asr-host/tests/stdio.rs:對編譯出的二進位做黑箱協定測試。
//
// exe 路徑解析:從本測試組件(NemotronAsrHost.Tests.dll)所在目錄往上推回 apps/,
// 再組出 apps/nemotron-asr-host/bin/Release/net8.0/nemotron-asr-host.exe。
// ProjectReference(見 csproj)保證編譯順序讓 exe 先於測試組件產生。
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using Xunit;

namespace NemotronAsrHost.Tests;

public class StdioBlackBoxTests
{
    private static string LocateHostExecutable()
    {
        var testAssemblyDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
        // testAssemblyDir 形如 .../apps/nemotron-asr-host.tests/bin/Release/net8.0
        // 往上 4 層回到 apps/,再進 nemotron-asr-host/bin/Release/net8.0。
        var appsDir = Path.GetFullPath(Path.Combine(testAssemblyDir, "..", "..", "..", ".."));
        var exeName = OperatingSystem.IsWindows() ? "nemotron-asr-host.exe" : "nemotron-asr-host";
        var exePath = Path.Combine(appsDir, "nemotron-asr-host", "bin", "Release", "net8.0", exeName);
        if (!File.Exists(exePath))
        {
            throw new FileNotFoundException(
                $"host executable not found at {exePath}; expected ProjectReference build order to produce it.",
                exePath);
        }
        return exePath;
    }

    private static Process SpawnFakeHost()
    {
        var psi = new ProcessStartInfo
        {
            FileName = LocateHostExecutable(),
            Arguments = "--fake-engine",
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };
        return Process.Start(psi) ?? throw new InvalidOperationException("failed to spawn nemotron-asr-host");
    }

    private static JsonElement ReadLine(Process process)
    {
        var line = process.StandardOutput.ReadLine()
            ?? throw new EndOfStreamException("unexpected EOF from host stdout");
        return JsonDocument.Parse(line.Trim()).RootElement;
    }

    // 對應 full_protocol_roundtrip_with_fake_engine
    [Fact]
    public void FullProtocolRoundtripWithFakeEngine()
    {
        using var process = SpawnFakeHost();

        var ready = ReadLine(process);
        Assert.Equal("ready", ready.GetProperty("event").GetString());
        Assert.Equal("fake", ready.GetProperty("backend").GetString());

        var stdin = process.StandardInput;
        var chunk = Convert.ToBase64String(new byte[1600 * 2]);
        stdin.WriteLine("{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":16000}");
        stdin.WriteLine($"{{\"cmd\":\"audio\",\"pcm16_b64\":\"{chunk}\"}}");
        stdin.WriteLine("{\"cmd\":\"stop\"}");

        var partial = ReadLine(process);
        Assert.Equal("partial", partial.GetProperty("event").GetString());
        Assert.Equal("字0", partial.GetProperty("text").GetString());

        var finalEvent = ReadLine(process);
        Assert.Equal("final", finalEvent.GetProperty("event").GetString());
        Assert.Equal("字0", finalEvent.GetProperty("text").GetString());

        stdin.Close(); // EOF → 行程正常結束
        Assert.True(process.WaitForExit(5000));
        Assert.Equal(0, process.ExitCode);
    }

    // 對應 malformed_line_yields_error_and_keeps_running
    [Fact]
    public void MalformedLineYieldsErrorAndKeepsRunning()
    {
        using var process = SpawnFakeHost();
        Assert.Equal("ready", ReadLine(process).GetProperty("event").GetString());

        var stdin = process.StandardInput;
        stdin.WriteLine("garbage");
        Assert.Equal("error", ReadLine(process).GetProperty("event").GetString());

        // 行程還活著:正常會話仍可走完
        stdin.WriteLine("{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":16000}");
        stdin.WriteLine("{\"cmd\":\"stop\"}");
        Assert.Equal("final", ReadLine(process).GetProperty("event").GetString());

        stdin.Close();
        Assert.True(process.WaitForExit(5000));
        Assert.Equal(0, process.ExitCode);
    }

    // 對應 two_sessions_in_one_process
    [Fact]
    public void TwoSessionsInOneProcess()
    {
        using var process = SpawnFakeHost();
        Assert.Equal("ready", ReadLine(process).GetProperty("event").GetString());

        var stdin = process.StandardInput;
        var chunk = Convert.ToBase64String(new byte[1600 * 2]);

        for (int i = 0; i < 2; i++)
        {
            stdin.WriteLine("{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":16000}");
            stdin.WriteLine($"{{\"cmd\":\"audio\",\"pcm16_b64\":\"{chunk}\"}}");
            stdin.WriteLine("{\"cmd\":\"stop\"}");

            var partial = ReadLine(process);
            Assert.Equal("partial", partial.GetProperty("event").GetString());
            Assert.Equal("字0", partial.GetProperty("text").GetString());

            var finalEvent = ReadLine(process);
            Assert.Equal("final", finalEvent.GetProperty("event").GetString());
            Assert.Equal("字0", finalEvent.GetProperty("text").GetString());
        }

        stdin.Close(); // EOF → 行程正常結束
        Assert.True(process.WaitForExit(5000));
        Assert.Equal(0, process.ExitCode);
    }
}

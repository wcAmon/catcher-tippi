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
        // 注意:路徑硬編 Release —— 本測試 harness 依 scripts/bootstrap-windows-host.md
        // 固定以 `dotnet test ... -c Release` 執行;Debug 組態不受支援。
        var appsDir = Path.GetFullPath(Path.Combine(testAssemblyDir, "..", "..", "..", ".."));
        var exeName = OperatingSystem.IsWindows() ? "nemotron-asr-host.exe" : "nemotron-asr-host";
        var exePath = Path.Combine(appsDir, "nemotron-asr-host", "bin", "Release", "net8.0", exeName);
        if (!File.Exists(exePath))
        {
            throw new FileNotFoundException(
                $"host executable not found at {exePath}; this harness requires the Release configuration — build with -c Release first (dotnet test ... -c Release).",
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
            // DEVIATION(非編譯器強制,見 task-3-report.md):不設定時 .NET 用系統 OEM
            // codepage(遠端機為 cp950 Big5)解碼子行程 stdout,把 host 寫出的原始 UTF-8
            // 中文位元組誤判成雙位元組 Big5 字元(觀察到 "字0" 被誤讀成 "摮?")。
            // 顯式釘住 UTF-8,與子行程 Program.cs 的 Console.OutputEncoding 對齊。
            // 注意:StandardInputEncoding 刻意不設 —— System.Text.Encoding.UTF8 帶 BOM
            // preamble,設了會在 stdin 第一次寫入時多寫 3 個 BOM 位元組,把 start 那行
            // 撞壞(已實測到:第一個 partial 變成 error)。stdin 內容全是 ASCII/base64,
            // 用預設編碼即可正確傳輸,不需要顯式 UTF-8。
            StandardOutputEncoding = System.Text.Encoding.UTF8,
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

        // cmd 非字串:合法 JSON 但型別錯 → error 且行程不得因未捕捉例外死亡
        stdin.WriteLine("{\"cmd\":123}");
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

// 真模型端到端測試。需要 NEMOTRON_ASR_MODEL_DIR 指向已下載並雜湊驗證的模型目錄
// (scripts/fetch-nemotron-onnx-model.ps1 產出,預設 C:\Users\i5491\catcher-tippi-models\nemotron-onnx-int4)。
//
// GATING 選擇說明:xUnit 2.x 的 [Fact(Skip=...)] 只能編譯期靜態決定,沒有原生執行期
// 動態 skip。採用 Xunit.SkippableFact 套件的 [SkippableFact] + Skip.If:缺
// NEMOTRON_ASR_MODEL_DIR 時測試回報「略過」(SKIPPED),而非早期版本「提早 return 假綠燈」
// ——單純 `dotnet test`(未濾掉 RealModel、未設環境變數)的總結行會顯示「略過: 1」,
// 不會讓人誤以為真模型斷言跑過了。fast-suite 主要工作流仍是
// `dotnet test --filter FullyQualifiedName!~RealModel`(見 task-4-report.md)。
//
// 對應 crates/catcher-asr-host/tests/real_model.rs 的 transcribes_fixture_wav_end_to_end:
// normalize_for_asr / normalized_levenshtein 兩個 helper 逐語義移植(char 級,先去標點/空白,
// 門檻同為 0.25)。
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace NemotronAsrHost.Tests;

public class RealModelTests(ITestOutputHelper output)
{
    private const string ExpectedText = "Hello, this is a streaming speech recognition test";
    private const double MaxNormalizedDistance = 0.25;

    [SkippableFact]
    public void RealModel_TranscribesHelloStreamingWav()
    {
        string? modelDir = Environment.GetEnvironmentVariable("NEMOTRON_ASR_MODEL_DIR");
        Skip.If(
            string.IsNullOrEmpty(modelDir),
            "NEMOTRON_ASR_MODEL_DIR 未設定。設定後執行:" +
            "set NEMOTRON_ASR_MODEL_DIR=<dir>&dotnet test --filter RealModel -c Release");

        string exePath = LocateHostExecutable();
        string wavPath = LocateFixture("hello-streaming.wav");
        byte[] pcm = ReadWavPcm16(wavPath);

        var psi = new ProcessStartInfo
        {
            FileName = exePath,
            // modelDir 已由上方 Skip.If 保證非空;編譯器看不穿 Skip.If 的 throw 語義,需 null-forgiving。
            ArgumentList = { "--model", modelDir!, "--language", "en-US" },
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            // 對齊 StdioBlackBoxTests.cs 的 DEVIATION 註解:遠端機系統 OEM codepage
            // 非 UTF-8,不顯式指定會把子行程真正的 UTF-8 輸出誤解成 Big5。
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        using var process = Process.Start(psi)
            ?? throw new InvalidOperationException("failed to spawn nemotron-asr-host");

        var stderrLines = new List<string>();
        process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data is not null)
            {
                stderrLines.Add(e.Data);
            }
        };
        process.BeginErrorReadLine();

        var reader = process.StandardOutput;

        string readyLine = reader.ReadLine() ?? throw new EndOfStreamException("host produced no stdout");
        JsonElement ready = JsonDocument.Parse(readyLine.Trim()).RootElement;
        string readyKind = ready.GetProperty("event").GetString()!;
        if (readyKind != "ready")
        {
            process.WaitForExit(5000);
            process.CancelErrorRead();
            foreach (string stderrLine in stderrLines)
            {
                output.WriteLine($"[host stderr] {stderrLine}");
            }
            output.WriteLine($"[host stdout first line] {readyLine}");
        }
        Assert.Equal("ready", readyKind);
        string backend = ready.GetProperty("backend").GetString()!;
        output.WriteLine($"ready backend={backend}");

        var stdin = process.StandardInput;
        stdin.WriteLine("{\"cmd\":\"start\",\"lang\":\"en-US\",\"sample_rate\":16000}");
        const int chunkSamples = 1600; // 100ms @ 16kHz,協定建議值
        const int chunkBytes = chunkSamples * 2; // PCM16 = 2 bytes/sample
        for (int offset = 0; offset < pcm.Length; offset += chunkBytes)
        {
            int length = Math.Min(chunkBytes, pcm.Length - offset);
            string b64 = Convert.ToBase64String(pcm, offset, length);
            stdin.WriteLine($"{{\"cmd\":\"audio\",\"pcm16_b64\":\"{b64}\"}}");
        }
        stdin.WriteLine("{\"cmd\":\"stop\"}");

        var partials = new List<string>();
        string finalText = "";
        bool sawFinal = false;
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            JsonElement evt = JsonDocument.Parse(line.Trim()).RootElement;
            string kind = evt.GetProperty("event").GetString()!;
            if (kind == "partial")
            {
                partials.Add(evt.GetProperty("text").GetString()!);
            }
            else if (kind == "final")
            {
                finalText = evt.GetProperty("text").GetString()!;
                sawFinal = true;
                break;
            }
            else if (kind == "error")
            {
                Assert.Fail($"host emitted error: {evt.GetProperty("message").GetString()}");
            }
        }
        Assert.True(sawFinal, "host 在 final 之前結束(stdout EOF)");

        stdin.Close();
        Assert.True(process.WaitForExit(30_000), "host 未在逾時內結束");
        process.CancelErrorRead();
        Assert.Equal(0, process.ExitCode);

        output.WriteLine($"partials observed: {partials.Count}");
        foreach (string stderrLine in stderrLines)
        {
            output.WriteLine($"[host stderr] {stderrLine}");
        }

        // task-3-report.md 遺留項(b):真引擎 Process() 每次回傳累積全文,HostSession 的
        // _lastEmitted 比對正是為此存在(只在文字真的變化時才發 partial)。這裡驗證觀察到
        // 的 partial 序列符合這個收斂語義:無相鄰重複、單調不縮短。
        // 先擋空集合:0 個 partial 會讓下方迴圈空轉,gating 行為等於沒驗證(vacuous pass)。
        Assert.True(partials.Count > 0, "real-model run produced no partial events — gating behavior unverified");
        for (int i = 1; i < partials.Count; i++)
        {
            Assert.NotEqual(partials[i - 1], partials[i]);
            Assert.True(
                partials[i].Length >= partials[i - 1].Length,
                $"partial #{i} 比前一個短(非單調成長):'{partials[i - 1]}' -> '{partials[i]}'");
        }

        string expectedNorm = NormalizeForAsr(ExpectedText);
        string gotNorm = NormalizeForAsr(finalText);
        double distance = NormalizedLevenshtein(expectedNorm, gotNorm);
        output.WriteLine($"expected (normalized): {expectedNorm}");
        output.WriteLine($"got (normalized):      {gotNorm}");
        output.WriteLine($"normalized edit distance: {distance:F4}");
        Assert.True(
            distance <= MaxNormalizedDistance,
            $"normalized edit distance {distance:F3} > {MaxNormalizedDistance}\n" +
            $"expected (normalized): {expectedNorm}\ngot (normalized): {gotNorm}");
    }

    // 與 StdioBlackBoxTests.LocateHostExecutable 相同的路徑推導,Release-only。
    private static string LocateHostExecutable()
    {
        var testAssemblyDir = Path.GetDirectoryName(typeof(RealModelTests).Assembly.Location)!;
        var appsDir = Path.GetFullPath(Path.Combine(testAssemblyDir, "..", "..", "..", ".."));
        var exeName = OperatingSystem.IsWindows() ? "nemotron-asr-host.exe" : "nemotron-asr-host";
        var exePath = Path.Combine(appsDir, "nemotron-asr-host", "bin", "Release", "net8.0", exeName);
        if (!File.Exists(exePath))
        {
            throw new FileNotFoundException(
                $"host executable not found at {exePath}; this harness requires the Release " +
                "configuration — build with -c Release first (dotnet test ... -c Release).",
                exePath);
        }
        return exePath;
    }

    private static string LocateFixture(string name)
    {
        var testAssemblyDir = Path.GetDirectoryName(typeof(RealModelTests).Assembly.Location)!;
        // .../apps/nemotron-asr-host.tests/bin/Release/net8.0 → repo root 往上 5 層。
        var repoRoot = Path.GetFullPath(Path.Combine(testAssemblyDir, "..", "..", "..", "..", ".."));
        var path = Path.Combine(repoRoot, "tests", "fixtures", name);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"fixture not found at {path}", path);
        }
        return path;
    }

    /// 讀 mono 16 kHz PCM16 WAV 的 data chunk raw bytes(協定走 PCM16 base64,不需轉 float)。
    /// 對應 crates/catcher-asr-host/tests/real_model.rs 的 read_wav_pcm16。
    private static byte[] ReadWavPcm16(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        int dataPos = -1;
        for (int i = 0; i + 4 <= bytes.Length; i++)
        {
            if (bytes[i] == (byte)'d' && bytes[i + 1] == (byte)'a'
                && bytes[i + 2] == (byte)'t' && bytes[i + 3] == (byte)'a')
            {
                dataPos = i;
                break;
            }
        }
        if (dataPos < 0)
        {
            throw new InvalidDataException($"WAV 檔缺少 data chunk:{path}");
        }
        int dataStart = dataPos + 8; // "data" + 4-byte chunk size
        return bytes[dataStart..];
    }

    /// 對應 real_model.rs 的 normalize_for_asr:比對前去除空白與標點
    /// (ASR 評測慣例;格式差異不是辨識錯誤)。
    private static string NormalizeForAsr(string text)
    {
        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
        {
            if (!char.IsWhiteSpace(c) && !IsPunct(c))
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    // 對應 Rust char::is_ascii_punctuation() 的定義範圍(!"#$%&'()*+,-./ : ;<=>?@ [\]^_` {|}~)。
    private static bool IsAsciiPunctuation(char c) =>
        (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');

    // 對應 real_model.rs 的 is_punct(含與 is_ascii_punctuation 重疊的 ?/!——逐語義移植,非新增)。
    private static bool IsPunct(char c) =>
        IsAsciiPunctuation(c) || c is '，' or '。' or '、' or '；' or '：' or '?' or '!'
            or '「' or '」' or '『' or '』' or '（' or '）' or '《' or '》' or '…' or '—' or '·' or '？' or '！';

    /// 對應 real_model.rs 的 normalized_levenshtein:字元級 Levenshtein 距離除以
    /// 較長字串的字元數(0.0 = 相同,1.0 = 完全不同)。
    private static double NormalizedLevenshtein(string a, string b)
    {
        if (a.Length == 0 && b.Length == 0)
        {
            return 0.0;
        }
        var prev = new int[b.Length + 1];
        var curr = new int[b.Length + 1];
        for (int j = 0; j <= b.Length; j++)
        {
            prev[j] = j;
        }
        for (int i = 0; i < a.Length; i++)
        {
            curr[0] = i + 1;
            for (int j = 0; j < b.Length; j++)
            {
                int cost = a[i] == b[j] ? 0 : 1;
                curr[j + 1] = Math.Min(Math.Min(prev[j] + cost, prev[j + 1] + 1), curr[j] + 1);
            }
            (prev, curr) = (curr, prev);
        }
        return (double)prev[b.Length] / Math.Max(a.Length, b.Length);
    }
}

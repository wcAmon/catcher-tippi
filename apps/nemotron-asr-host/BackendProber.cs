// DML/CPU 後端探測:去除 UI 相依版本,邏輯照抄
// apps/tippi-windows/Services/InferenceBackendLoader.cs 的 Probe() + Load() 決策路徑
// (那份程式碼另外處理 WPF 進度回報與磁碟快取,console host 不需要,故不連結整個檔案)。
// 探測音檔固定用 Assets\backend-probe.wav(csproj 連結自
// tests/fixtures/bang-wo-song-chu-zh-cn.wav,已知為 mono 16 kHz PCM16,免經 NAudio 轉檔)。
// 探測過程與結果一律寫 stderr——協定規定 stdout 只承載事件行。
using System.Diagnostics;
using Tippi.Windows.Services;

namespace NemotronAsrHost;

public static class BackendProber
{
    // 與 MainWindow 探測流程一致的 chunk 大小(略大於協定慣用的 1600,無關緊要,
    // 探測只在乎端到端一次轉錄能不能跑通、跑多快)。
    private const int ChunkSamples = 8_960;

    /// 依偏好探測並選定後端,回傳已就緒(可直接 BeginSession)的 engine。
    /// preference=Cpu:略過探測,直接載入 CPU。
    /// preference=DirectML:只測 DML,失敗回退 CPU。
    /// preference=Auto:兩者都測,交給 InferenceBackendPolicy.Select 決定。
    public static (NemotronEngine Engine, InferenceBackend Backend) Probe(
        string modelDirectory,
        InferenceBackendPreference preference)
    {
        if (preference == InferenceBackendPreference.Cpu)
        {
            Log("preference=cpu,略過探測。");
            return (LoadEngine(modelDirectory, InferenceBackend.Cpu), InferenceBackend.Cpu);
        }

        BackendProbeResult directMl = RunProbe(modelDirectory, InferenceBackend.DirectML);
        LogProbeResult("dml", directMl);

        if (preference == InferenceBackendPreference.DirectML)
        {
            InferenceBackend backend = directMl.Succeeded ? InferenceBackend.DirectML : InferenceBackend.Cpu;
            if (!directMl.Succeeded)
            {
                Log("preference=dml 但 GPU 探測失敗,回退 CPU。");
            }
            return (LoadEngine(modelDirectory, backend), backend);
        }

        // preference=auto:CPU 必須也測一次,InferenceBackendPolicy.Select 兩者都要。
        BackendProbeResult cpu = RunProbe(modelDirectory, InferenceBackend.Cpu);
        LogProbeResult("cpu", cpu);

        InferenceBackend selected = InferenceBackendPolicy.Select(preference, directMl, cpu);
        Log($"選定後端:{(selected == InferenceBackend.DirectML ? "dml" : "cpu")}" +
            $"(閾值 AutoGpuThreshold={InferenceBackendPolicy.AutoGpuThreshold})");
        return (LoadEngine(modelDirectory, selected), selected);
    }

    private static NemotronEngine LoadEngine(string modelDirectory, InferenceBackend backend)
    {
        try
        {
            return new NemotronEngine(modelDirectory, backend);
        }
        catch (Exception ex) when (backend == InferenceBackend.DirectML)
        {
            Log($"DirectML 載入失敗,回退 CPU:{CompactError(ex)}");
            return new NemotronEngine(modelDirectory, InferenceBackend.Cpu);
        }
    }

    private static BackendProbeResult RunProbe(string modelDirectory, InferenceBackend backend)
    {
        try
        {
            string probePath = Path.Combine(AppContext.BaseDirectory, "Assets", "backend-probe.wav");
            float[] probeAudio = ReadMono16KhzPcm16Wav(probePath);
            using var engine = new NemotronEngine(modelDirectory, backend);
            engine.BeginSession("zh-CN", useVad: false, traditionalChinese: false);

            var stopwatch = Stopwatch.StartNew();
            TranscriptionUpdate? latest = null;
            for (int offset = 0; offset < probeAudio.Length; offset += ChunkSamples)
            {
                int length = Math.Min(ChunkSamples, probeAudio.Length - offset);
                float[] chunk = new float[length];
                Array.Copy(probeAudio, offset, chunk, 0, length);
                latest = engine.Process(chunk) ?? latest;
            }
            latest = engine.Flush() ?? latest;
            stopwatch.Stop();

            if (latest is null || string.IsNullOrWhiteSpace(latest.RawText))
            {
                throw new InvalidDataException("後端探測沒有產生任何語音辨識文字。");
            }

            return new(backend, true, stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            return new(backend, false, TimeSpan.MaxValue, CompactError(ex));
        }
    }

    /// 讀 mono 16 kHz PCM16 WAV 的 data chunk,轉成 ±1.0 float samples。
    /// 最小 parser(不驗證 fmt chunk 的聲道/取樣率/位元深度——探測音檔與測試 fixture
    /// 已知固定是 mono/16kHz/16-bit,語義對齊 crates/catcher-asr-host/tests/real_model.rs
    /// 的 read_wav_pcm16)。
    private static float[] ReadMono16KhzPcm16Wav(string path)
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
        int sampleCount = (bytes.Length - dataStart) / 2;
        var samples = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++)
        {
            samples[i] = BitConverter.ToInt16(bytes, dataStart + i * 2) / 32768f;
        }
        return samples;
    }

    private static void LogProbeResult(string label, BackendProbeResult result) => Log(
        result.Succeeded
            ? $"{label} 探測成功:{result.Elapsed.TotalMilliseconds:F0} ms"
            : $"{label} 探測失敗:{result.Error}");

    private static string CompactError(Exception exception)
    {
        string message = exception.GetBaseException().Message.ReplaceLineEndings(" ").Trim();
        return message.Length <= 240 ? message : message[..240] + "…";
    }

    private static void Log(string message) => Console.Error.WriteLine($"[probe] {message}");
}

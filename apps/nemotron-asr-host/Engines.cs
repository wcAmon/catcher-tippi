// 推論引擎抽象。HostSession 只認識介面,讓狀態機在無模型環境可測(FakeEngine)。
// 語義與 mac host 的 AsrEngine trait 對齊,但 API 形狀貼合 NemotronEngine:
// Push 回傳「更新後的累積全文」(無新內容回 null),而非 token ids。
namespace NemotronAsrHost;

public interface IAsrEngine
{
    /// 會話開始:重置內部狀態,準備新 utterance。
    void Begin();
    /// 餵入 16 kHz mono float samples;回傳更新後的累積全文,無新內容回 null。
    string? Push(float[] samples);
    /// 沖洗解碼器,回傳定稿全文。
    string Finish();
    string Backend { get; }
}

/// 決定性假引擎:每滿 1600 samples 產出「字N」,語義與 mac FakeEngine 一致。
public sealed class FakeEngine : IAsrEngine
{
    private int _buffered;
    private int _nextId;
    private string _text = "";

    public string Backend => "fake";

    public void Begin()
    {
        _buffered = 0;
        _nextId = 0;   // 每會話歸零:第二會話必須重新從「字0」開始
        _text = "";
    }

    public string? Push(float[] samples)
    {
        _buffered += samples.Length;
        bool changed = false;
        while (_buffered >= 1600)
        {
            _buffered -= 1600;
            _text += $"字{_nextId++}";
            changed = true;
        }
        return changed ? _text : null;
    }

    public string Finish() => _text;
}

/// InferenceBackend enum → 協定線上字串("dml"/"cpu")的映射,獨立成 static method
/// 是為了讓 fast test 能不碰真引擎、不碰硬體地驗證這個映射本身(見
/// EnginesTests.BackendWireNameMapsDirectMlToDml / BackendWireNameMapsCpuToCpu)。
public static class BackendWireName
{
    public static string For(Tippi.Windows.Services.InferenceBackend backend) =>
        backend == Tippi.Windows.Services.InferenceBackend.DirectML ? "dml" : "cpu";
}

/// 真引擎:包裝既有 NemotronEngine(onnxruntime-genai)。
/// Begin = BeginSession(language, useVad:false, traditionalChinese:true)——
/// 繁體輸出與 mac host 的 opencc s2twp 對齊。
/// Backend 刻意不接受建構子參數,而是每次讀 inner.Backend 現算——inner 的 Backend
/// 屬性只在建構成功時被設成「實際套用的後端」(見 NemotronEngine.cs),因此這裡不可能
/// 存在「回報值」與「實際引擎」不一致的狀態(BackendProber.LoadEngine 的 DML→CPU
/// 靜默回退不再能繞過這個屬性——回退後建構出的是另一個 Backend=Cpu 的 NemotronEngine)。
public sealed class NemotronEngineAdapter(
    Tippi.Windows.Services.NemotronEngine inner,
    string language) : IAsrEngine
{
    public string Backend => BackendWireName.For(inner.Backend);

    public void Begin() => inner.BeginSession(language, useVad: false, traditionalChinese: true);

    public string? Push(float[] samples) => inner.Process(samples)?.DisplayText;

    public string Finish() => inner.Flush()?.DisplayText ?? "";
}

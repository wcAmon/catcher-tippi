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

/// 真引擎:包裝既有 NemotronEngine(onnxruntime-genai)。
/// Begin = BeginSession(language, useVad:false, traditionalChinese:true)——
/// 繁體輸出與 mac host 的 opencc s2twp 對齊。
public sealed class NemotronEngineAdapter(
    Tippi.Windows.Services.NemotronEngine inner,
    string language,
    Tippi.Windows.Services.InferenceBackend backend) : IAsrEngine
{
    public string Backend { get; } =
        backend == Tippi.Windows.Services.InferenceBackend.DirectML ? "dml" : "cpu";

    public void Begin() => inner.BeginSession(language, useVad: false, traditionalChinese: true);

    public string? Push(float[] samples) => inner.Process(samples)?.DisplayText;

    public string Finish() => inner.Flush()?.DisplayText ?? "";
}

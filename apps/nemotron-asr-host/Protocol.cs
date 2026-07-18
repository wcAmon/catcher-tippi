// asr-host protocol v1 的訊息編解碼。格式以 docs/protocol/asr-host-v1.md 為準,逐字不可改。
// 事件用字串插值而非 JsonSerializer:欄位順序與零空白是凍結格式的一部分,
// 插值讓格式在程式碼裡肉眼可驗;文字內容仍用 JsonEncodedText 正確跳脫。
using System.Text.Json;

namespace NemotronAsrHost;

public abstract record Command
{
    public sealed record Start(string Lang, int SampleRate) : Command;
    public sealed record Audio(string Pcm16B64) : Command;
    public sealed record Stop : Command;
}

public static class Protocol
{
    /// 解析一行指令。失敗時 error 帶原因(對應協定「無法解析的行 → error」)。
    public static bool TryParse(string line, out Command? command, out string error)
    {
        command = null;
        error = "";
        JsonDocument doc;
        try { doc = JsonDocument.Parse(line); }
        catch (JsonException e) { error = $"無法解析指令:{e.Message}"; return false; }
        using (doc)
        {
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object ||
                !root.TryGetProperty("cmd", out var cmdProp))
            { error = "無法解析指令:缺少 cmd"; return false; }
            string? cmd = cmdProp.GetString();
            var known = cmd switch
            {
                "start" => new[] { "cmd", "lang", "sample_rate" },
                "audio" => new[] { "cmd", "pcm16_b64" },
                "stop" => new[] { "cmd" },
                _ => null,
            };
            if (known is null) { error = $"無法解析指令:未知 cmd {cmd}"; return false; }
            foreach (var prop in root.EnumerateObject())
            {
                if (Array.IndexOf(known, prop.Name) < 0)
                { error = $"無法解析指令:未知欄位 {prop.Name}"; return false; }
            }
            switch (cmd)
            {
                case "start":
                    if (!root.TryGetProperty("lang", out var lang) ||
                        !root.TryGetProperty("sample_rate", out var rate) ||
                        lang.ValueKind != JsonValueKind.String ||
                        rate.ValueKind != JsonValueKind.Number)
                    { error = "無法解析指令:start 欄位不完整"; return false; }
                    command = new Command.Start(lang.GetString()!, rate.GetInt32());
                    return true;
                case "audio":
                    if (!root.TryGetProperty("pcm16_b64", out var pcm) ||
                        pcm.ValueKind != JsonValueKind.String)
                    { error = "無法解析指令:audio 欄位不完整"; return false; }
                    command = new Command.Audio(pcm.GetString()!);
                    return true;
                default:
                    command = new Command.Stop();
                    return true;
            }
        }
    }

    public static string EmitReady(string backend) =>
        $"{{\"event\":\"ready\",\"backend\":{Quote(backend)}}}";
    public static string EmitPartial(string text) =>
        $"{{\"event\":\"partial\",\"text\":{Quote(text)}}}";
    public static string EmitFinal(string text) =>
        $"{{\"event\":\"final\",\"text\":{Quote(text)}}}";
    public static string EmitError(string message) =>
        $"{{\"event\":\"error\",\"message\":{Quote(message)}}}";

    // DEVIATION(非編譯器強制,見 task-3-report.md):預設 JsonSerializerOptions 的
    // JavaScriptEncoder.Default 會把非 ASCII 字元(含中文)跳脫成 \uXXXX,
    // 與 mac serde_json 輸出的原始 UTF-8 位元組不對齊,破壞跨 host 逐字 parity。
    // 用 UnsafeRelaxedJsonEscaping 保留原始 UTF-8;JSON 必要跳脫(引號、反斜線、控制字元)不受影響。
    private static readonly JsonSerializerOptions QuoteOptions = new()
    {
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    private static string Quote(string value) =>
        JsonSerializer.Serialize(value, QuoteOptions); // 產生含雙引號的正確跳脫字串
}

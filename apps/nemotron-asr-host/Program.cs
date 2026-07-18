// nemotron-asr-host:tomato-ears 的 Windows engine host。
// 協定見 docs/protocol/asr-host-v1.md;本檔只做 stdio 轉發。
using NemotronAsrHost;
using Tippi.Windows.Services;

// stdout 必須是 raw UTF-8 JSON(mac host 對齊)。encoderShouldEmitUTF8Identifier: false:
// Encoding.UTF8 靜態實例帶 BOM,重導向 stdout 時可能把 EF BB BF 寫在 ready 行之前,
// 非 .NET 的 JSON-lines 消費端(Rust/Deno)會在第一行就解析失敗。
Console.OutputEncoding = new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
// 協定規定行尾是 \n;.NET 在 Windows 上 Console.Out 預設用 Environment.NewLine(\r\n)。
// 沒有這行,stdout 每個事件行尾會多一個 \r,破壞逐位元組協定 parity(見 task-3-report.md
// 「順帶觀察」;本行把該觀察轉成實際修復)。
Console.Out.NewLine = "\n";

string? modelDir = null;
string language = "auto";
string backendPref = "auto";
bool fakeEngine = false;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": modelDir = args[++i]; break;
        case "--language": language = args[++i]; break;
        case "--backend": backendPref = args[++i]; break;
        case "--fake-engine": fakeEngine = true; break;
        default:
            Console.Error.WriteLine($"unknown argument: {args[i]}");
            return 2;
    }
}

IAsrEngine engine;
if (fakeEngine)
{
    engine = new FakeEngine();
}
else
{
    if (modelDir is null)
    {
        Emit(Protocol.EmitError("缺少 --model 參數"));
        return 1;
    }

    InferenceBackendPreference preference;
    switch (backendPref)
    {
        case "auto": preference = InferenceBackendPreference.Auto; break;
        case "dml": preference = InferenceBackendPreference.DirectML; break;
        case "cpu": preference = InferenceBackendPreference.Cpu; break;
        default:
            Console.Error.WriteLine($"unknown --backend value: {backendPref} (expected auto|dml|cpu)");
            return 2;
    }

    try
    {
        (NemotronEngine nemotronEngine, InferenceBackend backend) = BackendProber.Probe(modelDir, preference);
        engine = new NemotronEngineAdapter(nemotronEngine, language, backend);
    }
    catch (Exception e)
    {
        // 協定:模型探測/載入失敗 → error + exit 1(ready 之前的唯一合法 stdout 行)。
        Emit(Protocol.EmitError($"引擎載入失敗:{e.Message}"));
        return 1;
    }
}

var session = new HostSession(engine);
Emit(Protocol.EmitReady(session.Backend));

string? line;
while (true)
{
    try { line = Console.In.ReadLine(); }
    catch (IOException) { break; }   // 協定:stdin 讀取錯誤視同 EOF
    if (line is null) break;
    if (line.Trim().Length == 0) continue;
    foreach (var evt in session.Handle(line))
    {
        Emit(evt);
    }
}
return 0;

static void Emit(string line)
{
    Console.Out.WriteLine(line);
    Console.Out.Flush();
}

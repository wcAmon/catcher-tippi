// nemotron-asr-host:tomato-ears 的 Windows engine host。
// 協定見 docs/protocol/asr-host-v1.md;本檔只做 stdio 轉發。
using NemotronAsrHost;

Console.OutputEncoding = System.Text.Encoding.UTF8; // stdout 必須是 raw UTF-8 JSON(mac host 對齊)

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
    // Task 4 接上 NemotronEngineAdapter;在那之前回報未接上(致命 → exit 1)。
    Emit(Protocol.EmitError("Nemotron engine 尚未接上,請用 --fake-engine"));
    return 1;
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

# nemotron-asr-host(Windows Engine Host)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 從 `codex/windows-auto-backend` 的既有資產抽出 Windows console engine host,講與 mac 完全相同的 asr-host-v1 協定(onnxruntime-genai + DirectML 實測探測→CPU),並升級 mac 端真模型測試斷言(edit-distance);產出可發布的 zip + SHA-256。

**Architecture:** 新 .NET 8 console 專案 `apps/nemotron-asr-host`,三層分離與 mac host 對齊:Protocol(System.Text.Json 型別)/ HostSession(狀態機,鏡射 Rust `session.rs` 語義)/ IAsrEngine(FakeEngine 供無模型測試 + NemotronEngineAdapter 連結既有 `NemotronEngine.cs`)。開發在 mac 本地(寫碼、commit、push),建置與測試透過 ssh 在 Windows 機器執行(pull → dotnet test / publish)。

**Tech Stack:** .NET 8(dotnet-install.ps1 user-local,免管理員)、System.Text.Json、xUnit、Microsoft.ML.OnnxRuntimeGenAI.Managed 0.13 + pinned Runtime DLLs(既有)、OpenccNetLib(s2twp)。

**相關 spec:** `docs/superpowers/specs/2026-07-18-tomato-ears-design.md` §3.2、§3.3;協定:`docs/protocol/asr-host-v1.md`(凍結,逐字遵守)

## Global Constraints

- 分支:`feat/nemotron-asr-host`(base = codex/windows-auto-backend + merge main);Windows 端 clone 於 `C:\Users\i5491\catcher-tippi`
- Windows 機器:`ssh i5491@100.91.128.2`,遠端 shell 是 **cmd.exe**(用 `&` 串接,不是 `;`);PowerShell 指令用 `powershell -NoProfile -Command "..."` 包裝;dotnet 以 `%USERPROFILE%\dotnet\dotnet.exe` 全路徑呼叫
- 協定 asr-host-v1 逐字:指令 start/audio/stop(未知欄位拒收)、事件 ready/partial/final/error;`backend` 值 Windows 為 `"dml"` 或 `"cpu"`(fake 引擎回 `"fake"`);start 成功靜默;partial 為累積全文且僅在內容變化時發;stop 後 final 恰一次;stop 失敗→error 且會話結束無 final;stdin EOF/讀取錯誤→正常結束 exit 0;致命錯誤(模型載入失敗)→error 後 exit 1;`start.lang` v1 保留欄位被接受但忽略,語言由 `--language` 決定
- 音訊:mono 16 kHz PCM16-LE;`sample_rate` 僅接受 16000;PCM16→float 為 `value / 32768f`;奇數位元組→error 且會話保留
- FakeEngine 語義與 mac 完全一致:每滿 1600 samples 產生一個「字N」(N 從 0 遞增,每會話歸零),partial/final 為累積串接
- 中文輸出一律繁體(OpenccNetLib s2twp),與 mac host 的 `opencc::to_traditional` 對齊
- 後端選擇:沿用 `InferenceBackendPolicy.Select`(DML 與 CPU 都實測,DML 需贏 0.85 門檻);`--backend auto|dml|cpu` 對應 `InferenceBackendPreference`
- Windows 模型:`onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4`(檔案清單與 SHA-256 以 `apps/tippi-windows/Services/ModelManifest.cs` 為準)
- mac 端指令在 repo 根(worktree `/Users/wake/Desktop/catcher-tippi/.worktrees/nemotron-asr-host`)執行;Rust 測試只跑 `cargo test -p catcher-asr-host`
- 每個 commit message 附 Claude trailer(與 Plan 1 相同)

---

### Task 1: mac 真模型斷言強化(edit-distance + 中文 fixture)

**Files:**
- Modify: `crates/catcher-asr-host/tests/real_model.rs`(替換 coverage 斷言)

**Interfaces:**
- Produces: `normalized_levenshtein(a: &str, b: &str) -> f64`(0.0 = 相同);斷言 `distance ≤ 0.25`。Windows 的 xUnit 真模型測試(Task 4)採同一 metric 與門檻。

- [ ] **Step 1: 查 fixture 真實文本**

讀 `tests/fixtures/README.md` 確認 `bang-wo-song-chu-zh-tw.wav` 的實際語音內容(預期為「幫我送出」;若 README 記載不同,以 README 為準)。將確認結果記入報告。

- [ ] **Step 2: 改寫斷言(紅燈:先確認舊斷言存在後直接替換——本 task 為測試品質升級,紅燈定義為「新 metric 對舊程式碼不存在」的編譯錯誤)**

把 `tests/real_model.rs` 中 coverage 區塊(`let hits = ...` 至 `assert!(coverage >= 0.6, ...)`)替換為:

```rust
    // 正規化編輯距離:0.0 = 完全相同。比 presence-based 覆蓋率嚴格——
    // 亂碼即使字元集重疊也會因插入/替換代價而距離飆高。
    let distance = normalized_levenshtein(&expected, &final_text);
    assert!(
        distance <= 0.25,
        "normalized edit distance {distance:.3} > 0.25\nexpected: {expected}\ngot: {final_text}"
    );
```

檔尾加:

```rust
/// 字元級 Levenshtein 距離除以較長字串的字元數(0.0 = 相同,1.0 = 完全不同)。
fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(curr[j] + 1);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()] as f64 / a.len().max(b.len()) as f64
}
```

並在測試函式的檔頭 doc-comment 補上第二組(中文)執行範例:

```text
//!   CATCHER_ASR_FIXTURE_WAV=tests/fixtures/bang-wo-song-chu-zh-tw.wav \
//!   CATCHER_ASR_FIXTURE_TEXT="幫我送出" \  (以 Step 1 確認的文本為準)
```

- [ ] **Step 3: 驗證(兩組 fixture 各跑一次 gated 測試)**

```bash
CATCHER_ASR_MODEL_DIR="/Users/wake/Library/Application Support/Tippi/Models/catcher-asr-mlx-int8" \
CATCHER_ASR_FIXTURE_WAV="$PWD/tests/fixtures/hello-streaming.wav" \
CATCHER_ASR_FIXTURE_TEXT="Hello, this is a streaming speech recognition test" \
cargo test -p catcher-asr-host --test real_model -- --ignored
```

Expected: PASS。再以 `bang-wo-song-chu-zh-tw.wav` + Step 1 確認的中文文本跑一次,Expected: PASS(記錄兩次的實際 distance)。另跑 `cargo test -p catcher-asr-host` 確認 14/14 不受影響。

- [ ] **Step 4: Commit**

```bash
git add crates/catcher-asr-host/tests/real_model.rs
git commit -m "test: replace presence coverage with normalized edit distance"
```

---

### Task 2: Windows 機器 bootstrap(.NET SDK + repo clone)

**Files:**
- Create: `scripts/bootstrap-windows-host.md`(記錄 bootstrap 指令與環境快照,供重建)

**Interfaces:**
- Produces: Windows 機器上 `%USERPROFILE%\dotnet\dotnet.exe`(SDK 8.x)與 `C:\Users\i5491\catcher-tippi`(checkout `feat/nemotron-asr-host`)。後續 task 的 ssh 驗證迴圈依賴這兩者。

- [ ] **Step 1: push 分支**

```bash
git push -u origin feat/nemotron-asr-host
```

- [ ] **Step 2: 安裝 .NET 8 SDK(user-local,免管理員)**

```bash
ssh i5491@100.91.128.2 "powershell -NoProfile -ExecutionPolicy Bypass -Command \"iwr https://dot.net/v1/dotnet-install.ps1 -OutFile $env:TEMP\\di.ps1; & $env:TEMP\\di.ps1 -Channel 8.0 -InstallDir $env:USERPROFILE\\dotnet\""
ssh i5491@100.91.128.2 "%USERPROFILE%\\dotnet\\dotnet.exe --version"
```

Expected: `8.0.x`。(若 PowerShell 引號經 ssh 轉義出錯,改為兩段:先把 install 指令寫成遠端 `%TEMP%\bootstrap.ps1` 再執行——實際可行寫法由實作者測定,結果記入 bootstrap 文件。)

- [ ] **Step 3: clone repo(public,免認證)**

```bash
ssh i5491@100.91.128.2 "git clone --branch feat/nemotron-asr-host https://github.com/wcAmon/catcher-tippi.git C:\\Users\\i5491\\catcher-tippi"
ssh i5491@100.91.128.2 "cd /d C:\\Users\\i5491\\catcher-tippi & git log --oneline -1"
```

Expected: HEAD = 本地分支尖端。

- [ ] **Step 4: 建置冒煙(既有 WPF app 專案能 restore/build 即證明 SDK + NuGet 通)**

```bash
ssh i5491@100.91.128.2 "cd /d C:\\Users\\i5491\\catcher-tippi & %USERPROFILE%\\dotnet\\dotnet.exe build apps\\tippi-windows\\Tippi.Windows.csproj -c Release"
```

Expected: Build succeeded(warnings 容忍)。

- [ ] **Step 5: 寫 bootstrap 文件並 commit**

`scripts/bootstrap-windows-host.md`:記錄機器規格(Win11 26200、RTX 4060 Laptop + Iris Xe、16GB、20 threads)、SDK 安裝方式與版本、clone 位置、驗證輸出。

```bash
git add scripts/bootstrap-windows-host.md
git commit -m "docs: windows host machine bootstrap record"
git push
```

---

### Task 3: console host(Protocol + HostSession + FakeEngine + stdio 迴圈 + 測試)

**Files:**
- Create: `apps/nemotron-asr-host/NemotronAsrHost.csproj`
- Create: `apps/nemotron-asr-host/Protocol.cs`
- Create: `apps/nemotron-asr-host/Engines.cs`(本 task 只有 IAsrEngine + FakeEngine;真引擎在 Task 4)
- Create: `apps/nemotron-asr-host/HostSession.cs`
- Create: `apps/nemotron-asr-host/Program.cs`
- Create: `apps/nemotron-asr-host.tests/NemotronAsrHost.Tests.csproj`
- Create: `apps/nemotron-asr-host.tests/ProtocolTests.cs`
- Create: `apps/nemotron-asr-host.tests/HostSessionTests.cs`
- Create: `apps/nemotron-asr-host.tests/StdioBlackBoxTests.cs`

**Interfaces:**
- Consumes: 協定 v1(docs/protocol/asr-host-v1.md)
- Produces: `IAsrEngine`(`void Begin()` / `string? Push(float[] samples)`(回傳更新後的累積全文,無新內容回 null)/ `string Finish()` / `string Backend { get; }`);`HostSession.Handle(string line) -> IReadOnlyList<string>`(輸入一行原始 JSON,輸出零至多行已序列化事件——Task 4 的真引擎與 Program 皆依賴此簽名)

- [ ] **Step 1: 專案骨架**

`apps/nemotron-asr-host/NemotronAsrHost.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <AssemblyName>nemotron-asr-host</AssemblyName>
    <RootNamespace>NemotronAsrHost</RootNamespace>
    <PlatformTarget>x64</PlatformTarget>
    <InvariantGlobalization>true</InvariantGlobalization>
  </PropertyGroup>
</Project>
```

(真引擎的套件與 DLL 連結在 Task 4 加入,本 task 保持零外部相依。)

`apps/nemotron-asr-host.tests/NemotronAsrHost.Tests.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <IsPackable>false</IsPackable>
    <PlatformTarget>x64</PlatformTarget>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.11.1" />
    <PackageReference Include="xunit" Version="2.9.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.8.2" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include="..\nemotron-asr-host\NemotronAsrHost.csproj" />
  </ItemGroup>
</Project>
```

- [ ] **Step 2: 寫 failing 測試(節錄核心;完整清單見下)**

`ProtocolTests.cs` 斷言與 mac `protocol.rs` 測試一一對應:parse start/audio/stop、未知 cmd 與非 JSON 拒收、**多餘欄位拒收**、四種事件序列化為逐字字串(`{"event":"ready","backend":"mlx"}` 樣式,雙引號、無空白、欄位順序 event 先)。

`HostSessionTests.cs` 與 mac `session.rs` 測試一一對應(用 FakeEngine):happy path(1600 samples → partial "字0" → stop → final "字0")、不足 1600 無 partial、狀態錯誤不殺會話(stop/audio 無 start、start-while-active,之後同會話仍能 partial)、壞 base64 與奇數位元組 → error 且會話保留、sample_rate≠16000 拒收、**第二會話從「字0」重新開始**、**Finish 失敗 → error 無 final 且會話已結束**(測試用 FailingEngine)。

`StdioBlackBoxTests.cs` 對應 mac `tests/stdio.rs`:以 `Process` 啟動建置產物(`dotnet run --no-build` 或直接執行 bin 下的 exe,由實作者擇一並固定),`--fake-engine` 模式走完整協定(ready→start→audio→partial→stop→final→EOF exit 0)、壞行→error 且行程續活、**單行程兩個完整會話**。

```csharp
// ProtocolTests.cs 核心樣例(其餘同型)
[Fact]
public void EmitsReadyVerbatim()
{
    Assert.Equal("{\"event\":\"ready\",\"backend\":\"fake\"}", Protocol.EmitReady("fake"));
}

[Fact]
public void RejectsUnknownField()
{
    Assert.False(Protocol.TryParse(
        "{\"cmd\":\"start\",\"lang\":\"auto\",\"sample_rate\":16000,\"x\":1}", out _, out _));
}
```

- [ ] **Step 3: 實作 Protocol.cs**

```csharp
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

    private static string Quote(string value) =>
        JsonSerializer.Serialize(value); // 產生含雙引號的正確跳脫字串
}
```

- [ ] **Step 4: 實作 Engines.cs(IAsrEngine + FakeEngine)**

```csharp
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
```

- [ ] **Step 5: 實作 HostSession.cs(鏡射 Rust session.rs 全部語義)**

```csharp
// 會話狀態機:一行指令 → 零至多行已序列化事件。不做 I/O(Program.cs 負責)。
// 語義鏡射 crates/catcher-asr-host/src/session.rs;差異僅在引擎 API 形狀
// (Push 直接回全文,因此不需要另外 decode)。
namespace NemotronAsrHost;

public sealed class HostSession(IAsrEngine engine)
{
    private bool _active;
    private string _lastEmitted = "";

    public string Backend => engine.Backend;

    public IReadOnlyList<string> Handle(string line)
    {
        if (!Protocol.TryParse(line, out var command, out var parseError))
        {
            return [Protocol.EmitError(parseError)];
        }
        switch (command)
        {
            case Command.Start start when start.SampleRate != 16000:
                return [Protocol.EmitError($"sample_rate 僅支援 16000,收到 {start.SampleRate}")];
            case Command.Start when _active:
                return [Protocol.EmitError("會話進行中,請先 stop")];
            case Command.Start:
                // start.lang 為 v1 保留欄位:接受但忽略(語言由 --language 決定)。
                try { engine.Begin(); }
                catch (Exception e) { return [Protocol.EmitError($"引擎重置失敗:{e.Message}")]; }
                _active = true;
                _lastEmitted = "";
                return [];
            case Command.Audio audio when !_active:
                return [Protocol.EmitError("尚未 start")];
            case Command.Audio audio:
            {
                float[] samples;
                try { samples = DecodePcm16(audio.Pcm16B64); }
                catch (FormatException) { return [Protocol.EmitError("pcm16_b64 非法 base64")]; }
                catch (ArgumentException e) { return [Protocol.EmitError(e.Message)]; }
                string? text;
                try { text = engine.Push(samples); }
                catch (Exception e) { return [Protocol.EmitError(e.Message)]; }
                if (text is null || text == _lastEmitted)
                {
                    return [];   // 協定:partial 僅在內容變化時輸出
                }
                _lastEmitted = text;
                return [Protocol.EmitPartial(text)];
            }
            case Command.Stop when !_active:
                return [Protocol.EmitError("尚未 start")];
            case Command.Stop:
            {
                _active = false;   // 協定:stop 失敗 → error 且會話視為已結束、無 final
                try { return [Protocol.EmitFinal(engine.Finish())]; }
                catch (Exception e) { return [Protocol.EmitError(e.Message)]; }
            }
            default:
                return [Protocol.EmitError("未知指令")];
        }
    }

    /// PCM16-LE bytes → float samples(±1.0)。奇數位元組數為格式錯誤。
    private static float[] DecodePcm16(string b64)
    {
        byte[] bytes = Convert.FromBase64String(b64);
        if (bytes.Length % 2 != 0)
        {
            throw new ArgumentException("pcm16_b64 位元組數必須為偶數");
        }
        var samples = new float[bytes.Length / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = BitConverter.ToInt16(bytes, i * 2) / 32768f;
        }
        return samples;
    }
}
```

- [ ] **Step 6: 實作 Program.cs(stdio 迴圈;真引擎分支本 task 回報未接上)**

```csharp
// nemotron-asr-host:tomato-ears 的 Windows engine host。
// 協定見 docs/protocol/asr-host-v1.md;本檔只做 stdio 轉發。
using NemotronAsrHost;

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
```

- [ ] **Step 7: push 後在 Windows 建置並跑測試(紅→綠證據以此為準)**

```bash
git add apps/nemotron-asr-host apps/nemotron-asr-host.tests
git commit -m "feat: nemotron-asr-host console skeleton with protocol v1 and fake engine"
git push
ssh i5491@100.91.128.2 "cd /d C:\\Users\\i5491\\catcher-tippi & git pull & %USERPROFILE%\\dotnet\\dotnet.exe test apps\\nemotron-asr-host.tests\\NemotronAsrHost.Tests.csproj -c Release 2>&1"
```

Expected: 全數通過(protocol + session + 黑箱;數量依實作,至少 protocol 6 / session 8 / 黑箱 3)。修到綠燈為止,綠燈後把最終版 commit+push(可 amend)。

---

### Task 4: 真引擎(NemotronEngineAdapter + 探測)+ Windows 真模型驗證

**Files:**
- Modify: `apps/nemotron-asr-host/NemotronAsrHost.csproj`(加 OnnxRuntimeGenAI.Managed 0.13.0 + OpenccNetLib 1.6.0 套件、連結 `..\tippi-windows\Runtime\win-x64\*.dll` 為 Content、連結 `..\..\tests\fixtures\bang-wo-song-chu-zh-cn.wav` 為 `Assets\backend-probe.wav`——寫法照抄 `apps/tippi-windows/Tippi.Windows.csproj` 對應區塊)
- Modify: `apps/nemotron-asr-host/Engines.cs`(加 `NemotronEngineAdapter`)
- Modify: `apps/nemotron-asr-host/Program.cs`(接上真引擎分支)
- Create: `apps/nemotron-asr-host/BackendProber.cs`
- Create: `scripts/fetch-nemotron-onnx-model.ps1`(Windows 端模型下載+SHA-256 驗證)

**Interfaces:**
- Consumes: `Tippi.Windows.Services.NemotronEngine`、`InferenceBackend`/`InferenceBackendPreference`/`InferenceBackendPolicy`/`BackendProbeResult`——以 `<Compile Include>` 連結 `..\tippi-windows\Services\NemotronEngine.cs` 與 `..\tippi-windows\Services\InferenceBackend.cs` 兩檔(不複製程式碼)
- Produces: `NemotronEngineAdapter : IAsrEngine`(backend 回 `"dml"` 或 `"cpu"`);`BackendProber.Probe(modelDir, preference) -> (NemotronEngine, InferenceBackend)`

- [ ] **Step 1: csproj 接線 + Adapter + Prober**

`NemotronEngineAdapter` 把 `NemotronEngine` 包成 `IAsrEngine`:

```csharp
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
```

`BackendProber`:照抄 `MainWindow.xaml.cs` 的探測流程(讀 `Assets\backend-probe.wav`、兩後端各實測載入+轉錄、`InferenceBackendPolicy.Select`),但去掉 UI 相依,回傳選定的 engine 與 backend。`--backend auto|dml|cpu` 映射 `InferenceBackendPreference`。Program.cs 真引擎分支:probe → adapter → ready(backend "dml"/"cpu");probe/載入失敗 → error + exit 1。

注意:HostSession 對 Push 的「內容變化才發 partial」判斷已在 Task 3 落地(`_lastEmitted` 比對),真引擎的 `Process` 每次回傳累積全文,不需再改 session。

- [ ] **Step 2: 模型下載腳本**

`scripts/fetch-nemotron-onnx-model.ps1`:從 `apps/tippi-windows/Services/ModelManifest.cs` 的檔案清單(name/size/sha256)產生下載迴圈——`Invoke-WebRequest https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/main/<name>` 到 `C:\Users\i5491\catcher-tippi-models\nemotron-onnx-int4\`,逐檔 `Get-FileHash -Algorithm SHA256` 比對 manifest 值,不符即刪檔報錯。冪等:已存在且 hash 相符則跳過。(檔案清單以 ModelManifest.cs 實際內容為準,實作者展開成腳本內的靜態表。)

- [ ] **Step 3: Windows 端全套驗證**

```bash
git add -A && git commit -m "feat: wire NemotronEngine with DML/CPU probe into asr host" && git push
ssh i5491@100.91.128.2 "cd /d C:\\Users\\i5491\\catcher-tippi & git pull & %USERPROFILE%\\dotnet\\dotnet.exe test apps\\nemotron-asr-host.tests\\NemotronAsrHost.Tests.csproj -c Release 2>&1"
ssh i5491@100.91.128.2 "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\Users\\i5491\\catcher-tippi\\scripts\\fetch-nemotron-onnx-model.ps1"
```

然後真模型端到端(黑箱,協定驅動):在 Windows 上把 `tests/fixtures/hello-streaming.wav` 的 PCM 轉 base64 chunks 餵給發布的 host(`--model C:\Users\i5491\catcher-tippi-models\nemotron-onnx-int4`),收 final,與 "Hello, this is a streaming speech recognition test" 做 normalized edit distance ≤ 0.25(門檻與 Task 1 相同)。實作方式:xUnit 加一個 `[Fact(Skip=...)]`→env-gated 的 `RealModelTests.cs`(gating 樣式仿 mac:環境變數 `NEMOTRON_ASR_MODEL_DIR` 缺席即 skip),在 ssh 上以 `NEMOTRON_ASR_MODEL_DIR=... dotnet test --filter RealModel` 執行。**記錄探測結果(DML vs CPU 基準毫秒數與勝出者)**——RTX 4060 預期 DML 勝。

Expected: 測試全綠 + 真模型 final 命中門檻 + probe 選擇 DML。

- [ ] **Step 4: Commit + push(綠燈版)**

---

### Task 5: 發布打包(publish + zip + SHA-256)

**Files:**
- Create: `scripts/build-nemotron-asr-host.ps1`

**Interfaces:**
- Produces: `dist\nemotron-asr-host-v0.1.0-windows-x64.zip` + `.zip.sha256`(bare filename 格式,與 mac 的 sha256 檔一致可獨立驗證);tomato-ears 配方 manifest(Plan 3)引用。

- [ ] **Step 1: 打包腳本**

```powershell
# 打包 nemotron-asr-host 為可發布的 zip 並產生 SHA-256(bare filename)。
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$dotnet = Join-Path $env:USERPROFILE "dotnet\dotnet.exe"
& $dotnet publish apps\nemotron-asr-host\NemotronAsrHost.csproj -c Release -r win-x64 `
    --self-contained -o publish\nemotron-asr-host
if ($LASTEXITCODE -ne 0) { throw "publish failed" }
$version = "0.1.0"
$name = "nemotron-asr-host-v$version-windows-x64"
New-Item -ItemType Directory -Force -Path dist | Out-Null
Copy-Item docs\protocol\asr-host-v1.md publish\nemotron-asr-host\PROTOCOL.md
Compress-Archive -Path publish\nemotron-asr-host\* -DestinationPath "dist\$name.zip" -Force
$hash = (Get-FileHash "dist\$name.zip" -Algorithm SHA256).Hash.ToLower()
"$hash  $name.zip" | Out-File -Encoding ascii "dist\$name.zip.sha256"
Write-Host "done: dist\$name.zip"
Write-Host "sha256: $hash"
```

- [ ] **Step 2: 在 Windows 執行、驗證 zip 內容(exe + DLLs + PROTOCOL.md + probe wav + Licenses)、冒煙(`--fake-engine` 兩行輸出)、記錄 sha256、commit+push 腳本**

- [ ] **Step 3: 發布(需 wake 同意後執行)**

`gh release create nemotron-asr-host-v0.1.0`(zip + sha256 從 Windows 取回 mac 或直接以 Windows gh?token 無效——從 Windows `scp` 回 mac 後用 mac 的 gh 發布)。執行前向 wake 確認。

---

## 後續計畫(本檔不含)

- **Plan 3** tomato-ears Deno 配方包(兩個 host release + 兩個模型 artifact 齊備後)
- **Plan 4** tmuh.ai mini-app store 實作文件

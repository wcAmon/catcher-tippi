# tomato-ears PLAN

給組裝 agent 的分階段建造指令。你的角色是**組裝工,不是工程師**(店規第 2
節):`reference/` 已經是 95% 完成、帶詳細註解、通過測試的成品,你只做四件
事——探測環境、下載並驗 hash、寫少量黏合(其實這個配方連黏合都不用你寫,
`reference/setup.ts`/`main.ts` 已經是完整的進入點)、跑驗收測試。

**在你開始之前:如果你不確定某個步驟該不該做,先讀完整份 PLAN.md 再動手**
——尤其是 Step 1 的目錄結構,寫錯會讓後面所有步驟都失敗在意料之外的地方。

> **已修復**:`asr-host-v0.1.0` 有一個上游打包缺陷——執行時會印出
> `MLX error: Failed to load the default metallib. library not found`
> 然後退出(release 只含 binary,沒有隨附 colocated `mlx.metallib`)。
> `asr-host-v0.1.1` 已修復(隨附 colocated metallib,手法對齊
> `apps/tippi/scripts/build-app.sh:20-22`),`manifest.json` 目前 pin 住
> 的就是 v0.1.1。如果你在 Step 4(verify)遇到上面那個確切錯誤訊息,
> 代表你手上的 engine 版本還是 v0.1.0——先確認 `<APP>/bin/engine-host`
> 是不是舊的殘留(例如 setup 中途被舊快取蓋掉),重跑 `deno task
> setup:mac` 讓它重新下載 pin 住的版本;若重跑後仍遇到同樣錯誤,才需要
> 停下來回報。Windows engine host 目前未知是否有同類問題。

## Step 0:讀 machine-profile,不要重新探測、不要重新安裝環境基座

在做任何 tomato-ears 專屬的事情之前:

1. 確認 `~/tmuh-apps/_machine/machine-profile.json` 存在。
   - **存在** → 直接讀它的 `os`/`arch` 欄位,跳到 Step 1。**不要**重新安裝
     Deno、**不要**重新跑硬體探測腳本,即使你不記得上次是誰跑的。
   - **不存在** → 先完整跑過 `recipes/env-base/RECIPE.md`(這是所有
     mini-app 共用的一次性環境基座,只需要跑一次),跑完再回來。
2. 確認你自己的持久記憶(CLAUDE.md/memory)裡是否已經記錄「env-base 已
   完成」——如果有,這一步到此為止;如果沒有但檔案確實存在,把這個事實
   寫進你的持久記憶(避免你自己下一個 session 又重跑一次)。

**為什麼這一步寫在最前面且加粗**:店規(`docs/superpowers/specs/2026-07-18-mini-app-store-design.md`
第 5 節)的核心設計是把「環境探測」這個 agent 建構期最大的 bug 來源,從
「每個 app 攤還一次」變成「全站只付一次」。跳過這條規則、每次都重新探測,
不會讓安裝更可靠,只會浪費時間並增加意外出錯的機率。

## Step 1:把配方檔案複製到安裝目錄

**這一步的目錄結構容易做錯,請逐字照做,不要自己憑直覺調整。**

先確認平台:讀 Step 0 拿到的 `machine-profile.json` 的 `os`/`arch`——
`darwin`+`aarch64` → 以下用 `mac`;`windows`+`x86_64`(或 `x64`)→ 以下用
`win`。其他組合這個配方不支援,停下來回報。

安裝目錄慣例(與 env-base 一致):`~/tmuh-apps/tomato-ears/`。下面用
`<RECIPE>` 代表**這份 PLAN.md 所在的資料夾**(也就是
`recipes/tomato-ears/` 本身,不論你是從完整 repo 拿到它,還是從配方
bundle 解壓出來的獨立資料夾),`<APP>` 代表 `~/tmuh-apps/tomato-ears/`。

```bash
mkdir -p "<APP>"
cp "<RECIPE>/manifest.json" "<RECIPE>/deno.json" "<RECIPE>/deno.lock" "<APP>/"

# reference/ 的 *.ts 檔案原樣複製(包含 *_test.ts——那些是開發期測試,
# 不會被 verify 或 start 執行到,留著無妨,不要花時間篩選)
mkdir -p "<APP>/reference"
cp "<RECIPE>"/reference/*.ts "<APP>/reference/"

# 重要:ui/ 在原始碼樹裡是 reference/ui/ 的子目錄，但安裝目錄要求 ui/
# 直接在 <APP> 根部（跟 reference/ 同一層），不是 <APP>/reference/ui/。
# server.ts 寫死 serve "${appDir}/ui"，放錯位置整個服務會回應 404。
mkdir -p "<APP>/ui"
cp "<RECIPE>"/reference/ui/* "<APP>/ui/"

# verify/ 原樣整份複製（含 fixtures/ 子目錄的 wav 檔）
cp -r "<RECIPE>/verify" "<APP>/verify"
```

Windows（cmd）對照（`cp`→`copy`，`cp -r`→`xcopy /E /I`，`mkdir -p`→
`mkdir`；`<RECIPE>`/`<APP>` 意義同上，換成實際反斜線路徑）：

```cmd
mkdir "<APP>"
copy "<RECIPE>\manifest.json" "<APP>\"
copy "<RECIPE>\deno.json" "<APP>\"
copy "<RECIPE>\deno.lock" "<APP>\"

mkdir "<APP>\reference"
copy "<RECIPE>\reference\*.ts" "<APP>\reference\"

mkdir "<APP>\ui"
copy "<RECIPE>\reference\ui\*" "<APP>\ui\"

xcopy /E /I "<RECIPE>\verify" "<APP>\verify"
```

`xcopy /E /I` 的 `/E` 遞迴複製含空目錄的子目錄結構、`/I` 在目的地不存在
時把它當成「要建立的目錄」（沒有這個旗標 xcopy 會反問你目的地是檔案還是
目錄，卡住互動）；`verify\` 底下的 `fixtures\` 子目錄需要這兩個旗標才會
一起複製過去。

**Task 6 Windows 演練確認**：逐一翻譯後 `dir <APP>` 核對，目錄結構與上方
「安裝完成後」的預期一致，無卡點；完整逐字稿見
`.superpowers/sdd/task-6-rehearsal-log.md`「階段二 Step 1」。

安裝完成後,`<APP>` 的結構應該長這樣(`bin/`/`model/`/`download/` 是
Step 3 才會出現,現在還不存在):

```
~/tmuh-apps/tomato-ears/
  manifest.json
  deno.json
  deno.lock
  reference/
    downloader.ts  downloader_test.ts
    engine.ts      engine_test.ts
    main.ts        main_test.ts
    server.ts      server_test.ts
    setup.ts
    permissions_probe_test.ts
  ui/
    index.html  app.js  downsampler-worklet.js  downsampler-core.js
    style.css   downsampler-core_test.ts
  verify/
    asr_metric.ts       asr_metric_test.ts
    integrity_test.ts   protocol_test.ts   service_test.ts
    binding_test.ts     permissions_test.ts
    real_service.ts     wav.ts
    fixtures/hello-streaming.wav
```

**常見錯誤**

| 症狀                                                                               | 原因                                                                                   | 解法                                                                                                                   |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `deno task setup:mac` 一開始就報 `Module not found "file:///…/reference/setup.ts"` | `<APP>/deno.json` 或 `<APP>/reference/` 沒複製對                                       | 確認 `<APP>` 下真的有 `deno.json` 且 `reference/setup.ts` 存在;`deno task` 的腳本路徑是相對 `deno.json` 所在目錄解析的 |
| `deno task start:mac` 跑起來,但瀏覽器/`curl` 打 `http://127.0.0.1:43117/` 得到 404 | `ui/` 被複製到 `<APP>/reference/ui/` 而不是 `<APP>/ui/`(最容易犯的錯,上面已用粗體標注) | 用 `ls <APP>` 確認 `ui/` 跟 `reference/`、`manifest.json` 是同一層;不是 `reference/ui/`                                |

## Step 2:切換到安裝目錄

```bash
cd "<APP>"
```

**之後所有 `deno task` 指令都要在這個目錄底下執行,不能在別的目錄下用
`deno task --cwd <APP> ...` 之類的方式繞過。** 原因(完整版見
SECURITY.md 第 1.1 節):`deno task` 定義上以 `deno.json` 所在目錄為工作
目錄(cwd)執行,而 `manifest.json`/`deno.json` 裡宣告的每一個權限旗標
(`--allow-read=.`、`--allow-run=bin/engine-host` 等)都是**相對這個 cwd**
解析的相對路徑。cwd 不對,權限旗標會指向錯誤的位置,行為會是各種難以
理解的 `NotCapable`。

## Step 3:安裝相依(`deno task setup:mac` / `setup:win`)

```bash
deno task setup:mac      # Windows 用 setup:win
```

這一步會下載 engine host(mac ≈4.2 MB、Windows 較大,見 `manifest.json`
的 `byteCount`)與模型檔案(mac ≈630 MB、Windows 較大),逐檔驗證
SHA-256,原子安裝到 `<APP>/download/`(暫存,保留供重跑時免重下)、
`<APP>/bin/`(engine host,解壓後 pin 一份到 `bin/engine-host[.exe]`)、
`<APP>/model/`(模型檔案,扁平存放)。**模型檔案較大,依網路狀況可能需要
幾分鐘到十幾分鐘**——這是唯一一個會等比較久的步驟,其餘步驟都是秒級的。

### 預期輸出(mac,逐字節錄自實際 dry run,見 `task-4-report.md`)

```
Task setup:mac deno run --allow-net --allow-read=.,../_machine --allow-write=.,../_machine --allow-run=tar --allow-env=TMUH_APPS_DIR reference/setup.ts
安裝 tomato-ears 相依到 <APP>(平台:macos-arm64)…
engine host:下載中…
engine host:下載完成
engine host:解壓中…
engine host:已 pin 到 <APP>/bin/engine-host
model/weights.safetensors:下載中…
model/weights.safetensors:下載完成
（… 其餘 9 個模型檔案逐一下載完成 …）
安裝完成。可執行 `deno task verify:mac`(Windows:`deno task verify:win`)驗收，或直接 `deno task start:mac`/`start:win` 啟動。
```

如果你之前已經跑過(例如重試),已存在且雜湊相符的檔案會印
`已存在且雜湊相符,略過下載`,不會重新下載——這是正常行為,不是卡住。

### 預期輸出(Windows,逐字節錄自 Task 6 演練;完整逐字稿見

`.superpowers/sdd/task-6-rehearsal-log.md`「階段二 Step 3」)

```
Task setup:win deno run --allow-net --allow-read=.,../_machine --allow-write=.,../_machine --allow-run=tar --allow-env=TMUH_APPS_DIR reference/setup.ts
安裝 tomato-ears 相依到 <APP>(平台:windows-x64)…
engine host:下載中…
engine host:下載完成
engine host:解壓中…
engine host:已 pin 到 <APP>/bin/engine-host.exe
model/audio_processor_config.json:下載中…
model/audio_processor_config.json:下載完成
（… 其餘 12 個 onnx/config 模型檔逐一下載完成 …）
安裝完成。可執行 `deno task verify:mac`(Windows:`deno task verify:win`)驗收,或直接 `deno task start:mac`/`start:win` 啟動。
```

這台演練機器的 manifest 在 Windows 平台 pin 了 13 個模型檔(含
`encoder.onnx.data` ≈690MB、`decoder.onnx.data` ≈59MB、
`joint.onnx.data` ≈37MB)+ engine host zip ≈55MB,總計約 850MB,全新
下載耗時約 7 分鐘,逐檔 SHA-256 驗證全通過。**Windows 內建的 tar
(bsdtar)可以正常解開 engine host 的 windows zip**,`--allow-run=tar`
縮圈下解壓 + pin 到 `bin/engine-host.exe` 均正常,無需額外安裝解壓工具。

**Task 6 Windows 演練確認(正常路徑,非錯誤)**:在全新模擬環境(演練前
確認 `where deno` 找不到 deno,走 winget 全新安裝)下實際跑過一次
`deno task setup:win`,exit code 0,逐字輸出與上方一致,沒有踩到下表任何
一列常見錯誤。

### 常見錯誤

| 症狀                                     | 原因                                                                 | 解法                                                                                                                             |
| ---------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `SHA-256 不符(...)` 然後整個指令失敗退出 | 下載過程中內容被竄改,或上游檔案已更新但 `manifest.json` 的雜湊沒同步 | 重跑一次 `deno task setup:mac`(下載器會自動清掉壞檔重下);如果重跑後仍然不符,停下來回報,不要手動改 `manifest.json` 的雜湊繞過檢查 |
| `下載失敗(...)：HTTP 4xx/5xx`            | 網路問題,或 GitHub/HF 暫時性錯誤                                     | 重跑指令;`ensureDependencies` 是冪等的,已下載成功的檔案不會重來一次                                                              |
| `NotCapable` 相關錯誤                    | Step 2 的 cwd 沒對(不在 `<APP>` 底下執行)                            | 確認 `pwd` 就是 `<APP>`,重新 `cd` 過去再跑                                                                                       |
| `tar 解壓失敗`                           | 系統 `tar`/bsdtar 版本異常或磁碟空間不足                             | 檢查磁碟剩餘空間;mac/Windows 10+ 都內建相容的 `tar`,通常不需要額外安裝                                                           |

**Task 5 mac 演練確認（正常路徑,非錯誤)**：在全新模擬環境(deno 已裝但
`PATH` 缺失,見 env-base RECIPE.md「已知結果」附註的第三種情境)下實際
跑過一次 `deno task setup:mac`,輸出逐字與上方「預期輸出」一致,一個字
都沒有偏差;`model/weights.safetensors` 實測 658,663,198 bytes ≈
628MB,符合文件說的「≈630 MB」。沒有踩到上表任何一列常見錯誤。完整逐字
稿見 `.superpowers/sdd/task-5-rehearsal-log.md`「Step 3」。

## Step 4:驗收(`deno task verify:mac` / `verify:win`)

```bash
deno task verify:mac      # Windows 用 verify:win
```

這是**唯一**的完成判準(SPEC.md 第 3 節、店規第 6 條):全綠才算建構完成,
不能自行宣告成功。跑起來大約幾秒鐘(真模型會做一次真推論,比純邏輯測試
慢一點,但不到十秒)。

### 預期輸出(mac,逐字節錄自實際 dry run;完整證據見 `task-4-report.md`)

```
Task verify:mac deno test --allow-net --allow-sys=networkInterfaces --allow-read=.,../_machine --allow-write=../_machine --allow-run=bin/engine-host --allow-env=TMUH_APPS_DIR verify/
running 13 tests from ./verify/asr_metric_test.ts
（13 個全部 ok）
running 1 test from ./verify/binding_test.ts
binding：服務只綁 127.0.0.1——透過本機 LAN 介面位址連線應被拒絕 ... ok
running 1 test from ./verify/integrity_test.ts
integrity：manifest 相依檔案在本機安裝目錄內存在且 SHA-256 相符 ... ok (12 steps)
running 2 tests from ./verify/permissions_test.ts
（2 個全部 ok）
running 1 test from ./verify/protocol_test.ts
protocol：真 host 真模型轉錄 hello-streaming.wav，與參考文字的正規化編輯距離 ≤ 0.25 ... ok
running 1 test from ./verify/service_test.ts
service：WS 全流程(真 host 真模型) ready → start → binary chunks → partial → stop → final ... ok

ok | 19 passed (12 steps) | 0 failed
```

（`protocol_test.ts` 實際跑出來的轉錄文字與參考文字 `"Hello, this is a
streaming speech recognition test"` 逐字相同,正規化編輯距離為 0——比
門檻 0.25 更好,這是模型/fixture 品質良好的正常結果,不代表門檻設太鬆。）

### 常見錯誤

| 症狀                                                                         | 原因                                                                 | 解法                                                                                    |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `MLX error: Failed to load the default metallib. library not found`(mac)     | 見本文件最開頭的說明——你手上的 engine host 是有上游打包缺陷的 v0.1.0 | 重跑 `deno task setup:mac` 換成 pin 住的 v0.1.1;若重跑後仍出現,才回報這個確切錯誤訊息   |
| `integrity` 測試某個 model/ 檔案 `SHA-256 不符`                              | Step 3 沒有真的跑完(可能中途被中斷),或安裝目錄被手動改動過           | 重跑 `deno task setup:mac`,它是冪等的,壞檔會被自動清掉重下                              |
| `binding` 測試被 `ignore`(不是失敗,是跳過)                                   | 這台機器沒有非 loopback 的 IPv4 網卡(例如某些沙箱/容器環境)          | 正常行為,不是錯誤——這個測試的斷言前提是「有 LAN 介面可以嘗試連線」,沒有就沒什麼好驗的   |
| `permissions` 測試失敗,訊息提到「旗標不一致」                                | `<APP>/deno.json` 或 `<APP>/manifest.json` 在複製過程中被截斷/改動   | 回到 Step 1 重新複製這兩個檔案,不要手動編輯它們                                         |
| `engine host(...)第一行不是合法的 ready 事件` 但訊息不是上面的 metallib 錯誤 | 引擎子行程啟動失敗的其他原因(模型路徑錯、binary 損毀等)              | 先確認 `<APP>/model/` 底下確實有 Step 3 下載完成的檔案;仍無法排除就完整回報這則錯誤訊息 |

**Task 5 mac 演練確認（正常路徑,非錯誤)**：`ok | 19 passed (12 steps) |
0 failed`,逐字重現上方「預期輸出」;而且**沒有**遇到本文件開頭特別
強調的 `MLX error: Failed to load the default metallib` 問題——manifest
pin 的 `asr-host-v0.1.1` 確實已修好那個上游打包缺陷。沒有踩到上表任何
一列常見錯誤。完整逐字稿見 `.superpowers/sdd/task-5-rehearsal-log.md`
「Step 4」。

### 預期輸出(Windows,逐字節錄自 Task 6 演練;完整逐字稿見

`.superpowers/sdd/task-6-rehearsal-log.md`「階段二 Step 4」)

```
Task verify:win deno test --allow-net --allow-sys=networkInterfaces --allow-read=.,../_machine --allow-write=../_machine --allow-run=bin/engine-host.exe --allow-env=TMUH_APPS_DIR verify/
running 13 tests from ./verify/asr_metric_test.ts（13 個全部 ok）
running 1 test from ./verify/binding_test.ts
binding:服務只綁 127.0.0.1——透過本機 LAN 介面位址連線應被拒絕 ... ok (29s)
running 1 test from ./verify/integrity_test.ts
integrity:manifest 相依檔案在本機安裝目錄內存在且 SHA-256 相符 ... ok (11s)
running 2 tests from ./verify/permissions_test.ts（2 個全部 ok）
running 1 test from ./verify/protocol_test.ts
protocol:真 host 真模型轉錄 hello-streaming.wav,與參考文字的正規化編輯距離 ≤ 0.25 ... ok (38s)
running 1 test from ./verify/service_test.ts
service:WS 全流程(真 host 真模型) ready → start → binary chunks → partial → stop → final ... ok (37s)

ok | 19 passed (15 steps) | 0 failed (1m57s)
```

跟 mac 的 `19 passed (12 steps)` 相比多 3 個 steps——這台演練機器的
manifest 在 Windows 平台 pin 了 13 個模型檔(mac 是 10 個),`integrity`
測試逐檔驗 SHA-256,檔案數不同純粹是這次演練機器 manifest 的平台差異,
不是缺陷。

> **平台差異警告:耗時與「DML 探測失敗→退回 CPU」是預期現象,不是卡住**
>
> - 沒有相容 GPU/驅動的 Windows 機器上,**verify 的每個真引擎 spawn 都
>   會先探測 DML、失敗後再探測 CPU**(binding/protocol/service 三個測試
>   各自 spawn 一次引擎,各自重探一次——探測結果**不會**回填進
>   machine-profile,回填是 `start`/`main.ts` 的職責,見 Step 5)。stderr
>   會依序印:
>   ```
>   [engine host stderr] ...onnxruntime... Non-zero status code returned while running DmlFusedNode_0_74 node ... 80070057 [probe] dml 探測失敗:...
>   [engine host stderr] [probe] cpu 探測成功:3924~3947 ms
>   [engine host stderr] [probe] 選定後端:cpu(閾值 AutoGpuThreshold=0.85)
>   ```
>   這是**已知/預期現象**(onnxruntime DmlFusedNode 80070057),DML 失敗
>   後自動退回 CPU 探測成功即為正常路徑,**不代表安裝失敗**。
> - 本文件開頭 Step 4 寫的「不到十秒」是 **mac(mlx 後端)** 的數字。
>   **Windows CPU 後端 + 每次 spawn 都重新探測**,三個真引擎測試各花
>   29–38 秒,整體約 **2 分鐘**(Task 6 實測 `1m57s`)才是 Windows 的
>   正常耗時,不要以為卡住了。

**Task 6 Windows 演練確認(正常路徑,非錯誤)**：`ok | 19 passed (15
steps) | 0 failed (1m57s)`,逐字重現上方「預期輸出」,DML 探測失敗、
CPU 探測成功(~3.9s)、選定 cpu 均符合這台機器的已知預期,沒有踩到上表
任何一列常見錯誤。完整逐字稿見
`.superpowers/sdd/task-6-rehearsal-log.md`「階段二 Step 4」。

## Step 5:啟動(`deno task start:mac` / `start:win`)

```bash
deno task start:mac      # Windows 用 start:win
```

### 預期輸出(mac,逐字節錄自實際 dry run)

```
Task start:mac deno run --allow-net=127.0.0.1:43117 --allow-read=.,../_machine --allow-write=../_machine --allow-run=bin/engine-host,open --allow-env=TMUH_APPS_DIR,TMUH_NO_BROWSER reference/main.ts
啟動引擎:<APP>/bin/engine-host --model <APP>/model --language auto
引擎就緒,backend = mlx
服務已啟動:http://127.0.0.1:43117/
```

接著瀏覽器應該會自動開啟 `http://127.0.0.1:43117/`,看到番茄耳的錄音頁。
如果瀏覽器沒有自動開啟(無 GUI 環境、SSH 連線等),終端機會多印一行
`無法自動開啟瀏覽器(...)`,手動打開印出來的網址即可——這是設計上允許的
降級,不是失敗。

用另一個終端機視窗驗證服務真的起來了:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:43117/
# 預期輸出:200
```

按 `Ctrl+C` 可以停止服務。

**Task 5 mac 演練確認（正常路徑,非錯誤)**：`引擎就緒,backend = mlx` +
`服務已啟動:http://127.0.0.1:43117/` 逐字重現上方「預期輸出」;`curl`
拿到 **200**;`lsof -nP -iTCP:43117 -sTCP:LISTEN` 只有一行且 ADDRESS 是
`127.0.0.1:43117`(不是 `*:43117`),符合 SECURITY.md 2.2 節的綁定範圍
審查標準;`kill -INT` 停止服務後 exit code 130(收到 SIGINT 的慣例值),
port 確認釋放。演練是在無 GUI 環境跑,額外用 `TMUH_NO_BROWSER=1`(文件
本身支援的降級開關,見 `main.ts` 註解)跳過自動開瀏覽器,只用 curl 驗證
——這不是文件要求的預設路徑,只是無 GUI agent 環境的必要調整,一般使用者
不需要設這個變數。完整逐字稿見 `.superpowers/sdd/task-5-rehearsal-log.md`
「Step 5」。

### Windows 對照(cmd,Task 6 演練逐字節錄;完整逐字稿見

`.superpowers/sdd/task-6-rehearsal-log.md`「階段二 Step 5/5'」)

```cmd
set TMUH_NO_BROWSER=1& cd /d <APP> & deno task start:win
```

(`TMUH_NO_BROWSER=1` 用法同 mac 的降級開關,遠端/無頭情境跳過自動開
瀏覽器。**注意寫法**:`set TMUH_NO_BROWSER=1&` 刻意**不留空格**再接
`&`——見 env-base RECIPE.md Step 3 的 cmd 陷阱警告,`set X=value & ...`
單行語法會把 `&` 前的空格也算進值裡,這裡任何非空值都能用,但養成不留
空格的習慣可以避免中招;同一份警告也提到 `%VAR%` 展開發生在整行解析
期、早於 `set` 執行,同樣適用於任何在 Windows 上用 cmd 單行組合
`set`+變數展開的場景。)

**首跑**(machine-profile 尚未回填 backend,引擎自行探測):

```
啟動引擎:<APP>/bin/engine-host.exe --model <APP>/model --language auto
引擎就緒,backend = cpu
已把 backend=cpu 回填進 machine-profile(下次啟動跳過探測)
服務已啟動:http://127.0.0.1:43117/
已依 TMUH_NO_BROWSER 略過自動開啟瀏覽器,請手動開啟:http://127.0.0.1:43117/
[engine host stderr] ...onnxruntime... Non-zero status code returned while running DmlFusedNode_0_74 node ... 80070057 [probe] dml 探測失敗:...
[engine host stderr] [probe] cpu 探測成功:3924 ms
[engine host stderr] [probe] 選定後端:cpu(閾值 AutoGpuThreshold=0.85)
```

送出指令到「服務已啟動」約 **24 秒**(含 deno 啟動 + DML 探測失敗 +
CPU 基準測量 ~3.9s + 模型載入)——這是**首跑才有**的一次性成本,DML
失敗、退回 CPU 是這台機器的已知/預期現象,不是錯誤(同 Step 4 的平台
差異警告)。回填訊息「已把 backend=cpu 回填進 machine-profile」如實
出現,對照 machine-profile.json 可看到 `inference.tomato-ears.backend =
"cpu"` 已寫入,env-base 探測欄位原樣保留。

驗證(Windows 上用 `curl.exe`,注意不是 PowerShell 的 `curl` alias):

```cmd
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:43117/
```

```
200
```

綁定範圍核對(等效 mac 的 `lsof -nP -iTCP:43117 -sTCP:LISTEN`):

```cmd
netstat -ano | findstr :43117
```

預期唯一一行 `TCP 127.0.0.1:43117 0.0.0.0:0 LISTENING <pid>`——只綁
loopback,不是 `0.0.0.0`,對齊 SECURITY.md 2.2 節的綁定範圍審查標準。

**停止服務(Windows)**:上方「按 `Ctrl+C` 可以停止服務」假設的是本機
互動式終端;ssh 遠端驅動/無互動 session 沒有 Ctrl+C 可送,改用
`taskkill`(`/T` 一併終止 engine-host.exe 子行程):

```cmd
taskkill /F /T /PID <pid>
```

`<pid>` 取自上面 `netstat` 那一行最後一欄。跑完再 `netstat` 核對
LISTENING 消失、port 已釋放。

**二跑**(machine-profile 已有 `backend=cpu` 回填,啟動引擎直接跳過
探測):

```
啟動引擎:<APP>/bin/engine-host.exe --model <APP>/model --language auto --backend cpu
引擎就緒,backend = cpu
服務已啟動:http://127.0.0.1:43117/
已依 TMUH_NO_BROWSER 略過自動開啟瀏覽器,請手動開啟:http://127.0.0.1:43117/
[engine host stderr] [probe] preference=cpu,略過探測。
```

三個跡象確認二跑跳過探測:啟動引擎那行帶 `--backend cpu`(首跑沒有,
`buildEngineArgs` 讀到回填值)、stderr 只剩一行 `preference=cpu,略過
探測`(首跑有三行 DML/CPU 探測輸出)、耗時**約 6 秒**(首跑 24 秒省下
的 ~18 秒即探測成本)。二跑不會再印「已把 backend=cpu 回填…」——已有
值不重寫。`curl.exe` 仍回 200,`netstat`/`taskkill` 停止方式同上。

**防火牆彈窗:無**——服務只綁 loopback(`127.0.0.1`),Windows Defender
防火牆對純 loopback listener 不彈「允許存取」對話框,curl 立即拿到
200,沒有任何被攔截的跡象(若未來服務改綁 `0.0.0.0` 才需要預期彈窗)。

**Task 6 Windows 演練確認(正常路徑,非錯誤)**：首跑「已把
backend=cpu 回填進 machine-profile」如實出現、二跑帶 `--backend cpu`
且略過探測,兩跑 `curl.exe` 均回 200,`netstat` 兩次皆只有
`127.0.0.1:43117` 一行 LISTENING,`taskkill /F /T` 兩次皆成功釋放
port。沒有踩到下表任何一列常見錯誤(下表原本只列 mac/通用症狀,
Windows 特有的「找不到怎麼停止服務」已在上方單獨說明)。完整逐字稿見
`.superpowers/sdd/task-6-rehearsal-log.md`「階段二 Step 5/5'」。

**錄音前**：按下「開始錄音」後,瀏覽器會先跳出麥克風授權提示才會真的
開始收音。如果按下去畫面看起來沒有反應(按鈕變成 disabled、但沒有任何
文字變化),**請找瀏覽器的麥克風授權提示**——它可能不是一個明顯的彈窗,
有些瀏覽器會把它做成網址列旁邊一個容易被忽略的小圖示/下拉提示,或是
被瀏覽器直接擋下(需要到網站設定裡手動允許)。**這個提示懸置(使用者
還沒回應)期間,「開始錄音」/「停止錄音」兩顆按鈕都會是 disabled**,
畫面上的連線狀態文字會顯示「等待麥克風授權…」提示這一點。

**人力步驟**：從這裡開始,「按下瀏覽器的麥克風授權提示」是整個
tomato-ears 安裝流程裡唯一必須由使用者本人完成的動作——agent 沒有辦法
代替使用者點擊瀏覽器原生的權限對話框。Step 1–4(複製檔案、下載相依、
跑 `verify:mac`/`verify:win`)與 Step 5 啟動服務本身,agent 都能獨力
完成、不需要人力介入;只有「實際錄音收到真的逐字稿」這一段,因為需要
真人授權麥克風 + 真人講話產生音訊,超出了 agent 的能力範圍,留給人工
驗證(Task 5 演練也只驗到 curl 200 為止,真實錄音未驗證)。

### 常見錯誤

| 症狀                                                         | 原因                                                                        | 解法                                                                                |
| ------------------------------------------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `尚未完成安裝。請先執行:deno task setup:mac`                 | Step 3 還沒跑,或跑到一半失敗                                                | 回到 Step 3                                                                         |
| `curl` 得到 404 而不是 200                                   | 同 Step 1 的 `ui/` 位置錯誤(見上面該表格)                                   | 檢查 `<APP>/ui/index.html` 是否存在                                                 |
| 終端機卡住不動,沒有任何輸出                                  | 引擎子行程正在載入模型(第一次啟動需要幾秒到十幾秒),或前面步驟未真正完成     | 稍等;如果超過 30 秒仍無輸出,`Ctrl+C` 後重跑,並確認 Step 4 的 verify 已經全綠過      |
| 按下「開始錄音」後畫面靜止,兩顆按鈕都 disabled、沒有文字變化 | 瀏覽器的麥克風授權提示懸置中(使用者還沒回應,或提示被瀏覽器擋下沒有顯眼彈出) | 找瀏覽器的麥克風授權提示(見上方「錄音前」說明)並允許;這是**唯一**需要人工操作的步驟 |

## 附錄 A:兩階段權限模型(摘要)

`setup` 需要對外網路(下載相依),`start` 執行期完全沒有對外網路權限
(只綁 `127.0.0.1:43117`)。完整的逐條權限對照表、cwd 相對路徑的設計
理由、`setup` 階段 `--allow-net` 全開的 trade-off 說明,見 **SECURITY.md**
——那份文件也是你在宣告「建構完成」之前必須執行過一次的審查步驟清單
(店規第 4 節第 5 條),不是可以跳過的附加閱讀。

## 附錄 B:dev-time 測試與 verify/ 的分工

這個配方裡有兩層測試,職責不同,不要混淆:

- **`reference/*_test.ts`(dev-time)**:配方**維護者**在改動
  `reference/` 程式碼時用的快速回歸測試,對假的/fake-engine 的引擎跑
  (不需要真模型、不需要真的下載),追求秒級回饋。**你(組裝 agent)
  不需要執行這些**,它們甚至不會被複製進乾淨的安裝目錄慣例之外的地方
  (複製了也無妨,只是不會被 `deno task verify:*`/`start:*` 引用到)。
- **`verify/*_test.ts`(使用者驗收)**:對**真正安裝好的**引擎與模型跑,
  驗證「這次組裝出來的東西,對這個使用者的機器來說,真的能動」。
  **這是你唯一需要跑、也唯一需要全綠的測試套件**(Step 4)。

兩者用不同的引擎(fake vs. 真 host)是刻意設計:dev-time 測試的目標是
「這段程式碼邏輯對不對」,不需要每次改一行程式碼就等模型載入;verify 的
目標是「這次真實安裝對不對」,必須是真引擎真模型才有意義,慢一點也
值得。

# tomato-ears 配方包(Deno Recipe)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 產出 tmuh.ai mini-app store 的第一個配方 `tomato-ears`:使用者的 agent 依配方在 mac/Windows 組裝出「瀏覽器錄音 → 本機 Nemotron ASR 即時轉錄」應用;並附 `env-base`(App 0 環境基座)配方。以兩平台的「agent 全程組裝演練」驗收。

**Architecture:** 配方源碼樹在 `recipes/`(env-base、tomato-ears)。tomato-ears 執行形態:Deno 服務(單一 process)serve 靜態 UI + WebSocket;瀏覽器 getUserMedia → AudioWorklet 降採樣為 16 kHz PCM16(1600-sample chunks)→ WS binary frames → Deno 轉 base64 餵 engine host stdin(asr-host-v1 協定)→ partial/final 經 WS text frames 回 UI。引擎與模型是 pinned prebuilt 相依(Plan 1/2 的兩個 GitHub Release + 兩個模型 artifact),`deno task setup` 下載並驗 hash,`deno task start` 以最小權限運行。

**Tech Stack:** Deno ≥ 2.x(std 庫,零 npm 相依)、AudioWorklet、系統 `tar` 解壓(mac tar.gz / Windows bsdtar 開 zip)。

**相關 spec:** `docs/superpowers/specs/2026-07-18-tomato-ears-design.md`、`2026-07-18-mini-app-store-design.md`(店規九條)、`docs/protocol/asr-host-v1.md`(凍結)

## Global Constraints

- 分支 `feat/tomato-ears-recipe`(base main 9ed9236);mac 端在 worktree `/Users/wake/Desktop/catcher-tippi/.worktrees/tomato-ears-recipe` 工作
- **兩階段權限**(資安核心設計):
  - `deno task setup`(下載階段):`--allow-net --allow-read=<appdir> --allow-write=<appdir> --allow-run=tar`(HF/GitHub CDN redirect 網域不可枚舉,故 net 全開,完整性由 SHA-256 pin 保證——SECURITY.md 必須如實說明此 trade-off)
  - `deno task start`(運行階段):`--allow-net=127.0.0.1:43117 --allow-read=<appdir> --allow-run=<engine-host路徑>` ——**運行時零對外網路**
- Port:43117(HTTP + 同 port `/ws`);服務只綁 `127.0.0.1`(店規 5)
- 引擎相依(pin 死,寫進 manifest):
  - mac:`https://github.com/wcAmon/catcher-tippi/releases/download/asr-host-v0.1.0/catcher-asr-host-v0.1.0-macos-arm64.tar.gz`,sha256 `4a536c0c95e70d5d9b8cfbd764ddf4d4208395b1407dfce93532081f67ed251c`
  - win:`https://github.com/wcAmon/catcher-tippi/releases/download/nemotron-asr-host-v0.1.0/nemotron-asr-host-v0.1.0-windows-x64.zip`,sha256 `10750b3e28ea2686c8442d6dc6de03fd5490f0dc0033977ccfbeb5be1fcb4667`
- 模型相依(pin 死):
  - mac:HF `wcamon/catcher-asr-mlx-int8` 10 檔,hash 表逐字取自 `apps/tippi/Sources/TippiCore/ModelManifest.swift:19-28`
  - win:HF `onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4` 13 檔,hash 表逐字取自 `scripts/fetch-nemotron-onnx-model.ps1` 靜態表(revision-pinned URL 同該腳本)
- 協定語義以 `docs/protocol/asr-host-v1.md` 為準:partial 是快照替換、error.message 不可解析分支、`start.lang` 保留欄位(語言由 host `--language` 旗標決定)
- Windows host 探測成本:首次 ready 後把 `backend` 值存進 machine-profile,之後以 `--backend <值>` 重生 host(BackendProber 註解建議的做法)
- Reference 程式碼註解密度遵守店規第 4 條:每個 exported symbol 有目的說明、每段非顯然邏輯有 why 註解(註解是給重建 agent 的語義錨點)
- 安裝目錄慣例:`~/tmuh-apps/tomato-ears/`(bin/、model/、ui/…);machine-profile 在 `~/tmuh-apps/_machine/machine-profile.json`
- deno 指令一律 `deno fmt --check` + `deno lint` 乾淨;測試 `deno test`(權限旗標各測試檔自帶)
- Windows 驗證走 ssh(模式照 `scripts/bootstrap-windows-host.md`);Windows 端 deno 以 `winget install DenoLand.Deno`(2.9.x)
- 每個 commit 附 Claude trailer(同前)

---

### Task 1: mac Deno bootstrap + 配方骨架 + manifest + env-base 配方

**Files:**
- Create: `recipes/env-base/{RECIPE.md, probe/machine-profile.ts}`
- Create: `recipes/tomato-ears/manifest.json`
- Create: `recipes/tomato-ears/deno.json`(tasks:setup/start/verify;fmt/lint 設定)
- Modify: `scripts/bootstrap-windows-host.md`(附註 mac 端 deno 安裝紀錄)

**Steps:**

- [ ] **Step 1:** mac 安裝 deno(官方腳本 user-local `curl -fsSL https://deno.land/install.sh | sh`,或 brew——擇一,記錄版本與路徑)
- [ ] **Step 2:** `recipes/env-base/RECIPE.md`:App 0 配方(店規 §5)——安裝 Deno 一條指令、建 `~/tmuh-apps/` 與 `_machine/`、跑 `probe/machine-profile.ts` 產 machine-profile.json(OS、arch、RAM、CPU 執行緒數、deno 版本;推論後端欄位留空由各 app 首跑回填)、指示 agent 把環境事實寫入其持久記憶(CLAUDE.md/memory)並「之後不重探測不重裝」
- [ ] **Step 3:** `probe/machine-profile.ts`:Deno 腳本,產出 JSON;冪等(已存在則 merge 不覆蓋 app 回填欄位);單元測試附在同目錄 `machine-profile_test.ts`
- [ ] **Step 4:** `recipes/tomato-ears/manifest.json`,schema 遵守店規 §3:

```json
{
  "name": "tomato-ears",
  "version": "0.1.0",
  "stack": "deno",
  "ports": { "http": 43117, "protocol": "asr-host-v1" },
  "permissions": {
    "setup": ["--allow-net", "--allow-read=~/tmuh-apps/tomato-ears", "--allow-write=~/tmuh-apps/tomato-ears", "--allow-run=tar"],
    "start": ["--allow-net=127.0.0.1:43117", "--allow-read=~/tmuh-apps/tomato-ears", "--allow-run=~/tmuh-apps/tomato-ears/bin"]
  },
  "dependencies": { "engine": { "macos-arm64": {...}, "windows-x64": {...} }, "model": { "macos-arm64": [...10 檔...], "windows-x64": [...13 檔...] } },
  "verify": "deno task verify"
}
```

(hash 表逐字轉錄自 Global Constraints 指定的兩處來源;轉錄後寫一個一次性核對腳本比對 Swift/ps1 原檔,防抄寫錯誤,核對後刪除)
- [ ] **Step 5:** `deno fmt --check` + `deno lint` 過;commit `feat: tomato-ears recipe scaffold with env-base and pinned manifest`

---

### Task 2: reference/ 後端模組(downloader / engine / server / main)

**Files:**
- Create: `recipes/tomato-ears/reference/{main.ts, downloader.ts, engine.ts, server.ts}` + 各自 `*_test.ts`

**Interfaces(後續 task 依賴,精確簽名):**
- `downloader.ts`:`ensureDependencies(manifest: Manifest, appDir: string, platform: "macos-arm64"|"windows-x64", onProgress?: (msg: string) => void): Promise<void>` ——逐檔:已存在且 hash 符 → skip;下載到 `<target>.part` → 驗 hash → rename(原子);engine 壓縮包解壓用系統 `tar -xf`(bsdtar 兩平台通吃);任何 hash 不符 → 刪檔 throw
- `engine.ts`:`class EngineClient { static spawn(binPath: string, args: string[]): Promise<EngineClient>`(等 ready,回傳含 `backend: string`)`; start(lang?: string): void; pushPcm(chunk: Uint8Array): void; stop(): Promise<string>`(resolve final text)`; onPartial: (text: string) => void; kill(): void }` ——stdin 寫 JSON lines(pcm base64)、stdout 逐行解析事件;host crash → onError + 可重生
- `server.ts`:`startServer(appDir: string, engine: EngineClient, port: number): Deno.HttpServer` ——靜態檔(ui/)+ `/ws`:binary frame = PCM16 chunk → engine.pushPcm;text frame `{"type":"start"}`/`{"type":"stop"}`;回推 `{"type":"partial"|"final"|"error"|"ready", ...}`;**僅綁 127.0.0.1**
- `main.ts`:讀 machine-profile 決定 platform 與 host 旗標(win:有存 backend 則 `--backend <值>`)、setup 未完成則提示先跑 setup、spawn engine、起 server、`open`/`start` 開瀏覽器(允許失敗,印 URL)

**Steps:**

- [ ] **Step 1(測試先行):** `engine_test.ts` 用 **mac 本地 build 的 host**(`cargo build --release -p catcher-asr-host` 產物,`--fake-engine`)黑箱測 EngineClient:spawn→ready(backend "fake")、start→pushPcm(1600 samples)→partial "字0"、stop→final、二會話、host 殺掉後 error surface。`downloader_test.ts` 用本地 HTTP server(Deno 內建)供假檔案:hash 對/錯/斷點殘檔三情境。`server_test.ts` 用假 EngineClient(依同介面手寫 stub)走 WS 全流程 + 驗非 127.0.0.1 綁定失敗(嘗試連 `0.0.0.0`/LAN IP 應拒)
- [ ] **Step 2:** 實作四模組使測試轉綠;註解密度照店規第 4 條(這是 reference code,註解是產品的一部分)
- [ ] **Step 3:** `deno fmt --check`、`deno lint`、`deno test` 全綠;commit

---

### Task 3: reference/ui(錄音頁 + AudioWorklet 降採樣)

**Files:**
- Create: `recipes/tomato-ears/reference/ui/{index.html, app.js, downsampler-worklet.js, style.css}`

**行為規格:**
- getUserMedia(mono)→ AudioContext(native rate)→ AudioWorklet:線性內插重採樣到 16 kHz、累積滿 **1600 samples** 發一個 Int16Array chunk(port.postMessage, transferable)→ 主執行緒經 WS binary 送出
- UI:開始/停止錄音鈕、即時 partial 顯示(快照替換語義——直接整段覆寫,不 append)、final 後訊息列表累積、「複製全部」與「匯出 .txt」、backend 徽章(顯示 ready 的 backend 值,僅展示不分支)
- 無框架、無外部 CDN(店規:零外部資源;單檔可讀,詳細註解)
- 中文介面文案;downsampler 的 why 註解要講清楚「為何 1600 samples/100 ms」(協定 chunk 建議值)與線性內插的取捨

**Steps:**

- [ ] **Step 1:** 實作四檔
- [ ] **Step 2:** 冒煙:`deno task start` 前置條件未備(引擎未裝)時的降級路徑——server_test.ts 已測 WS;UI 端以 headless 瀏覽器手動/腳本驗證留給 Task 5 演練(mac 有真環境);本 task 驗 `deno fmt --check`(html/js 也格式化)+ 靜態 lint + server 靜態檔 serve 測試(fetch index.html 200)
- [ ] **Step 3:** commit

---

### Task 4: SPEC / PLAN / SECURITY + verify/ 驗收套件

**Files:**
- Create: `recipes/tomato-ears/{SPEC.md, PLAN.md, SECURITY.md}`
- Create: `recipes/tomato-ears/verify/{integrity_test.ts, protocol_test.ts, service_test.ts, binding_test.ts, permissions_test.ts}` + fixture `verify/fixtures/hello-streaming.wav`(複製自 `tests/fixtures/`)

**內容規格:**
- `SPEC.md`:要做出什麼(功能清單同 Task 3 行為規格)、驗收標準(verify 全綠 = 完成)、非目標(diarization/TTS/語音注入)
- `PLAN.md`:給重建 agent 的分階段指令——**第一步固定「讀 machine-profile,勿重探測勿重裝」**;之後:複製 reference → `deno task setup`(下載+驗 hash)→ `deno task verify` → `deno task start`。每步含預期輸出與常見錯誤對照表(演練後回填)
- `SECURITY.md`:兩階段權限模型逐條解釋(含 setup 階段 net 全開的 trade-off 與 hash-pin 補償)、審查步驟:agent 必須核對「run script 旗標 == manifest.permissions」、確認服務只綁 127.0.0.1、確認零外部 CDN
- `verify/`:
  - `integrity_test.ts`:manifest 每個 dependency 檔存在且 sha256 符
  - `protocol_test.ts`:對已安裝的真 host 餵 fixture wav(1600-sample chunks),final 與 "Hello, this is a streaming speech recognition test" normalized edit distance ≤ 0.25(移植 Task 1 的 normalize+levenshtein 到 TS——第三個實作,語義一致)
  - `service_test.ts`:起服務走 WS start→binary chunks→partial→stop→final 全流程(真 host 真模型)
  - `binding_test.ts`:非 loopback 介面連不上
  - `permissions_test.ts`:解析 deno.json tasks 的旗標,比對 manifest.permissions 逐字相等
- verify 是使用者驗收(真引擎真模型);開發期快速測試(fake-engine)留在 reference/*_test.ts——兩者分工寫進 PLAN.md

**Steps:** 實作 → mac 本機以「已安裝」狀態(本地 host build + 本機既有模型目錄軟連結模擬安裝)預跑 verify 五件套 → commit

---

### Task 5: mac 演練(agent 全程組裝)+ PLAN.md 回填

**程序:**
1. 準備乾淨演練環境:`TMUH_HOME=$(mktemp -d)`(模擬使用者家目錄的 `~/tmuh-apps`;配方支援 `TMUH_APPS_DIR` 環境變數覆寫預設路徑——若 reference 尚未支援,本 task 先補)
2. 派一個「使用者 agent」subagent:**只給它 `recipes/env-base/RECIPE.md` 與 `recipes/tomato-ears/` 的 SPEC/PLAN/SECURITY/manifest+reference+verify,不給任何本 repo 其他知識**;要求照 PLAN.md 全程執行:env-base → setup(真下載 mac host tar.gz + HF 模型 ≈630MB)→ verify 全綠 → start 起服務 → `curl 127.0.0.1:43117` 200
3. 記錄 agent 卡住/困惑/自行發明的每一點 → 回填 PLAN.md(補預期輸出、錯誤對照)與 reference 註解
4. 迭代直到一次乾跑通過;commit `docs: backfill tomato-ears PLAN from mac rehearsal`

---

### Task 6: Windows 演練(ssh)+ 回填

**程序:**
1. Windows 機 `winget install DenoLand.Deno`(記錄進 bootstrap doc);git pull 後把 `recipes/` 複製到演練目錄(不給 repo 其他內容,同 Task 5 隔離原則)
2. 同 Task 5 流程:env-base → setup(win zip + onnx 模型;模型已在 `C:\Users\i5491\catcher-tippi-models\`,演練仍走配方下載路徑驗證全新安裝,目錄用演練專屬路徑)→ verify → start → `curl` 200;**確認首跑後 machine-profile 記到 backend=cpu,二跑 host 以 `--backend cpu` 啟動(跳過探測)**
3. 卡點回填 PLAN.md(Windows 段:cmd/PowerShell 差異、tar/bsdtar、防火牆彈窗預期);commit

---

### Task 7: 配方 bundle 打包 + 終審材料

**Files:**
- Create: `scripts/build-recipe-bundle.sh`(tar.gz `recipes/tomato-ears` + `recipes/env-base` → `dist/tomato-ears-recipe-v0.1.0.tar.gz` + `.sha256`,bare-filename LF 格式)

**Steps:** 腳本 → 執行驗證(解包 diff 原樹一致)→ commit;(上架 tmuh.ai 屬 Plan 4,發布動作不在本計畫)

---

## 後續計畫(本檔不含)

- **Plan 4** tmuh.ai mini-app store 實作文件(store 端上傳/審查/展示;交伺服器端 Claude Code)

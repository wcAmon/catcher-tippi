# tomato-ears SPEC

一句話:瀏覽器錄音 → 本機 Nemotron ASR 引擎即時轉錄,零雲端依賴。

對應 `docs/superpowers/specs/2026-07-18-mini-app-store-design.md`(店規)第 3
節配方格式;本文件是 SPEC.md 的角色——「要做出什麼」與「驗收標準」,同時也是
組裝 agent(PLAN.md 的執行者)的目標定義。

## 1. 使用者看到的東西

打開瀏覽器到 `http://127.0.0.1:43117/`,是一個單頁應用:

- **開始/停止錄音**按鈕;
- **backend 徽章**:顯示引擎回報的推論後端(`mlx`/`dml`/`cpu`,僅展示,不影響
  任何行為);
- **即時 partial 顯示**:錄音中持續更新的辨識文字——**快照替換語義**,每次
  更新是整段覆寫,不是逐字附加;
- **final 訊息列表**:每次「停止」後,把定稿文字累加進一個列表(不會清空
  上一輪的內容,可以錄好幾段分開看);
- **複製全部**:把目前列表的全部文字複製到剪貼簿;
- **匯出 .txt**:把目前列表存成一個純文字檔下載。

全程只有一個 Deno process:serve 這個網頁,也透過 WebSocket 把音訊轉給本機
推論引擎、把辨識結果轉回瀏覽器。沒有第二個服務、沒有資料庫、沒有雲端 API。

## 2. 功能清單(行為規格)

延續 Task 3(`reference/ui/`)訂定的行為規格,原樣列出供驗收比對:

1. **錄音管線**:`getUserMedia(mono)` → `AudioContext`(瀏覽器原生取樣率)→
   `AudioWorklet`:線性內插重採樣到 16 kHz、累積滿 **1600 samples**(≈100ms,
   對齊 `docs/protocol/asr-host-v1.md` 的 chunk 建議值)發一個 `Int16Array`
   chunk(`port.postMessage`,transferable,不複製記憶體)。停止錄音時,緩衝區
   裡不足 1600 samples 的殘餘也會被沖洗出來送出(`flush()` 交握,2 秒逾時
   保底),避免每次停止都固定丟失最後一小段語音。
2. **傳輸**:chunk 經 WebSocket **binary frame** 直接送(不包 JSON/base64);
   控制訊息(`{"type":"start"}`/`{"type":"stop"}`)走 **text frame**。伺服端
   把 binary frame 轉成 base64 塞進 `asr-host-v1` 協定的 `audio` 指令,寫進
   引擎子行程的 stdin。
3. **引擎**:一個常駐子行程,用 `docs/protocol/asr-host-v1.md`(凍結版本)描述
   的 stdin/stdout JSON-lines 協定溝通——
   - mac:`catcher-asr-host`(MLX / Metal 推論);
   - Windows:`nemotron-asr-host`(onnxruntime-genai + DirectML,實測輸 CPU
     則退 CPU,見店規第 6 節的推論引擎政策)。
     兩個 host 都是 wake 發布的 prebuilt binary,SHA-256 pin 在 `manifest.json`
     ——配方本身不編譯任何原生碼(店規第 8 條)。
4. **服務**:單一 Deno process,監聽 `127.0.0.1:43117`(**只綁 loopback**,
   店規第 5 條),同一個 port 上同時 serve 靜態 UI(`ui/`)與 `/ws`
   WebSocket。啟動後嘗試自動開啟系統預設瀏覽器(店規第 3 條「容易打開」);
   自動開啟失敗時降級為印出網址,不視為致命錯誤。
5. **安裝(setup)**:讀 `manifest.json` 的 `dependencies`,下載 engine host
   壓縮包與模型檔案,逐檔驗 SHA-256,原子安裝(下載到 `.part` → 驗證 →
   `rename`)。已存在且雜湊相符的檔案直接略過,`setup` 可安全重跑。
6. **啟動(start)**:讀 `~/tmuh-apps/_machine/machine-profile.json` 決定
   平台與（Windows 專屬的）已回填後端旗標,不重新探測、不重新下載——
   `start` 執行期**零對外網路**(見 SECURITY.md 的兩階段權限模型)。
7. **文字**:引擎輸出一律已轉繁體中文(host 端 opencc `s2twp`),消費端
   (`app.js`/`server.ts`)不再轉換,也不對 `backend` 值或 `error.message`
   內容做任何分支邏輯(協定文件明文禁止)。

## 3. 驗收標準

**`deno task verify:mac`(Windows:`deno task verify:win`)全數通過 = 完成。**
店規第 6 條:「`verify/` 測試全數通過才算建構完成,agent 不得自行宣告成功」
——沒有例外,不接受「大致上可以動」這種主觀判斷。

`verify/` 套件涵蓋五個面向,對已完成 `setup` 的真實安裝目錄跑(真 engine
host、真模型,不是 fake/stub):

| 測試檔                | 驗什麼                                                                     |
| --------------------- | -------------------------------------------------------------------------- |
| `integrity_test.ts`   | `manifest.json` 宣告的每一個相依檔案,在安裝目錄裡存在且 SHA-256 相符       |
| `protocol_test.ts`    | 對真 host 餵 fixture 音訊,轉錄結果與參考文字的正規化編輯距離 ≤ 0.25        |
| `service_test.ts`     | 完整 HTTP + WebSocket 服務堆疊走一輪 ready→start→chunks→partial→stop→final |
| `binding_test.ts`     | 服務無法從非 loopback 網路介面連線                                         |
| `permissions_test.ts` | `deno.json` 各平台 task 的執行旗標與 `manifest.json.permissions` 逐字相等  |

另有 `asr_metric_test.ts`(11 個純函式單元測試,固定案例)把 `protocol_test.ts`
用到的正規化編輯距離演算法本身釘住——這不是「使用者驗收」的一部分(不需要
真 host),但同樣包含在 `deno test verify/` 的執行範圍內,全綠是整體驗收的
一部分。

開發期用的 dev-time 測試(`reference/*_test.ts`,fake-engine/stub,不需要真
模型)**不是**驗收標準的一部分——它們的角色與分工說明見 PLAN.md 附錄。

## 4. 非目標(v1 不做)

- **Diarization(說話者分離)**:留給後續 mini-app(sortformer 引擎已在
  `crates/sortformer-mlx` 存在,但沒有接進這個配方)。
- **TTS(文字轉語音)**:同樣留給後續 mini-app(`tts-speaker`/VoxCPM2 資產)。
- **語音輸入注入(voice input injection)**:跨 app 文字注入需要 OS 特權,
  依店規第 8 節「非目標」明文排除,留在原生 Tippi app,不進 mini-app store。
- **檔案批次轉錄**:v1 只做即時麥克風串流,不接受上傳既有音訊檔案。
- **Linux 支援**:引擎相依只 pin 了 macOS arm64 與 Windows x64 兩份
  prebuilt binary(`platformFromProfile` 對其他組合直接 throw)。
- **多使用者/多分頁並發**:`server.ts` 的 `EngineClient` 是單一訂閱者,
  新分頁連上會蓋掉舊分頁的訂閱(見 `reference/server.ts` 檔頭 why 說明),
  這是刻意的 v1 降級,不是 bug。
- **匯出 .json**:UI 只提供「複製全部」與「匯出 .txt」,不含結構化 JSON
  匯出(Task 3 明文範圍,見 `task-3-report.md` 偏差 4)。

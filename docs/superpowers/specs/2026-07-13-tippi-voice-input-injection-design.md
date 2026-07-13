# Tippi 子專案 B 設計：語音輸入與跨 App 文字注入

> **狀態：已核准。** 2026-07-13 完成產品與技術設計確認，可進入實作計畫。
>
> 前置：子專案 A（雙分頁外框與轉錄分頁強化）已合入 `main`（`cc535bc`）。
> 本專案以可用的語音輸入功能取代 `VoiceInputPlaceholderView.swift`。

## 1. 目標

在「語音輸入」分頁開始錄音後，Tippi 將穩定的語音轉錄文字持續注入目前位於前景、
且已有鍵盤焦點的其他 macOS App 或網頁輸入框。使用者說出固定口令 **`Tippi Go`** 時，
Tippi 不會輸入口令本身，而會完成口令前的文字注入並送出一次 Enter。

典型流程：

1. 使用者在 Tippi 的「語音輸入」分頁按下「開始語音輸入」。
2. 使用者切回 ChatGPT、瀏覽器或其他目標 App，將游標放入輸入框。
3. Tippi 將穩定的轉錄增量注入目標輸入框。
4. 使用者說 `Tippi Go`。
5. Tippi 排除口令音訊、補完口令前文字，然後送出一次 Enter。
6. Tippi 重設當輪 ASR 與 KWS 串流，繼續等待下一段訊息。

## 2. 已核准決策

### 2.1 跨 App 注入與權限

- 移除 App Sandbox；本地 ad-hoc 簽章的 Tippi 主程式直接執行注入，不增加 helper 或 XPC。
- 使用 `AXIsProcessTrustedWithOptions` 檢查並請求 macOS「輔助使用」權限。
- 文字以 `CGEvent` Unicode 鍵盤事件注入，送出以 Return 鍵事件完成。
- v1 不使用 `AXUIElement.setValue`，也不覆寫剪貼簿後模擬貼上。
- Tippi 不主動切換 App、視窗或鍵盤焦點；使用者負責先聚焦目標輸入框。
- Tippi 自己位於前景時暫停注入，避免把轉錄文字打回本 App，但錄音與辨識可繼續。

`CGEvent` 的 Unicode 事件對不同 UI 框架仍可能有相容差異，因此 TextEdit、Chrome 與
ChatGPT 都列為真實環境驗收項目；若驗收發現特定目標不接受 Unicode 事件，再另案評估
剪貼簿 fallback，不先把它放入 v1。

### 2.2 第二模型：離線關鍵字偵測

送出命令不依賴主 ASR 的文字結果。短口令在既有 Nemotron ASR 實測中可能被轉成
`Tippy` 且漏掉 `Go`，因此使用獨立的離線 keyword spotting（KWS）模型。

選定 **sherpa-onnx open-vocabulary KWS**，搭配
`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` 的 chunk-16 模型：

- 可在本機 Apple Silicon 離線執行，Rust 端以 `sherpa-onnx` crate 靜態連結。
- 不需要為 `Tippi Go` 另外蒐集資料或訓練模型。
- 口令詞條固定為 `TIPPY GO :1.5 #0.25 @TIPPI_GO`；發音 token 為
  `T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO`。
- `Tippy` 是模型詞典可辨識的發音拼法，UI 與產品名稱仍顯示 `Tippi Go`。
- 使用 encoder int8、decoder fp32、joiner int8 與 tokens，安裝後約 5.45 MB。

模型來源固定為：

- Archive：[`sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2`](https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2)
- Archive SHA-256：`68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6`
- `encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx`：
  `408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48`
- `decoder-epoch-13-avg-2-chunk-16-left-64.onnx`：
  `63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21`
- `joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx`：
  `190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1`
- `tokens.txt`：`2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0`

首次準備語音輸入時下載約 32.9 MB 的官方 archive 至 staging，先驗證 archive hash，
以系統 `tar` 解壓，再驗證四個選定檔案的 hash，最後以原子移動安裝。正式模型目錄只保留
上述四個檔案、產生的 `keywords.txt` 與第三方授權說明。任何驗證失敗都不得啟用 KWS。

### 2.3 單一音訊流、雙辨識分支

App 維持單一 `AudioRecorder`，每批 16 kHz mono PCM 同時送往兩條分支：

```text
AudioRecorder
    ├── 主 ASR（Nemotron）── 穩定文字增量 ── TextInjectionCoordinator ── CGEvent
    └── KWS（sherpa-onnx）── TIPPI_GO + 起始時間 ────────────────┘
```

「轉錄」與「語音輸入」共用全域錄音所有權，一次只能有一個模式錄音。啟動其中一個模式後，
另一個分頁的開始按鈕停用。轉錄模式不載入或執行 KWS。

### 2.4 穩定文字增量

語音輸入模式使用 `catcher_text` 所代表的 append-only timed tokens，不依賴 diarization segment
第一次變成 final 的時點；後者仍可能因後續語音而延長。

Controller 保存已注入的穩定前綴，每次只注入新增加的 suffix：

- 新文字以前次文字為 prefix：只送出新增 suffix。
- 文字沒有共同的 append-only 關係：fail closed，停止注入並顯示可恢復錯誤。
- v1 不送 Backspace、不刪除或改寫目標 App 既有文字。

這提供「小段增量出現」的體驗，同時避免跨 App 倒退修訂造成誤刪。

### 2.5 `Tippi Go` 的時間切除與送出順序

KWS 成功事件必須包含 `TIPPI_GO` 與口令起始時間。收到事件後依序執行：

1. 暫停將新音訊送入本輪辨識器。
2. 主 ASR flush 尚未完成的音訊，但只保留時間早於口令起始點的 timed tokens。
3. 計算並注入尚未送出的口令前文字 suffix。
4. 捨棄位於口令起始點之後的 tokens，確保 `Tippi Go` 不進入目標輸入框。
5. 送出一組 Return key-down / key-up 事件。
6. 顯示「已送出」，並重設 ASR、KWS、已注入前綴與本輪時間基準。
7. 恢復同一錄音工作階段，接收下一段訊息。

同一偵測只可送出一次；stream reset 與 coordinator 狀態共同作為 debounce。若 flush 或 cutoff
失敗，不送 Enter，避免送出不完整內容。

## 3. 元件設計

### 3.1 Rust / FFI

- 在 `catcher-ffi` 增加 KWS runtime wrapper，負責模型載入、PCM push、結果與時間戳、reset。
- KWS handle 與主 catcher handle 分開，生命週期由 Swift controller 明確管理。
- 增加主 ASR 的「在指定時間前完成並取得文字」能力，直接使用既有 `TimedToken` 時間資料。
- 新增的 C ABI 函式維持錯誤碼與 `catcher_last_error` 慣例，不讓 panic 跨過 FFI。
- Rust 層擁有模型 runtime 狀態；Swift 不直接接觸 sherpa-onnx 型別。

### 3.2 Swift 核心

- `KeywordSpotterClient` actor：包裝 KWS C ABI，提供 prepare、push、reset、finish。
- `TextInjecting` protocol：`inject(_:)`、`submit()`、權限狀態與權限請求。
- `CGEventTextInjector`：唯一包含 ApplicationServices / CoreGraphics 系統呼叫的薄實作。
- `TextInjectionCoordinator`：管理 append-only diff、命令 cutoff、注入後送出順序、debounce 與
  fail-closed 狀態；透過 mock `TextInjecting` 單元測試。
- `TranscriptionController` 增加明確的錄音模式（transcription / voiceInput），集中協調單一
  `AudioRecorder`、ASR 與 KWS，不讓兩個 View 各自搶錄音資源。

跨 actor 或 audio callback 的狀態變化必須序列化；UI 更新仍在 MainActor。

### 3.3 模型路徑遷移

移除 sandbox 後，模型根目錄改為：

`~/Library/Application Support/Tippi/Models`

啟動準備流程在新目錄為空、舊 sandbox 目錄存在時，自動遷移：

`~/Library/Containers/com.wcamon.tippi/Data/Library/Application Support/Tippi/Models`

- 優先同檔案系統 move，失敗再 copy 到 staging、驗證完整後原子替換。
- 新目錄已有內容時絕不覆寫。
- 遷移失敗時保留舊來源，UI 顯示原因，允許走既有下載流程；不可刪除唯一模型副本。

## 4. 語音輸入分頁

`VoiceInputTabView` 取代占位畫面，使用與轉錄分頁一致的視覺語言，包含：

- 權限狀態：未授權時說明需求，提供「要求輔助使用權限」與「開啟系統設定」。
- 模型狀態：第一次進入分頁便自動準備主 ASR 與 KWS，顯示下載、驗證、載入、失敗與重試狀態。
- 主要控制：單一「開始語音輸入／停止」按鈕；權限或模型未就緒時停用。
- 固定口令 badge：`口令：Tippi Go`。
- 目標狀態：顯示目前前景 App 名稱；Tippi 位於前景時提示「請切到目標輸入框」。
- 最近活動：顯示最近注入文字與短暫的「已送出」狀態。
- 使用說明：先聚焦目標輸入框，開始說話，最後說 `Tippi Go`。

若無法證明目標輸入框存在，App 不嘗試用 AX 尋找或改變焦點；UI 顯示「已嘗試注入至
<App 名稱>」，真實目標是否接受文字由使用者可見結果確認。

## 5. 錯誤與安全行為

- 輔助使用權限缺失：禁止開始語音輸入，不送任何合成鍵盤事件。
- KWS 模型缺失、hash 不符或載入失敗：禁止開始，不退回主 ASR 文字比對口令。
- 主 ASR 可用但 KWS 中途失敗：停止整個語音輸入工作階段，不繼續無命令保護的注入。
- 穩定前綴分歧：停止注入並提示重啟本輪；不送 Backspace 或 Enter。
- 命令 cutoff / flush 失敗：不送 Enter，保留可診斷錯誤。
- Tippi 在前景：暫停注入及送出並提示切換目標；不得注入本 App。
- 停止按鈕：結束兩條辨識串流，但不自動送出尚未完成的文字。

## 6. 測試與驗收

### 6.1 Rust 自動測試

- padded `Tippi Go` fixture 能回傳 `TIPPI_GO` 與合理的起始時間。
- 至少三個無關英文／中文 fixture 不觸發命令（smoke-level negative coverage）。
- cutoff 只保留口令起點前的 `TimedToken`，口令內容不出現在完成文字。
- reset 後可辨識第二段命令，且不沿用前一輪狀態。
- 模型路徑錯誤、runtime 錯誤與空 handle 不造成跨 FFI panic。

### 6.2 Swift 自動測試

- append-only prefix 只注入 suffix；prefix 分歧會 fail closed。
- 命令流程嚴格維持「補注入 → submit → reset」順序。
- 重複 KWS event 只送出一次；cutoff 失敗不送出。
- mock 權限未授權、KWS 未就緒與中途失敗時，開始或持續注入會被阻擋。
- transcription / voiceInput 錄音所有權互斥。
- 舊模型遷移涵蓋 move、copy fallback、不覆寫與失敗保留來源。
- KWS installer 涵蓋 archive hash、個別檔案 hash、staging cleanup 與原子安裝。

### 6.3 Build 驗證

- `cargo test -p catcher-ffi`
- `cargo build -p catcher-ffi --release`
- `swift test --package-path apps/tippi`
- `apps/tippi/scripts/build-app.sh`
- `apps/tippi/scripts/verify-app.sh`

### 6.4 真實環境驗收

- 首次要求與撤銷後重新要求輔助使用權限。
- TextEdit 原生文字欄位。
- Chrome 的 textarea 與 contenteditable。
- ChatGPT 網頁與可用的桌面 App。
- 中文、英文與中英混合內容注入。
- 說完內容後說 `Tippi Go`：口令不出現，只送一次 Enter。
- 送出後不停止錄音，可再完成第二段訊息。
- Tippi 位於前景、焦點移開、非文字目標及切換目標 App 的行為。
- 已有 sandbox 模型時只遷移、不重新下載大型 ASR 模型。

## 7. v1 非目標

- 自訂口令、敏感度 UI、多組 voice commands 或 Cmd+Enter 設定。
- 自動尋找、選擇或聚焦目標 App／輸入框。
- partial revision 的 Backspace、游標定位或跨 App 文字修訂。
- 剪貼簿 paste fallback；只有真實相容性驗收失敗才另案加入。
- 保留 sandbox 的 helper / XPC 架構或 Mac App Store 發佈。
- 改變既有「轉錄」分頁的顯示、匯出與清除語意。

## 8. 已知限制

- 使用者必須自行保持正確輸入框焦點；在注入期間切換焦點，文字可能進入不同目標。
- `CGEvent` 無法回報目標 UI 是否真正接受 Unicode 文字，因此 UI 只能回報事件已嘗試送出。
- KWS smoke test 不能替代不同口音、距離與噪音環境的實測；v1 先固定口令與參數，根據實際
  誤觸／漏觸結果再調整 threshold、boost 或模型 chunk。
- 移除 sandbox 與 ad-hoc 重簽後，macOS 可能要求使用者重新確認輔助使用權限。

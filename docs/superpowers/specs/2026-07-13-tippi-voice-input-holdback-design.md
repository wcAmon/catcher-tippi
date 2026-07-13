# Tippi 語音輸入 1.5 秒安全緩衝修正設計

> **狀態：已核准。** 2026-07-13 已確認採用方案 A：保留即時注入，但加入約
> 1.5 秒安全緩衝。本文件是
> `2026-07-13-tippi-voice-input-injection-design.md` 的錯誤修正補充；未被本文件修改的
> 原設計仍然有效。

## 1. 問題與實機證據

第一輪真實語音輸入在 `Tippi Go` 被偵測後停止，UI 顯示：

```text
Transcription stopped being append-only
(previous: <已注入的長文字>, current: 你)
```

權限、主 ASR 與 KWS 模型均已就緒。錯誤發生在 Controller 將
`KeywordDetection.startMs` 傳給 `catcher_finish_before` 時。

真實模型診斷顯示，sherpa-onnx chunk-16 KWS 的第一個 token timestamp 不是整段錄音的
絕對時間。相同 `Tippi Go` fixture 前置不同長度的靜音後，回傳值會循環或歸零：

| 前置靜音 | 偵測時已送入音訊 | 回傳 `startMs` |
| ---: | ---: | ---: |
| 1 秒 | 2.1 秒 | 960 ms |
| 5 秒 | 6.0 秒 | 0 ms |
| 10 秒 | 11.1 秒 | 320 ms |
| 20 秒 | 21.0 秒 | 640 ms |

舊流程把這些 chunk-relative 值當成 utterance-relative cutoff。長句在偵測命令後可能被裁成
最前方一小段，與已注入前綴分歧；此外，KWS 完成偵測前，主 ASR 也可能已把口令的部分文字
注入目標 App。

## 2. 目標與非目標

### 2.1 目標

- 保留邊說邊注入的體驗，但允許約 1.5 秒顯示延遲。
- cutoff 只依賴本輪實際送入的 PCM 樣本數，不再依賴 KWS token timestamp。
- `Tippi Go` 的 ASR token 不得進入目標輸入框。
- 命令前的安全文字補完後只送一次 Return。
- 送出後第二輪使用全新的 ASR、KWS、樣本時間與注入前綴。
- 繼續維持不使用剪貼簿、不送 Backspace、不改寫目標文字的安全界線。

### 2.2 非目標

- 不嘗試解讀或展開 sherpa-onnx 的 chunk-relative timestamp。
- 不用主 ASR 文字比對或刪除 `Tippi Go` 的拼字變體。
- 不加入第三個 VAD／alignment 模型。
- 不自動切換或聚焦目標 App。
- 不在本次修正加入可調整的延遲 UI。

## 3. 時間模型

語音輸入 Controller 為每一輪維護 `receivedSampleCount`。錄音固定為 16 kHz mono，因此：

```text
audioEndMs = floor(receivedSampleCount * 1000 / 16000)
stableCutoffMs = max(0, audioEndMs - 1500)
```

每個 audio chunk 在送入主 ASR 與 KWS 前先加入 `receivedSampleCount`，兩條分支因而共用同一個
utterance-relative 時間軸。1.5 秒是固定的 v1 安全常數，集中定義為
`VoiceInputTiming.holdbackMs`，不散落於 Controller 或測試。

KWS 的 `startMs` 仍保留在 ABI 與 Swift 型別中作診斷用途，但 Controller 不得用它裁切主
ASR。

## 4. 非破壞性穩定文字快照

Rust FFI 新增非破壞性查詢：

```c
const char *catcher_text_before(
    catcher_handle_t *handle,
    uint64_t cutoff_ms
);
```

行為：

- 從 `timed_tokens` 選出 `token.frame * 80 < cutoff_ms` 的 token。
- 使用既有 tokenizer 與 OpenCC 路徑產生文字，存入 handle 擁有的獨立
  `text_before_cutoff` buffer。
- 不 flush、不刪除 token、不改變 session state，也不改寫 `catcher_text` 的完整 transcript。
- 成功回傳 borrowed UTF-8 pointer；失敗回傳 `NULL` 並寫入既有 `catcher_last_error`。
- pointer 的生命週期與其他 borrowed string API 相同：下一次修改 handle 或再次查詢前有效。

Swift `CatcherServing` 增加：

```swift
func text(before cutoffMs: UInt64) async throws -> String
```

`CatcherClient` 只負責將 `NULL` 轉為 `CatcherClientError.operationFailed`，不在 Swift 重做 token
或時間計算。

## 5. 一般注入資料流

未偵測到命令時：

1. 增加 `receivedSampleCount`。
2. 將同一批 PCM 依序送入主 ASR 與 KWS。
3. 計算 `stableCutoffMs`。
4. 每個 audio chunk 都讀取 `catcher.text(before: stableCutoffMs)`；不能只在主 ASR 回報新
   token 時查詢，因為後續靜音仍會讓既有 token 跨過 1.5 秒穩定界線。
5. 將穩定快照交給 `TextInjectionCoordinator.consume`；Coordinator 仍只注入相對於前次穩定
   快照的新 suffix。

當錄音不足 1.5 秒時 cutoff 為 0，穩定快照為空，不注入任何文字。cutoff 隨樣本數單調
增加，正常路徑的穩定文字也必須維持 append-only；若仍分歧，原本 fail-closed 行為保留，
因為那代表另一個未預期的資料一致性問題。

## 6. `Tippi Go` 送出資料流

KWS 偵測 `TIPPI_GO` 時：

1. 使用本批 PCM 更新後的 `receivedSampleCount` 計算 `stableCutoffMs`。
2. 呼叫 `catcher.finish(before: stableCutoffMs)`，完成主 ASR 並只保留安全緩衝之前的 token。
3. 將完成文字交給 `TextInjectionCoordinator.submit`，補注入尚未送出的安全 suffix。
4. 成功時送出一次 Return。
5. 重啟主 ASR、reset KWS、reset Coordinator，並將 `receivedSampleCount` 歸零。
6. 繼續同一個麥克風工作階段，等待下一段訊息。

若 cutoff 前沒有文字，命令只重設本輪並顯示「沒有可送出的文字」，不得送出空白 Return。
`TextInjectionEvent` 因此新增 `.nothingToSubmit`；`TextInjectionCoordinator.submit` 在安全完成
文字為空且尚未注入任何前綴時回傳此事件，不呼叫 injector 的 `submit()`。
若目標是 Tippi 或沒有可用前景 App，沿用原設計：不注入、不送出，提示使用者切回目標後
重說命令。

立即重複的 KWS event 被丟棄時，也必須同步重設 `receivedSampleCount`，避免下一輪時間軸從
舊音訊延續。

## 7. 使用者體驗

- 文字約在說出後 1.5 秒出現在目標輸入框。
- UI 與 README 將操作提示改為「內容說完後短暫停頓，再說 Tippi Go」。
- 建議停頓約 0.5 秒；一般語速的口令與停頓會共同落在 1.5 秒安全區內，命令前內容則落在
  cutoff 之前。
- 若內容與口令完全連在一起，安全優先：最後極短的一小段內容可能被捨棄，不能以讓口令
  外洩作為補償。
- 既有「停止」按鈕仍不注入或送出 holdback 中的尾段。

## 8. 錯誤處理

- `catcher_text_before` 失敗：停止本輪，不注入、不送 Return，保留底層錯誤。
- KWS 失敗：停止整個語音輸入工作階段，不退回文字口令比對。
- 穩定快照分歧：維持 fail closed；錯誤文案改為指出穩定緩衝分歧，而不是把整個主 ASR
  模型標記成準備失敗。
- 模型已成功載入後的串流錯誤，語音輸入分頁標題顯示「語音輸入已停止」與實際原因；只有
  prepare/load 失敗才顯示「語音辨識模型準備失敗」。
- command finish/cutoff 失敗：不送 Return。
- 所有失敗清理都必須歸零 `receivedSampleCount` 並 reset Coordinator/KWS，讓重試從乾淨本輪
  開始。

## 9. 測試與驗收

### 9.1 Rust / FFI

- cutoff 嚴格使用 `frame * 80 < cutoff_ms`。
- `catcher_text_before` 不修改完整 transcript、timed tokens 或 session state。
- cutoff 從 0 單調增加時，快照只增加安全 token。
- `NULL` handle 與 tokenizer 錯誤遵循既有錯誤慣例。

### 9.2 Swift

- 前 1.5 秒不注入；之後只注入穩定快照的新 suffix。
- KWS 回傳 `startMs = 0`、`320`、`960` 時，只要樣本數相同，Controller 必須呼叫相同的
  `finish(before:)` cutoff。
- 偵測命令時順序仍是「安全補注入 → Return → ASR/KWS/Coordinator/樣本計數 reset」。
- cutoff 前沒有文字時不送空白 Return。
- 送出後第二輪從 0 ms 開始；立即重複命令不造成第二次 Return。
- 停止、KWS 失敗、注入失敗與 retry 都會歸零樣本計數。

### 9.3 真實模型與 App

- 以 1、5、10、20 秒前置音訊驗證 KWS timestamp 即使循環，cutoff 仍由樣本數決定。
- TextEdit：中文、英文、中英混合與 emoji；口令不出現，Return 恰好一次。
- 送出後第二段訊息仍可運作。
- Chrome textarea、contenteditable、ChatGPT 網頁及桌面 App（有安裝時）。
- Tippi 前景、非文字焦點與沒有目標時不崩潰、不送出。
- 驗收時記錄內容結尾、停頓長度、口令語速及是否遺失最後文字，作為日後調整 1.5 秒常數的
  依據。

## 10. 成功條件

- 原始實機案例不再出現 `previous: <長文字>, current: 你` 的 cutoff 分歧。
- `Tippi Go` 及其主 ASR 近似拼字不進入目標輸入框。
- 使用者短暫停頓後說命令時，命令前內容完整、只送一次 Return。
- 完整 Rust、Swift、App bundle 測試通過，並完成至少 TextEdit 的真實權限注入驗收。

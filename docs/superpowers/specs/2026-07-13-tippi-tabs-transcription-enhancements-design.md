# Tippi 子專案 A 設計:雙分頁外框與轉錄分頁強化

前置:Sortformer 三階段已全部合入 main(`e329dd0`)並通過實機驗收。
後續:子專案 B(語音輸入注入其他 app + 「送出」語音指令)另立 spec,本文件只預留分頁占位。

## 目標

把 Tippi 主視窗改為雙分頁結構,並強化轉錄分頁:清除重來、JSON 匯出、命名教學提示;同時修掉匯出寫檔失敗只 beep 的既有 Minor。

## 決策(已與使用者確認)

- **拆成兩個子專案**:A(本文件)先做;B(語音輸入)依賴 A 的分頁外框,獨立 spec。
- **命名維持現行 popover**(點訊息上的名字改名),不做設定 modal;加 hover 教學提示。
- **清除重來與 JSON 匯出都放 footer**,本階段不引入設定 modal。
- **清除只在停止後可按**(`state == .ready`),附確認對話框;錄音中 disabled。
- **分頁二先出現但為占位畫面**(「語音輸入——即將推出」),外框結構一次到位。
- **方案一:原生元件到底**——SwiftUI `TabView`、單一「匯出…」按鈕在 `NSSavePanel` 內選格式、`.help()` tooltip。

## 1. 分頁外框(TippiApp)

- `ContentView` 頂層改為 `TabView`,兩個分頁:
  - 「轉錄」:現有整個轉錄畫面(狀態列、警告橫幅、訊息列表、footer)原封搬進新檔 `TranscriptionTabView.swift`。
  - 「語音輸入」:新檔 `VoiceInputPlaceholderView.swift`,置中顯示 SF Symbol `waveform`(大尺寸、次要色)與文案「語音輸入——即將推出」(次要前景色)。
- `TranscriptionController` 仍由 App 層建立並注入,切換分頁不影響錄音狀態。
- 分頁標題字串:「轉錄」、「語音輸入」。

## 2. `Message` 補回 `endMs`(TippiCore)

- `Message` 新增 `public let endMs: UInt64`;`init(id:segment:)` 帶入 `segment.endMs`。
- 既有引用與測試同步更新;無行為變更,純資料欄位補齊(JSON 匯出需要)。

## 3. `TranscriptJSONExporter`(TippiCore,新檔)

- 輸入 `[Message]` 與 `speakerNames: [Int: String]`,輸出 `Data`(pretty-printed、sorted keys、UTF-8):

```json
{
  "messages" : [
    {
      "end_ms" : 5678,
      "final" : true,
      "name" : "小明",
      "speaker" : 0,
      "start_ms" : 1234,
      "text" : "今天先討論這個。"
    }
  ]
}
```

- `name` 一律填顯示名:`speakerNames[speaker] ?? "說話者 \(speaker + 1)"`(與 `TranscriptFormatter` 同規則,抽共用)。
- key 用 snake_case,與 FFI segments JSON 契約一致;`JSONEncoder` 設 `.prettyPrinted` + `.sortedKeys`(golden 測試可確定性比對)。
- 空 `messages` 輸出 `{ "messages" : [ ] }`,不視為錯誤(匯出按鈕本來就在無訊息時 disabled,此為防禦行為)。

## 4. 匯出 UI(TippiApp)

- footer 維持單一「匯出…」按鈕;`NSSavePanel.allowedContentTypes = [.plainText, .json]`(`import UniformTypeIdentifiers`),使用者在面板的格式選單選 .txt 或 .json,預設檔名維持「Tippi 逐字稿.txt」。
- 依使用者最終選擇的 URL 副檔名分流:`json` → `TranscriptJSONExporter`;其餘 → 既有 `TranscriptFormatter` 全文(行格式不變,名稱分隔為全形冒號 U+FF1A)。
- **修既有 Minor**:寫檔拋錯時以 `NSAlert` 顯示(title「匯出失敗」,informative 帶 `error.localizedDescription`),不再只 beep。

## 5. 清除重來

- footer 新增「清除」按鈕:啟用條件 `state == .ready && !messages.isEmpty`,其餘狀態 disabled。
- 按下跳確認 alert:title「清除全部訊息?」,message「將移除所有訊息與說話者命名,無法復原。」,按鈕「清除」(destructive)/「取消」。
- 確認後呼叫 `TranscriptionController.clearTranscript()`(新方法):`guard state == .ready`,清空 `messages`、`speakerNames`、`warningMessage`。

## 6. 命名教學提示

- `MessageRow` 的名字按鈕加 `.help("點擊可重新命名")`;hover 時名字加底線(`onHover` + `.underline(isHovering)`)暗示可點。

## 7. 測試

- **TippiCore(swift test;前置 `cargo build -p catcher-ffi --release`)**:
  - `TranscriptJSONExporter` golden 測試:自訂名、未命名預設「說話者 N」、`start_ms`/`end_ms`/`final` 欄位、空訊息、非 ASCII 文字。
  - `clearTranscript()`:ready 時清空三者;非 ready 呼叫為 no-op。
  - `Message.endMs` 由 segment 帶入。
- **UI(手動驗收)**:分頁切換不中斷錄音、占位畫面、save panel 格式切換與兩種格式檔案內容、清除確認流程、匯出失敗 alert(可用唯讀目的地模擬)、tooltip 與 hover 底線。

## 非目標

- 設定 modal、說話者命名跨錄音持久化、錄音中清除。
- 分頁二任何實質功能(子專案 B)。
- JSON 以外的新匯出格式(SRT 等)。
- ModelStore/引擎/FFI 任何變更。

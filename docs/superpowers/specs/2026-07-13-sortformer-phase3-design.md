# Sortformer 階段三設計:Tippi 訊息 UI 與雙模型

母設計文件:`docs/plans/2026-07-12-sortformer-diarization-design.md`(階段三節,line 188)。
前置:階段二已合入 main(`655e17d`)——AOSC 串流、fusion、s2twp、C ABI v2、`catcher transcribe --diar-model`。

## 目標

Tippi 端到端的 who-said-what 體驗:首次啟動合併下載 ASR+diar 雙模型、錄音顯示分說話者訊息列表、說話者命名、複製/匯出;並完成階段二最終審查指定的三個 Rust 側旗標。

## 決策(已與使用者確認)

- **diar 模型必裝,合併下載**:首次啟動一次下載兩個 artifact(約 660MB + 121MB),進度按位元組加權合併。狀態機不變,永遠有說話者標記,無 ASR-only UI 模式。
- **執行期降級用非阻斷橫幅**:diar 執行期錯誤時錄音繼續,訊息列表上方顯示「說話者分離已暫停,文字繼續轉寫」;後續文字歸到最後一位已知說話者(fusion 既有行為)。下次錄音 `catcher_start` 重建 diarizer 後橫幅消失。
- **架構方案 A**:兩個 ModelStore 實例 + 薄協調器;segments JSON 在 CatcherClient 層解碼成型別。
- 說話者命名與訊息在新錄音開始時重設(母文件既定)。
- 複製/匯出行格式:`[03:24] 小明:今天先討論這個。`(fullwidth colon,母文件既定);錄音中也可複製/匯出,tentative 文字照當前內容輸出。

## 1. Rust 側前置修正(三個必辦旗標)

### (a) diar 降級可恢復(`crates/catcher-ffi`)

- `catcher_create` 時把 diar_model_path 複製保存在 handle 內(現況:載入後即丟棄路徑,執行期一次錯誤永久失去 diarization)。
- `catcher_start` 語義:若 handle 有 diar 路徑但 diarizer 已被執行期錯誤丟棄,就地重建。重建成功 → 清除 warning;重建失敗 → 設定 warning、以純 ASR 繼續,**start 本身不失敗**(與執行期降級語義一致)。
- create 時 diar 載入失敗 → create 失敗,維持既有行為不變。
- 兩份 `catcher.h`(crates/catcher-ffi/include 與 apps/tippi/Sources/CCatcher/include)同步更新 `catcher_start` 的契約註解。

### (b) fusion 邊界標點(`crates/nemotron-mlx/src/fusion.rs`)

- `segments()` 產出各段落文字後加後處理:若段落文字以標點開頭(至少涵蓋 `,。?!、:;…`),把整串前導標點移到前一段落尾端(首段落除外)。純外觀修正,不改 attribution/anti-flicker 邏輯,不改 `SpeakerSegment` 欄位。

### (c) rust-toolchain.toml(workspace 根)

- 鎖定目前開發使用的 stable 版本:`channel = "1.95.0"`(本機 `rustc --version` 實測),消除 cargo fmt 版本飄移。

## 2. ModelStore 雙 artifact(Swift,方案 A)

- `ModelStore`:寫死的 `modelDirectoryName` 改為建構參數(預設 `catcher-asr-mlx-int8`,既有呼叫不變),下載/staging/SHA-256 驗證邏輯零改動。
- `ModelManifest.swift` 新增 `[ModelFile].diarizationRelease`,7 個檔案,hash 與位元組數取自 main commit `e556258` body:
  - `weights.safetensors` a02b1a83… 127,218,628 bytes
  - `manifest.json` b777fce8… 139,278
  - `config.json` 6c0418a4… 4,469
  - `LICENSE` aae70c7d… 10,321
  - `NOTICE.md` 5ce32ffb… 458
  - `NVIDIA_MODEL_CARD.md` 86d9ff08… 27,357
  - `README.md` 0e2e2491… 642
  - repo URL:`https://huggingface.co/wcamon/catcher-diar-mlx-int8/resolve/main/`
- 新增:
  ```swift
  public struct ModelBundle: Equatable, Sendable { public let asr: URL; public let diar: URL }
  public protocol ModelBundleInstalling: Sendable {
      func installIfNeeded(progress: @escaping @Sendable (Double) -> Void) async throws -> ModelBundle
  }
  ```
- `ModelBundleInstaller`:持兩個 `ModelInstalling` 與各自總位元組數,循序安裝(先 ASR 後 diar),進度按位元組加權合併(ASR 佔 ~0.845、diar 佔 ~0.155,由 manifest byteCount 加總計算,不寫死)。已裝好的一側其內部 progress(1.0) 自然映射為該側佔比。任一側失敗即拋錯。

## 3. CatcherClient / CatcherServing v2(Swift)

```swift
public struct SpeakerSegment: Codable, Equatable, Sendable {
    public let speaker: Int
    public let startMs: UInt64   // JSON key "start_ms"
    public let endMs: UInt64     // JSON key "end_ms"
    public let text: String
    public let isFinal: Bool     // JSON key "final"
}
public struct TranscriptUpdate: Equatable, Sendable {
    public let segments: [SpeakerSegment]
    public let warning: String?
}
public protocol CatcherServing: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> TranscriptUpdate?  // CATCHER_NO_UPDATE → nil
    func finish() async throws -> TranscriptUpdate
}
```

- `CatcherClient.init(modelDirectory:diarModelDirectory:language:lookahead:)`——diar 目錄為必填(必裝決策),`catcher_create` 第二參數不再遞 nil。
- push/finish 在 `CATCHER_OK`(finish 亦接受 `CATCHER_NO_UPDATE`)時讀 `catcher_segments` 解 JSON + 讀 `catcher_warning`;解碼失敗視為 `operationFailed`。`catcher_text` 不再使用。
- `CatcherFactory` 簽名改為 `(ModelBundle) async throws -> any CatcherServing`。

## 4. TranscriptionController(Swift)

- `text: String` 移除,改 `messages: [Message]`:
  ```swift
  public struct Message: Identifiable, Equatable, Sendable {
      public let id: Int          // 列表索引;每次更新整批重建
      public let speaker: Int
      public let startMs: UInt64
      public let text: String
      public let isFinal: Bool
  }
  ```
- `speakerNames: [Int: String]`(公開可變);顯示名規則 `speakerNames[n] ?? "說話者 \(n + 1)"`(speaker 0 → 說話者 1)。
- `warningMessage: String?`:由 push/finish 的 `warning` 更新;`startRecording` 成功後歸 nil,同時清空 `messages` 與 `speakerNames`。
- `prepare()` 改用 `ModelBundleInstalling`,其餘狀態機轉移(modelMissing→downloading→loading→ready→recording→finishing→ready、任意→failed)不變。

## 5. ContentView 訊息列表(SwiftUI)

- transcript pane 改為 `ScrollView + LazyVStack` 訊息列表,自動捲到最新。每則訊息:
  - 說話者名(可點,popover 內 TextField 改名,寫回 `speakerNames`,全列表立即重繪)+ 穩定 accent 色(4 色調色盤,speaker index 取模)。
  - `[mm:ss]` 起始時間戳(次要色)。
  - 內文;最後一則若 `isFinal == false` 以次要前景色呈現並隨 push 即時更新。
- warning 橫幅:訊息列表上方,非阻斷,黃色系,文案「說話者分離已暫停,文字繼續轉寫」;`warningMessage == nil` 時不顯示。
- 工具列/footer 動作:
  - 「複製全部」→ NSPasteboard 寫入全文。
  - 「匯出…」→ NSSavePanel(UTF-8 `.txt`,sandbox 由 user-selected file write 滿足)。
  - 每則訊息 context menu「複製此則」。
  - messages 為空時三者 disabled。
- 行格式共用 `TranscriptFormatter`(TippiCore):`[mm:ss] 顯示名:內文`,mm 超過 99 分鐘自然進位(如 `[102:07]`);「複製全部」與匯出為逐行 join(`\n`)。

## 6. 測試

- **Rust**:
  - fusion 標點後處理單元測試(前導標點搬移、首段不動、非標點開頭不動、連續多字元標點)。
  - FFI gated lifecycle(`NEMOTRON_MLX_ARTIFACT` + `SORTFORMER_MLX_ARTIFACT`)延伸:模擬 degrade 後 `catcher_start` 重建成功、segments 恢復非空;重建失敗(以無效路徑注入)時 warning 保留且 start 回 OK。
  - 既有 36 個測試 binary 全綠;rust-toolchain.toml 下 `cargo fmt --check` 通過。
- **Swift**(`swift test`,apps/tippi):
  - fake `CatcherServing` 腳本化餵 `TranscriptUpdate`:訊息組裝、tentative→final、改名即時重繪、新錄音 reset、warning 設定/清除。
  - `TranscriptFormatter` golden 測試(含未命名預設、超長分鐘)。
  - `ModelBundleInstaller`:fake installer 驗證加權進度單調遞增至 1.0、單側失敗傳遞、已安裝側直接回報佔比。
- **驗收**:Tippi 實機端到端——刪除模型目錄後首次啟動合併下載雙模型、錄音顯示分說話者訊息、改名、複製/匯出檔案內容正確。

## 非目標

- 說話者命名跨錄音持久化、歷史紀錄、多視窗。
- 匯出 .txt 以外格式(SRT/JSON)。
- ASR-only 模式開關。
- ModelStore 斷點續傳。

# Tippi 雙分頁外框與轉錄分頁強化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tippi 主視窗改為雙分頁(轉錄 + 語音輸入占位),轉錄分頁新增清除重來、JSON 匯出、匯出失敗 NSAlert、命名 hover 教學提示。

**Architecture:** 純 Swift 變更(spec:`docs/superpowers/specs/2026-07-13-tippi-tabs-transcription-enhancements-design.md`)。TippiCore 補 `Message.endMs`、新增 `TranscriptJSONExporter` 與 `TranscriptionController.clearTranscript()`(均可單元測試);TippiApp 把現有畫面搬進 `TranscriptionTabView`,頂層 `ContentView` 改為原生 `TabView`,footer 加清除按鈕、匯出改雙格式。Rust/FFI 零變更。

**Tech Stack:** Swift 6、SwiftUI、swift-testing(`@Test`/`#expect`)、AppKit(NSSavePanel/NSAlert)、UniformTypeIdentifiers。

## Global Constraints

- 所有面向使用者的字串用繁體中文,逐字使用:分頁「轉錄」「語音輸入」、占位「語音輸入——即將推出」、按鈕「清除」、alert title「清除全部訊息?」、alert message「將移除所有訊息與說話者命名,無法復原。」、alert 按鈕「清除」/「取消」、匯出失敗 title「匯出失敗」、tooltip「點擊可重新命名」。
- 行格式的名稱/內文分隔是**全形冒號 U+FF1A(`：`)**,本計畫不得改動任何既有行格式;新增字串若含冒號需位元組檢查。
- JSON 匯出 key 為 snake_case:`speaker`,`name`,`start_ms`,`end_ms`,`text`,`final`;`name` 規則 `names[speaker] ?? "說話者 \(speaker + 1)"`,必須經 `TranscriptFormatter.displayName` 取得,不得重複實作。
- Swift 測試/建置前置:先在 repo 根執行 `cargo build -p catcher-ffi --release`(連結 `target/release/libcatcher_ffi`)。
- 測試指令:`swift test --package-path apps/tippi`(在 repo 根執行);建置:`swift build --package-path apps/tippi`。
- 不得修改:`apps/tippi/Resources/Tippi.entitlements`、任何 `crates/` 下檔案、`~/Library/Application Support/Tippi/` 下非 `Models/` 的內容。
- commit 訊息結尾加上:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01AQPo6RPGhw4KBtdoZbMVK4`

---

### Task 1: `Message.endMs`

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/Transcript.swift`(`Message` struct,約 line 47-71)
- Test: `apps/tippi/Tests/TippiCoreTests/TranscriptTests.swift`
- Test(連帶更新): `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Consumes: 既有 `SpeakerSegment`(已含 `endMs: UInt64`)。
- Produces: `Message` 新增 `public let endMs: UInt64`;memberwise init 簽名變為 `init(id:speaker:startMs:endMs:text:isFinal:)`;`init(id:segment:)` 帶入 `segment.endMs`。Task 2 的 exporter 依賴 `message.endMs`。

- [ ] **Step 1: 更新測試(先失敗)**

`apps/tippi/Tests/TippiCoreTests/TranscriptTests.swift` 中:

`messageIsBuiltFromSegment` 改為:

```swift
@Test
func messageIsBuiltFromSegment() {
    let segment = SpeakerSegment(speaker: 2, startMs: 80, endMs: 160, text: "喂?", isFinal: false)
    let message = Message(id: 5, segment: segment)
    #expect(message == Message(id: 5, speaker: 2, startMs: 80, endMs: 160, text: "喂?", isFinal: false))
}
```

`formatsLinesWithNamesAndDefaults` 開頭兩行改為:

```swift
    let named = Message(id: 0, speaker: 0, startMs: 204_000, endMs: 206_000, text: "今天先討論這個。", isFinal: true)
    let unnamed = Message(id: 1, speaker: 1, startMs: 6_132_000, endMs: 6_133_000, text: "好。", isFinal: true)
```

`apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift` 的 `recordingPublishesMessagesThenFinalizesOnStop` 中(約 line 106-108;該檔 `segment()` helper 的 `endMs` 為 `startMs + 80`):

```swift
    #expect(controller.messages == [
        Message(id: 0, speaker: 0, startMs: 400, endMs: 480, text: "今天先討論這個。", isFinal: false)
    ])
```

- [ ] **Step 2: 跑測試確認失敗(編譯錯誤)**

Run: `cargo build -p catcher-ffi --release && swift test --package-path apps/tippi`
Expected: 編譯失敗,`extra argument 'endMs' in call`(`Message` 尚無此參數)。

- [ ] **Step 3: 實作**

`apps/tippi/Sources/TippiCore/Transcript.swift` 的 `Message` 改為:

```swift
/// One row in Tippi's message list. `id` is the row's index; the whole list
/// is rebuilt from segments on every update.
public struct Message: Identifiable, Equatable, Sendable {
    public let id: Int
    public let speaker: Int
    public let startMs: UInt64
    public let endMs: UInt64
    public let text: String
    public let isFinal: Bool

    public init(id: Int, speaker: Int, startMs: UInt64, endMs: UInt64, text: String, isFinal: Bool) {
        self.id = id
        self.speaker = speaker
        self.startMs = startMs
        self.endMs = endMs
        self.text = text
        self.isFinal = isFinal
    }

    public init(id: Int, segment: SpeakerSegment) {
        self.init(
            id: id,
            speaker: segment.speaker,
            startMs: segment.startMs,
            endMs: segment.endMs,
            text: segment.text,
            isFinal: segment.isFinal
        )
    }
}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `swift test --package-path apps/tippi`
Expected: 全數 PASS(既有 18 個測試不減)。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/Transcript.swift apps/tippi/Tests/TippiCoreTests/TranscriptTests.swift apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "feat: carry segment end time through Message"
```

---

### Task 2: `TranscriptJSONExporter`

**Files:**
- Create: `apps/tippi/Sources/TippiCore/TranscriptJSONExporter.swift`
- Test: `apps/tippi/Tests/TippiCoreTests/TranscriptJSONExporterTests.swift`(新檔)

**Interfaces:**
- Consumes: Task 1 的 `Message`(含 `endMs`)、既有 `TranscriptFormatter.displayName(for:names:)`。
- Produces: `public enum TranscriptJSONExporter { public static func data(messages: [Message], names: [Int: String]) throws -> Data }`。Task 5 的匯出 UI 依賴。

- [ ] **Step 1: 寫失敗測試**

新檔 `apps/tippi/Tests/TippiCoreTests/TranscriptJSONExporterTests.swift`:

```swift
import Foundation
import Testing
@testable import TippiCore

@Test
func exportsMessagesAsSortedPrettyJSON() throws {
    let messages = [
        Message(id: 0, speaker: 0, startMs: 1234, endMs: 5678, text: "今天先討論這個。", isFinal: true),
        Message(id: 1, speaker: 1, startMs: 6000, endMs: 6400, text: "好。", isFinal: false),
    ]
    let data = try TranscriptJSONExporter.data(messages: messages, names: [0: "小明"])
    let expected = """
    {
      "messages" : [
        {
          "end_ms" : 5678,
          "final" : true,
          "name" : "小明",
          "speaker" : 0,
          "start_ms" : 1234,
          "text" : "今天先討論這個。"
        },
        {
          "end_ms" : 6400,
          "final" : false,
          "name" : "說話者 2",
          "speaker" : 1,
          "start_ms" : 6000,
          "text" : "好。"
        }
      ]
    }
    """
    #expect(String(decoding: data, as: UTF8.self) == expected)
}

@Test
func exportsEmptyMessageListAsEmptyArray() throws {
    let data = try TranscriptJSONExporter.data(messages: [], names: [:])
    let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    let messages = object?["messages"] as? [Any]
    #expect(messages?.isEmpty == true)
}
```

(golden 測試依賴 Apple 平台 `JSONEncoder` 的 `.prettyPrinted`+`.sortedKeys` 輸出格式——兩空格縮排、`" : "` 分隔;本專案僅 macOS,格式穩定。空列表用語意比對,避免對空陣列的 pretty 排版做 golden。)

- [ ] **Step 2: 跑測試確認失敗**

Run: `swift test --package-path apps/tippi`
Expected: 編譯失敗,`cannot find 'TranscriptJSONExporter' in scope`。

- [ ] **Step 3: 實作**

新檔 `apps/tippi/Sources/TippiCore/TranscriptJSONExporter.swift`:

```swift
import Foundation

/// Serialises the message list for the `.json` export choice. Keys are
/// snake_case to match the FFI segments contract; output is pretty-printed
/// with sorted keys so golden tests can compare byte-for-byte.
public enum TranscriptJSONExporter {
    private struct ExportMessage: Encodable {
        let speaker: Int
        let name: String
        let startMs: UInt64
        let endMs: UInt64
        let text: String
        let isFinal: Bool

        enum CodingKeys: String, CodingKey {
            case speaker
            case name
            case startMs = "start_ms"
            case endMs = "end_ms"
            case text
            case isFinal = "final"
        }
    }

    private struct ExportDocument: Encodable {
        let messages: [ExportMessage]
    }

    public static func data(messages: [Message], names: [Int: String]) throws -> Data {
        let document = ExportDocument(messages: messages.map { message in
            ExportMessage(
                speaker: message.speaker,
                name: TranscriptFormatter.displayName(for: message.speaker, names: names),
                startMs: message.startMs,
                endMs: message.endMs,
                text: message.text,
                isFinal: message.isFinal
            )
        })
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(document)
    }
}
```

- [ ] **Step 4: 跑測試確認通過**

Run: `swift test --package-path apps/tippi`
Expected: 全數 PASS,新增 2 個測試。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/TranscriptJSONExporter.swift apps/tippi/Tests/TippiCoreTests/TranscriptJSONExporterTests.swift
git commit -m "feat: add JSON transcript exporter"
```

---

### Task 3: `TranscriptionController.clearTranscript()`

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- Test: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Consumes: 既有 `state`/`messages`/`speakerNames`/`warningMessage` 與測試檔既有 `FakeCatcher`/`FakeAudio`/`FakeInstaller`/`segment()` helper。
- Produces: `public func clearTranscript()`——僅 `state == .ready` 時清空 `messages`、`speakerNames`、`warningMessage`,其餘狀態 no-op。Task 6 的清除按鈕依賴。

- [ ] **Step 1: 寫失敗測試**

`apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift` 檔尾新增(沿用該檔既有的 `FakeCatcher`/`FakeAudio`/`FakeInstaller`/`testBundle`/`segment()`):

```swift
@MainActor
@Test
func clearTranscriptClearsOnlyWhenReady() async throws {
    let catcher = FakeCatcher()
    await catcher.script(
        pushes: [TranscriptUpdate(
            segments: [segment(0, 400, "今天先討論這個。", final: false)],
            warning: "diarization disabled after a runtime error: injected"
        )],
        finish: TranscriptUpdate(
            segments: [segment(0, 400, "今天先討論這個。", final: true)],
            warning: "diarization disabled after a runtime error: injected"
        )
    )
    let audio = FakeAudio()
    let controller = TranscriptionController(
        modelInstaller: FakeInstaller(bundle: testBundle),
        audio: audio,
        catcherFactory: { _ in catcher }
    )
    await controller.prepare()

    await controller.toggleRecording()
    await audio.emit([0.1])
    try await Task.sleep(for: .milliseconds(20))
    controller.speakerNames[0] = "小明"

    // 錄音中呼叫是 no-op。
    controller.clearTranscript()
    #expect(!controller.messages.isEmpty)
    #expect(controller.speakerNames == [0: "小明"])

    await controller.toggleRecording()
    #expect(controller.state == .ready)
    #expect(controller.warningMessage != nil)

    controller.clearTranscript()
    #expect(controller.messages.isEmpty)
    #expect(controller.speakerNames.isEmpty)
    #expect(controller.warningMessage == nil)
    #expect(controller.state == .ready)
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `swift test --package-path apps/tippi`
Expected: 編譯失敗,`value of type 'TranscriptionController' has no member 'clearTranscript'`。

- [ ] **Step 3: 實作**

`apps/tippi/Sources/TippiCore/TranscriptionController.swift`,加在 `toggleRecording()` 之後:

```swift
    /// Wipes the finished transcript so the user can start over without
    /// recording. No-op unless idle: recording keeps its live transcript.
    public func clearTranscript() {
        guard state == .ready else { return }
        messages = []
        speakerNames = [:]
        warningMessage = nil
    }
```

- [ ] **Step 4: 跑測試確認通過**

Run: `swift test --package-path apps/tippi`
Expected: 全數 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/TranscriptionController.swift apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "feat: add clearTranscript to reset a finished session"
```

---

### Task 4: TabView 雙分頁外框

**Files:**
- Rename: `apps/tippi/Sources/TippiApp/ContentView.swift` → `apps/tippi/Sources/TippiApp/TranscriptionTabView.swift`(`git mv`,struct 同步改名)
- Create: `apps/tippi/Sources/TippiApp/ContentView.swift`(新內容,TabView 外框)
- Create: `apps/tippi/Sources/TippiApp/VoiceInputPlaceholderView.swift`

**Interfaces:**
- Consumes: 既有 `TranscriptionController`(`@Bindable` 注入)。
- Produces: `TranscriptionTabView`(即原 ContentView 全部內容,僅 struct 改名)、`VoiceInputPlaceholderView`、新 `ContentView`(TabView)。`TippiApp.swift` 不需改動(仍引用 `ContentView(controller:)`)。Task 5、6 修改的是 `TranscriptionTabView.swift`。

- [ ] **Step 1: 搬移並改名**

```bash
git mv apps/tippi/Sources/TippiApp/ContentView.swift apps/tippi/Sources/TippiApp/TranscriptionTabView.swift
```

`TranscriptionTabView.swift` 內把 `struct ContentView: View {` 改為 `struct TranscriptionTabView: View {`,其餘不動。

- [ ] **Step 2: 新增占位 view**

新檔 `apps/tippi/Sources/TippiApp/VoiceInputPlaceholderView.swift`:

```swift
import SwiftUI

struct VoiceInputPlaceholderView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "waveform")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text("語音輸入——即將推出")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
```

- [ ] **Step 3: 新的 ContentView(TabView)**

新檔 `apps/tippi/Sources/TippiApp/ContentView.swift`:

```swift
import SwiftUI
import TippiCore

struct ContentView: View {
    @Bindable var controller: TranscriptionController

    var body: some View {
        TabView {
            TranscriptionTabView(controller: controller)
                .tabItem { Label("轉錄", systemImage: "text.bubble") }
            VoiceInputPlaceholderView()
                .tabItem { Label("語音輸入", systemImage: "keyboard") }
        }
    }
}
```

- [ ] **Step 4: 建置與全測試**

Run: `swift build --package-path apps/tippi && swift test --package-path apps/tippi`
Expected: 建置成功、測試全 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiApp/
git commit -m "feat: split Tippi window into transcription and voice-input tabs"
```

---

### Task 5: 匯出雙格式 + 失敗 NSAlert

**Files:**
- Modify: `apps/tippi/Sources/TippiApp/TranscriptionTabView.swift`(原 `exportTranscript()`,原檔約 line 214-224)

**Interfaces:**
- Consumes: Task 2 `TranscriptJSONExporter.data(messages:names:)`、既有 `fullTranscript`。
- Produces: 無新介面;`exportTranscript()` 行為變更。

- [ ] **Step 1: 實作**

`TranscriptionTabView.swift` 中把 `exportTranscript()` 整個替換為:

```swift
    private func exportTranscript() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.plainText, .json]
        panel.nameFieldStringValue = "Tippi 逐字稿.txt"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try exportData(for: url).write(to: url)
        } catch {
            let alert = NSAlert()
            alert.alertStyle = .warning
            alert.messageText = "匯出失敗"
            alert.informativeText = error.localizedDescription
            alert.runModal()
        }
    }

    private func exportData(for url: URL) throws -> Data {
        if url.pathExtension.lowercased() == "json" {
            return try TranscriptJSONExporter.data(
                messages: controller.messages,
                names: controller.speakerNames
            )
        }
        return Data(fullTranscript.utf8)
    }
```

(檔案頂部已有 `import UniformTypeIdentifiers` 與 `import AppKit`,不需新增。)

- [ ] **Step 2: 建置與全測試**

Run: `swift build --package-path apps/tippi && swift test --package-path apps/tippi`
Expected: 建置成功、測試全 PASS(此 UI 分流無單元測試,格式產生邏輯已在 Task 2 覆蓋;實機驗收在 Task 7)。

- [ ] **Step 3: Commit**

```bash
git add apps/tippi/Sources/TippiApp/TranscriptionTabView.swift
git commit -m "feat: offer txt or json export and surface write failures"
```

---

### Task 6: 清除按鈕 + 命名 hover 教學提示

**Files:**
- Modify: `apps/tippi/Sources/TippiApp/TranscriptionTabView.swift`(footer)
- Modify: `apps/tippi/Sources/TippiApp/MessageRow.swift`(名字按鈕)

**Interfaces:**
- Consumes: Task 3 `controller.clearTranscript()`。
- Produces: 無新介面。

- [ ] **Step 1: footer 加「清除」按鈕與確認 alert**

`TranscriptionTabView.swift`:struct 內加狀態:

```swift
    @State private var isConfirmingClear = false
```

footer 的 `default:` 分支中,在 `Button("複製全部")` 之前插入:

```swift
                Button("清除") { isConfirmingClear = true }
                    .disabled(controller.state != .ready || controller.messages.isEmpty)
```

body 最外層 `VStack`(有 `.padding(36)` 那層)加上:

```swift
        .alert("清除全部訊息?", isPresented: $isConfirmingClear) {
            Button("清除", role: .destructive) { controller.clearTranscript() }
            Button("取消", role: .cancel) {}
        } message: {
            Text("將移除所有訊息與說話者命名,無法復原。")
        }
```

- [ ] **Step 2: MessageRow 名字按鈕加 tooltip 與 hover 底線**

`apps/tippi/Sources/TippiApp/MessageRow.swift`:struct 內加狀態:

```swift
    @State private var isHoveringName = false
```

名字按鈕的 label 改為(`Text(name)` 那行起):

```swift
                    Text(name)
                        .font(.callout.weight(.semibold))
                        .foregroundStyle(accent)
                        .underline(isHoveringName)
```

並在該 Button 的 `.popover(...)` 之後追加:

```swift
                .help("點擊可重新命名")
                .onHover { isHoveringName = $0 }
```

- [ ] **Step 3: 建置與全測試**

Run: `swift build --package-path apps/tippi && swift test --package-path apps/tippi`
Expected: 建置成功、測試全 PASS。

- [ ] **Step 4: Commit**

```bash
git add apps/tippi/Sources/TippiApp/TranscriptionTabView.swift apps/tippi/Sources/TippiApp/MessageRow.swift
git commit -m "feat: add clear-transcript button and rename affordance hint"
```

---

### Task 7: README 更新與全套驗證

**Files:**
- Modify: `README.md`(Tippi 段落:雙分頁、清除、JSON 匯出)

**Interfaces:**
- Consumes: Task 1-6 全部成果。
- Produces: 文件與驗證報告。

- [ ] **Step 1: 更新 README**

`README.md` 的 Tippi 功能描述段落,把訊息 UI 敘述擴充為(找到現有描述 speaker message list / copy / export 的段落,依實際文字融入,不重寫整份):

- 主視窗為雙分頁:「轉錄」與「語音輸入」(占位,後續子專案)。
- 匯出支援 .txt(行格式 `[mm:ss] 顯示名：內文`)與 .json(`messages[]`,snake_case key,`name` 為顯示名)。
- 「清除」按鈕在停止錄音後清空訊息、說話者命名與警告(附確認)。

- [ ] **Step 2: 全套驗證**

Run(repo 根):

```bash
cargo build -p catcher-ffi --release
swift build --package-path apps/tippi
swift test --package-path apps/tippi
cargo fmt --check
```

Expected: 全部通過;swift 測試數 = 既有 18 + 本計畫新增 3(exporter 2 + clearTranscript 1)= 21。

- [ ] **Step 3: 打包 app 供手動驗收**

Run(repo 根): `bash apps/tippi/scripts/build-app.sh`
Expected: 產出 `apps/tippi/build/Tippi.app`。

手動驗收清單(交給使用者,不阻擋 commit):分頁切換不中斷錄音、占位畫面文案、save panel 選 .txt/.json 與檔案內容、清除確認流程、匯出到唯讀位置跳「匯出失敗」alert、名字 hover 底線與 tooltip。

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document Tippi tabs, clear, and JSON export"
```

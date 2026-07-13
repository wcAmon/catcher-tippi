# Tippi 語音輸入與跨 App 文字注入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將「語音輸入」占位分頁改為可用功能：離線轉錄穩定文字並注入目前前景 App，偵測到 `Tippi Go` 後排除口令音訊並送出一次 Enter。

**Architecture:** 單一 `AudioRecorder` 依錄音模式將相同 16 kHz mono PCM 送到主 Catcher ASR，語音輸入模式再 fan out 到 sherpa-onnx KWS。Rust FFI 提供 append-only 平面文字、按命令起始時間截斷的 finish，以及獨立 KWS handle；Swift 的 `TextInjectionCoordinator` 管理 prefix diff、目標 App、安全送出與 CGEvent，`TranscriptionController` 統一管理模式互斥、模型與音訊生命週期。

**Tech Stack:** Rust 1.85 / edition 2024、`sherpa-onnx = 1.13.4`（預設 static feature）、C ABI、Swift 6.2、SwiftUI / Observation、ApplicationServices / CoreGraphics / AppKit、swift-testing、CryptoKit、macOS 15+。

## Global Constraints

- 正式設計以 `docs/superpowers/specs/2026-07-13-tippi-voice-input-injection-design.md` 為準。
- 固定口令顯示為 `Tippi Go`；KWS 詞條固定為 `T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO`，v1 不提供自訂口令或敏感度 UI。
- KWS 模型固定為 `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20` chunk-16：encoder epoch-13 int8、decoder epoch-13 fp32、joiner epoch-13 int8、`tokens.txt`。
- 所有音訊輸入都是 mono Float32 16 kHz；KWS `start_time` 轉為毫秒後，主 ASR 只保留 `token.frame * 80 < commandStartMs` 的 token。
- 送出順序必須是：flush/cutoff → 注入剩餘 suffix → Return key-down/key-up → reset ASR/KWS/coordinator；任何一步失敗都不得送 Enter。
- 注入只用 CGEvent Unicode；不得使用剪貼簿、AXValue、Backspace、游標移動或自動聚焦其他 App。
- `com.wcamon.tippi` 位於前景時不得注入或送出；命令在此前景狀態發生時捨棄該輪並顯示「請切到目標輸入框後重說 Tippi Go」，不得延遲到之後突然送出。
- 「轉錄」與「語音輸入」共用一個 `AudioRecorder`，一次只能有一個 `RecordingMode`；轉錄模式不得建立或 push KWS stream。
- 停止語音輸入只結束辨識，不補注入尚未完成文字，也不自動送出。
- 移除 App Sandbox；保留 microphone usage description，注入前必須通過 `AXIsProcessTrustedWithOptions`。
- Swift 測試前置：repo 根目錄先執行 `cargo build -p catcher-ffi --release`，讓 Swift Package 可連結 `target/release/libcatcher_ffi.dylib`。
- 不把 KWS ONNX 二進位或下載 archive 提交進 git；只提交小型測試 WAV、manifest、hash、口令檔內容與第三方授權說明。
- 每個 task 只 stage 自己列出的檔案；開始前與 commit 後都執行 `git status --short`，不得帶入使用者的其他變更。

## File Map

- Create `crates/catcher-ffi/src/kws.rs`: sherpa-onnx 安全 Rust wrapper 與固定模型設定。
- Create `crates/catcher-ffi/tests/kws_ffi.rs`: KWS C ABI null/lifecycle、正例與負例測試。
- Modify `crates/catcher-ffi/src/lib.rs`: `catcher_finish_before`、KWS C ABI 與錯誤邊界。
- Modify `crates/nemotron-mlx/src/fusion.rs`: 截斷 fusion token，使 segment 與平面文字一致。
- Modify兩份 `catcher.h`: 暴露 cutoff 與 KWS API。
- Create `apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift`: Swift actor 包裝 KWS C ABI。
- Create `apps/tippi/Sources/TippiCore/KeywordModelManifest.swift`: archive、檔名、大小與 SHA-256。
- Create `apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift`: 下載、解壓、驗證與原子安裝。
- Create `apps/tippi/Sources/TippiCore/ModelDirectoryMigrator.swift`: 從舊 sandbox Models 安全搬遷。
- Create `apps/tippi/Sources/TippiCore/TextInjector.swift`: Accessibility、前景 App 與 CGEvent 平台薄層。
- Create `apps/tippi/Sources/TippiCore/TextInjectionCoordinator.swift`: append-only diff、送出順序與 debounce。
- Modify `TranscriptionController.swift`, `TippiState.swift`, `Transcript.swift`, `CatcherClient.swift`: 雙模式資料流與可觀察狀態。
- Create `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`; delete placeholder；modify app composition and transcription tab.
- Modify packaging resources/scripts to remove sandbox and preserve sherpa-onnx third-party notice.

---

### Task 1: 主 ASR 平面文字與時間 cutoff

**Files:**
- Modify: `crates/nemotron-mlx/src/fusion.rs`
- Modify: `crates/nemotron-mlx/tests/fusion.rs`
- Modify: `crates/catcher-ffi/src/lib.rs`
- Modify: `crates/catcher-ffi/tests/ffi_lifecycle.rs`
- Modify: `crates/catcher-ffi/include/catcher.h`
- Modify: `apps/tippi/Sources/CCatcher/include/catcher.h`
- Modify: `apps/tippi/Sources/TippiCore/Transcript.swift`
- Modify: `apps/tippi/Sources/TippiCore/CatcherClient.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Produces Rust C ABI: `catcher_status_t catcher_finish_before(catcher_handle_t *, uint64_t cutoff_ms)`.
- Produces Swift: `TranscriptUpdate(text: String, segments: [SpeakerSegment], warning: String?)`.
- Extends `CatcherServing` with `func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate`.

- [ ] **Step 1: 寫 cutoff 與 flat text 失敗測試**

在 `crates/nemotron-mlx/tests/fusion.rs` 加入：

```rust
#[test]
fn retain_tokens_before_ms_removes_command_tail() {
    let mut fusion = Fusion::new(FusionConfig::default());
    fusion.push_tokens(&[
        TimedToken { id: 10, frame: 2 },
        TimedToken { id: 11, frame: 10 },
        TimedToken { id: 12, frame: 15 },
    ]);
    fusion.retain_tokens_before_ms(1_000);
    fusion.flush();
    let segments = fusion.segments(|ids| format!("{ids:?}"));
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "[10, 11]");
}
```

在 `crates/catcher-ffi/tests/ffi_lifecycle.rs` 的 null 測試加入：

```rust
assert_eq!(
    catcher_finish_before(ptr::null_mut(), 1_000),
    CATCHER_INVALID_ARGUMENT
);
```

所有 Swift 測試建立 `TranscriptUpdate` 時加入 `text:`；例如：

```swift
TranscriptUpdate(
    text: "今天先討論這個。",
    segments: [segment(0, 400, "今天先討論這個。", final: false)],
    warning: nil
)
```

- [ ] **Step 2: 跑測試確認失敗**

Run:

```bash
cargo test -p nemotron-mlx retain_tokens_before_ms
cargo test -p catcher-ffi null_arguments_report_errors_without_unwinding
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
```

Expected: Rust 先因 `retain_tokens_before_ms` / `catcher_finish_before` 不存在而編譯失敗；Swift 之後因 `TranscriptUpdate` 尚無 `text` 或 protocol 尚無 `finish(before:)` 失敗。

- [ ] **Step 3: 實作 token cutoff**

`Fusion` 增加：

```rust
pub fn retain_tokens_before_ms(&mut self, cutoff_ms: u64) {
    let frame_ms = self.cfg.frame_ms;
    self.tokens
        .retain(|token| token.frame.saturating_mul(frame_ms) < cutoff_ms);
}
```

在 `catcher-ffi` 以共用 helper 取代既有 `catcher_finish` 內文：

```rust
const ASR_FRAME_MS: u64 = 80;

fn finish_session(
    handle: &mut CatcherHandle,
    cutoff_ms: Option<u64>,
) -> Result<i32, (i32, String)> {
    if handle.state != SessionState::Started {
        return Err((CATCHER_INVALID_STATE, "catcher session is not recording".to_string()));
    }
    let tokens = handle
        .transcriber
        .finish()
        .map_err(|error| (CATCHER_RUNTIME_ERROR, error.to_string()))?;
    let has_new_tokens = !tokens.is_empty();
    if has_new_tokens {
        handle.fusion.push_tokens(&tokens);
        handle.timed_tokens.extend(tokens);
    }
    finish_diar(handle);
    if let Some(cutoff_ms) = cutoff_ms {
        handle
            .timed_tokens
            .retain(|token| token.frame.saturating_mul(ASR_FRAME_MS) < cutoff_ms);
        handle.fusion.retain_tokens_before_ms(cutoff_ms);
    }
    handle.fusion.flush();
    handle.state = SessionState::Finished;
    let segments_changed = rebuild_strings_and_report_segment_change(handle)?;
    Ok(if has_new_tokens || segments_changed { CATCHER_OK } else { CATCHER_NO_UPDATE })
}
```

兩個 ABI 入口都走 `with_handle_mut`：

```rust
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_finish(handle: *mut CatcherHandle) -> i32 {
    unsafe { with_handle_mut(handle, |handle| finish_session(handle, None)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_finish_before(
    handle: *mut CatcherHandle,
    cutoff_ms: u64,
) -> i32 {
    unsafe { with_handle_mut(handle, |handle| finish_session(handle, Some(cutoff_ms))) }
}
```

兩份 header 加入：

```c
catcher_status_t catcher_finish_before(catcher_handle_t *handle, uint64_t cutoff_ms);
```

- [ ] **Step 4: 實作 Swift flat text contract**

`TranscriptUpdate` 改為：

```swift
public struct TranscriptUpdate: Equatable, Sendable {
    public let text: String
    public let segments: [SpeakerSegment]
    public let warning: String?

    public init(text: String, segments: [SpeakerSegment], warning: String?) {
        self.text = text
        self.segments = segments
        self.warning = warning
    }
}
```

`CatcherServing` 與 `CatcherClient` 增加：

```swift
func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate
```

```swift
public func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate {
    try check(catcher_finish_before(owner.pointer, cutoffMs), allowNoUpdate: true)
    return try currentUpdate()
}
```

`currentUpdate()` 第一行讀取：

```swift
let text = catcher_text(owner.pointer).map { String(cString: $0) } ?? ""
```

回傳：

```swift
return TranscriptUpdate(text: text, segments: segments, warning: warning)
```

- [ ] **Step 5: 跑 formatter 與完整測試**

Run:

```bash
cargo fmt --all -- --check
cargo test -p nemotron-mlx
cargo test -p catcher-ffi
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
```

Expected: 全數 PASS；既有 ignored real-model tests 維持 ignored。

- [ ] **Step 6: Commit**

```bash
git add crates/nemotron-mlx/src/fusion.rs crates/nemotron-mlx/tests/fusion.rs crates/catcher-ffi/src/lib.rs crates/catcher-ffi/tests/ffi_lifecycle.rs crates/catcher-ffi/include/catcher.h apps/tippi/Sources/CCatcher/include/catcher.h apps/tippi/Sources/TippiCore/Transcript.swift apps/tippi/Sources/TippiCore/CatcherClient.swift apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "feat: expose timed transcript cutoff"
```

---

### Task 2: sherpa-onnx KWS runtime 與 C ABI

**Files:**
- Modify: `crates/catcher-ffi/Cargo.toml`
- Modify: `Cargo.lock`
- Create: `crates/catcher-ffi/src/kws.rs`
- Modify: `crates/catcher-ffi/src/lib.rs`
- Create: `crates/catcher-ffi/tests/kws_ffi.rs`
- Modify: `crates/catcher-ffi/include/catcher.h`
- Modify: `apps/tippi/Sources/CCatcher/include/catcher.h`
- Create: `tests/fixtures/tippi-go.wav`
- Modify: `tests/fixtures/README.md`

**Interfaces:**
- `catcher_kws_create(model_directory)`, `catcher_kws_start`, `catcher_kws_push_audio`, `catcher_kws_keyword`, `catcher_kws_start_ms`, `catcher_kws_last_error`, `catcher_kws_destroy`.
- `CATCHER_COMMAND_DETECTED = 2`; push 回傳 2 時 keyword 必為 `TIPPI_GO` 且 start ms 可讀。

- [ ] **Step 1: 產生固定正例 fixture**

Run:

```bash
/usr/bin/say -v Samantha -r 170 -o /tmp/tippi-go.aiff "Tippy go"
/usr/bin/afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/tippi-go.aiff tests/fixtures/tippi-go.wav
file tests/fixtures/tippi-go.wav
```

Expected: `Microsoft PCM, 16 bit, mono 16000 Hz`。測試載入後會自行加前後各一秒 silence，保留 KWS 所需左側 context。

- [ ] **Step 2: 寫 KWS 失敗測試**

`kws_ffi.rs` 包含：

```rust
#[test]
fn null_kws_handle_is_safe() {
    unsafe {
        assert_eq!(catcher_kws_start(ptr::null_mut()), CATCHER_INVALID_ARGUMENT);
        assert_eq!(
            catcher_kws_push_audio(ptr::null_mut(), ptr::null(), 0),
            CATCHER_INVALID_ARGUMENT
        );
        assert!(catcher_kws_keyword(ptr::null()).is_null());
        catcher_kws_destroy(ptr::null_mut());
    }
}

#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn detects_padded_tippi_go_once_and_resets() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null());
    let spoken = read_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tippi-go.wav"
    ));
    let mut samples = vec![0.0; 16_000];
    samples.extend(spoken);
    samples.extend(vec![0.0; 16_000]);

    unsafe {
        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        let detected = feed_until_detected(handle, &samples);
        assert!(detected);
        assert_eq!(CStr::from_ptr(catcher_kws_keyword(handle)).to_str().unwrap(), "TIPPI_GO");
        assert!((800..=2_000).contains(&catcher_kws_start_ms(handle)));
        assert_eq!(
            catcher_kws_push_audio(handle, ptr::null(), 0),
            CATCHER_NO_UPDATE
        );
        assert_eq!(catcher_kws_start(handle), CATCHER_OK);
        assert!(feed_until_detected(handle, &samples));
        catcher_kws_destroy(handle);
    }
}
```

同檔另以 `hello-streaming.wav`、`conversation.wav` 前半與後半三組音訊斷言不回傳 `CATCHER_COMMAND_DETECTED`。

- [ ] **Step 3: 跑測試確認失敗**

Run: `cargo test -p catcher-ffi --test kws_ffi`

Expected: 編譯失敗，KWS 常數與函式尚不存在。

- [ ] **Step 4: 加入依賴與安全 wrapper**

`crates/catcher-ffi/Cargo.toml`：

```toml
sherpa-onnx = "1.13.4"
```

`kws.rs` 的核心型別：

```rust
use std::path::{Path, PathBuf};
use sherpa_onnx::{KeywordSpotter, KeywordSpotterConfig, OnlineStream};

pub const EXPECTED_KEYWORD: &str = "TIPPI_GO";

pub struct KeywordDetection {
    pub keyword: String,
    pub start_ms: u64,
}

pub struct KeywordSpotterSession {
    spotter: KeywordSpotter,
    stream: OnlineStream,
    latched: bool,
}

impl KeywordSpotterSession {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let mut config = KeywordSpotterConfig::default();
        config.model_config.transducer.encoder =
            Some(path(directory, "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx")?);
        config.model_config.transducer.decoder =
            Some(path(directory, "decoder-epoch-13-avg-2-chunk-16-left-64.onnx")?);
        config.model_config.transducer.joiner =
            Some(path(directory, "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx")?);
        config.model_config.tokens = Some(path(directory, "tokens.txt")?);
        config.model_config.modeling_unit = Some("cjkchar".to_string());
        config.model_config.num_threads = 1;
        config.keywords_file = Some(path(directory, "keywords.txt")?);
        let spotter = KeywordSpotter::create(&config)
            .ok_or_else(|| "sherpa-onnx could not create keyword spotter".to_string())?;
        let stream = spotter.create_stream();
        Ok(Self { spotter, stream, latched: false })
    }

    pub fn reset(&mut self) {
        self.spotter.reset(&self.stream);
        self.latched = false;
    }

    pub fn push(&mut self, samples: &[f32]) -> Option<KeywordDetection> {
        if self.latched { return None; }
        self.stream.accept_waveform(16_000, samples);
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
        }
        let result = self.spotter.get_result(&self.stream)?;
        if result.keyword.is_empty() { return None; }
        self.latched = true;
        Some(KeywordDetection {
            keyword: result.keyword,
            start_ms: (result.start_time.max(0.0) * 1_000.0).round() as u64,
        })
    }
}

fn path(directory: &Path, name: &str) -> Result<String, String> {
    let path: PathBuf = directory.join(name);
    if !path.is_file() {
        return Err(format!("missing KWS model file: {}", path.display()));
    }
    Ok(path.to_string_lossy().into_owned())
}
```

- [ ] **Step 5: 加入 KWS C ABI**

`lib.rs` 宣告 `mod kws;`，增加 `KwsHandle`（session、keyword CString、start ms、last error），以既有 `catch_unwind` / safe CString 慣例實作以下簽名：

```rust
pub const CATCHER_COMMAND_DETECTED: i32 = 2;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_create(
    model_directory: *const c_char,
) -> *mut KwsHandle;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_start(handle: *mut KwsHandle) -> i32;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_push_audio(
    handle: *mut KwsHandle,
    samples: *const f32,
    count: usize,
) -> i32;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_keyword(handle: *const KwsHandle) -> *const c_char;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_start_ms(handle: *const KwsHandle) -> u64;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_last_error(handle: *const KwsHandle) -> *const c_char;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_kws_destroy(handle: *mut KwsHandle);
```

`catcher_kws_push_audio` 只有 `KeywordDetection` 出現時更新兩個 borrowed accessors 並回傳 2；latched 後回傳 `CATCHER_NO_UPDATE`，直到 `catcher_kws_start` reset。

兩份 header 同步 typedef、常數與完整 prototypes：

```c
typedef struct catcher_kws_handle catcher_kws_handle_t;
enum { CATCHER_COMMAND_DETECTED = 2 };
```

- [ ] **Step 6: 執行 unit 與 real-model 測試**

先準備 pinned real model：

```bash
KWS_ARCHIVE=/tmp/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
KWS_ROOT=/tmp/tippi-kws-model-check
curl -fL --retry 2 -o "${KWS_ARCHIVE}" https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
test "$(shasum -a 256 "${KWS_ARCHIVE}" | cut -d ' ' -f 1)" = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
rm -rf "${KWS_ROOT}"
mkdir -p "${KWS_ROOT}"
tar -xjf "${KWS_ARCHIVE}" -C "${KWS_ROOT}"
```

再執行：

```bash
cargo fmt --all -- --check
cargo test -p catcher-ffi --test kws_ffi
SHERPA_KWS_MODEL=/tmp/tippi-kws-model-check/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 cargo test -p catcher-ffi --test kws_ffi -- --ignored --nocapture
```

Expected: null/negative unit 測試 PASS；ignored 正例在指定模型下偵測兩輪 `TIPPI_GO`，start ms 落在 800–2000。

- [ ] **Step 7: Commit**

```bash
git add Cargo.lock crates/catcher-ffi/Cargo.toml crates/catcher-ffi/src/kws.rs crates/catcher-ffi/src/lib.rs crates/catcher-ffi/tests/kws_ffi.rs crates/catcher-ffi/include/catcher.h apps/tippi/Sources/CCatcher/include/catcher.h tests/fixtures/tippi-go.wav tests/fixtures/README.md
git commit -m "feat: add offline Tippi Go detector"
```

---

### Task 3: Swift KWS client

**Files:**
- Create: `apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/KeywordSpotterClientTests.swift`

**Interfaces:**
- Produces `KeywordDetection(keyword: String, startMs: UInt64)`.
- Produces protocol `KeywordSpotting` with `start()`, `push(_:)`, `reset()`.
- Produces factory typealias `KeywordSpotterFactory`.

- [ ] **Step 1: 寫 protocol contract 測試**

```swift
import Foundation
import Testing
@testable import TippiCore

private actor FakeKeywordSpotter: KeywordSpotting {
    func start() async throws {}
    func push(_ samples: [Float]) async throws -> KeywordDetection? {
        KeywordDetection(keyword: "TIPPI_GO", startMs: UInt64(samples.count))
    }
    func reset() async throws {}
}

@Test
func keywordDetectionCarriesCommandAndCutoff() async throws {
    let service: any KeywordSpotting = FakeKeywordSpotter()
    let detection = try await service.push([0, 0, 0])
    #expect(detection == KeywordDetection(keyword: "TIPPI_GO", startMs: 3))
}
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `cargo build -p catcher-ffi --release && swift test --package-path apps/tippi --filter KeywordSpotterClientTests`

Expected: 編譯失敗，Swift KWS 型別尚不存在。

- [ ] **Step 3: 實作 actor wrapper**

```swift
import CCatcher
import Foundation

public struct KeywordDetection: Equatable, Sendable {
    public let keyword: String
    public let startMs: UInt64
}

public protocol KeywordSpotting: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> KeywordDetection?
    func reset() async throws
}

public typealias KeywordSpotterFactory =
    @Sendable (URL) async throws -> any KeywordSpotting

public enum KeywordSpotterClientError: Error, LocalizedError {
    case creationFailed(String)
    case operationFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .creationFailed(message):
            "Could not load Tippi Go detector: \(message)"
        case let .operationFailed(message):
            "Tippi Go detector failed: \(message)"
        }
    }
}

private final class KeywordHandleOwner: @unchecked Sendable {
    let pointer: OpaquePointer
    init(pointer: OpaquePointer) { self.pointer = pointer }
    deinit { catcher_kws_destroy(pointer) }
}

public actor KeywordSpotterClient: KeywordSpotting {
    private let owner: KeywordHandleOwner

    public init(modelDirectory: URL) throws {
        let pointer = modelDirectory.path.withCString { catcher_kws_create($0) }
        guard let pointer else {
            throw KeywordSpotterClientError.creationFailed(Self.globalError())
        }
        owner = KeywordHandleOwner(pointer: pointer)
    }

    public func start() async throws {
        try check(catcher_kws_start(owner.pointer))
    }
    public func reset() async throws {
        try check(catcher_kws_start(owner.pointer))
    }

    public func push(_ samples: [Float]) async throws -> KeywordDetection? {
        let status = samples.withUnsafeBufferPointer {
            catcher_kws_push_audio(owner.pointer, $0.baseAddress, $0.count)
        }
        if status == CATCHER_NO_UPDATE { return nil }
        guard status == CATCHER_COMMAND_DETECTED else {
            throw KeywordSpotterClientError.operationFailed(currentError())
        }
        guard let keyword = catcher_kws_keyword(owner.pointer) else {
            throw KeywordSpotterClientError.operationFailed(
                "detected a command without a keyword"
            )
        }
        return KeywordDetection(
            keyword: String(cString: keyword),
            startMs: catcher_kws_start_ms(owner.pointer)
        )
    }

    private func check(_ status: Int32) throws {
        guard status == CATCHER_OK else {
            throw KeywordSpotterClientError.operationFailed(currentError())
        }
    }

    private func currentError() -> String {
        guard let pointer = catcher_kws_last_error(owner.pointer) else {
            return "unknown KWS error"
        }
        return String(cString: pointer)
    }

    private static func globalError() -> String {
        guard let pointer = catcher_kws_last_error(nil) else {
            return "unknown KWS error"
        }
        return String(cString: pointer)
    }
}
```

- [ ] **Step 4: 跑測試**

Run: `cargo build -p catcher-ffi --release && swift test --package-path apps/tippi`

Expected: 全數 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/KeywordSpotterClient.swift apps/tippi/Tests/TippiCoreTests/KeywordSpotterClientTests.swift
git commit -m "feat: bridge keyword spotting into Swift"
```

---

### Task 4: KWS 模型下載、驗證與原子安裝

**Files:**
- Create: `apps/tippi/Sources/TippiCore/ModelChecksum.swift`
- Modify: `apps/tippi/Sources/TippiCore/ModelStore.swift`
- Create: `apps/tippi/Sources/TippiCore/KeywordModelManifest.swift`
- Create: `apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/ModelStoreTests.swift`

**Interfaces:**
- `KeywordModelInstalling.installIfNeeded(progress:) async throws -> URL`.
- `KeywordModelManifest.release` 固定 archive URL/hash 與四個 runtime files。
- `ArchiveExtracting` 讓 unit test 不執行真正 `tar`。

- [ ] **Step 1: 寫 installer 失敗測試**

用 `FakeDownloader` 寫 archive payload、`FakeArchiveExtractor` 在 unpack 目錄產生指定檔案，覆蓋四個案例：

```swift
@Test
func installsOnlyVerifiedRuntimeFilesAndGeneratedMetadata() async throws

@Test
func archiveHashMismatchCleansEveryStagingPath() async throws

@Test
func extractedFileHashMismatchDoesNotReplaceExistingInstall() async throws

@Test
func verifiedExistingInstallSkipsDownloadAndExtraction() async throws
```

第一個案例需斷言目的目錄只有：

```swift
[
    "THIRD_PARTY_NOTICES.md",
    "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "keywords.txt",
    "tokens.txt",
]
```

並斷言 `keywords.txt` 位元組等於：

```text
T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `swift test --package-path apps/tippi --filter KeywordModelInstallerTests`

Expected: 編譯失敗，installer/manifest 尚不存在。

- [ ] **Step 3: 抽出共用 checksum**

新增：

```swift
import CryptoKit
import Foundation

enum ModelChecksum {
    static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            digest.update(data: data)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
```

`ModelStore` 的驗證改呼叫 `ModelChecksum.sha256(of:)`，刪除原本 private helper，既有 ModelStore 測試必須保持通過。

- [ ] **Step 4: 實作 pinned manifest**

```swift
public struct KeywordModelArchive: Sendable {
    public let url: URL
    public let sha256: String
    public let byteCount: Int64
    public let directoryName: String
    public let files: [ModelFile]
}

public enum KeywordModelManifest {
    public static let keywords =
        "T IH1 P IY0 G OW1 :1.5 #0.25 @TIPPI_GO\n"

    public static let release = KeywordModelArchive(
        url: URL(string: "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2")!,
        sha256: "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6",
        byteCount: 32_885_699,
        directoryName: "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
        files: [
            ModelFile(name: "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx", sha256: "408bbd740838c42d5bf6d1c5b80b3c88b616c7860b92d980328b5b068c76ae48", required: true, byteCount: 4_599_656),
            ModelFile(name: "decoder-epoch-13-avg-2-chunk-16-left-64.onnx", sha256: "63a22dd60f40fff082ac3e09afa507f6787da36df76ded2fbe145fa233e22c21", required: true, byteCount: 759_829),
            ModelFile(name: "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx", sha256: "190d4067b4cc20b72a42a1916e69d92052000fb7051a427ebb1bc72a69207dc1", required: true, byteCount: 86_629),
            ModelFile(name: "tokens.txt", sha256: "2d3f32311f9b692b964da3c90e830258d3e78e013cb0c992dbfb15cd5a1a71b0", required: true, byteCount: 1_928),
        ]
    )
}
```

- [ ] **Step 5: 實作 tar extractor 與原子 installer**

定義：

```swift
public protocol ArchiveExtracting: Sendable {
    func extract(archive: URL, to directory: URL) async throws
}

public protocol KeywordModelInstalling: Sendable {
    func installIfNeeded(
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> URL
}
```

`TarArchiveExtractor` 在 `Task.detached` 執行 `/usr/bin/tar -xjf <archive> -C <directory>`，擷取 stderr，非零 exit 轉成 `KeywordModelInstallerError.extractionFailed(String)`。

`KeywordModelInstaller.installIfNeeded` 使用同一 root 下的：

```text
.<directoryName>.archive.partial
.<directoryName>.unpack.partial
.<directoryName>.install.partial
```

流程固定為 download → archive SHA-256 → extract → 四檔 SHA-256 → 寫 `keywords.txt` 與 `THIRD_PARTY_NOTICES.md` → verify staging → 移除無效舊目的地 → move install staging 至正式目錄。catch 區塊移除三個 partial 路徑，已驗證的既有目的地不得先刪除。

第三方 notice 內容固定：

```text
sherpa-onnx and the sherpa-onnx KWS model
Copyright (c) k2-fsa contributors
Licensed under the Apache License, Version 2.0.
Source: https://github.com/k2-fsa/sherpa-onnx
```

Production actor 的 initializer 固定為：

```swift
public actor KeywordModelInstaller: KeywordModelInstalling {
    public init(
        rootDirectory: URL? = nil,
        manifest: KeywordModelArchive = KeywordModelManifest.release,
        downloader: any ModelDownloading = URLSessionModelDownloader(),
        extractor: any ArchiveExtracting = TarArchiveExtractor(),
        fileManager: FileManager = .default
    ) {
        self.rootDirectory =
            rootDirectory ?? ModelStore.defaultRootDirectory(fileManager: fileManager)
        self.manifest = manifest
        self.downloader = downloader
        self.extractor = extractor
        self.fileManager = fileManager
    }
}
```

`TarArchiveExtractor` 提供 `public init() {}`，使 `TippiApp` 可直接建立 production installer。

- [ ] **Step 6: 跑 installer 與既有 store 測試**

Run:

```bash
swift test --package-path apps/tippi --filter KeywordModelInstallerTests
swift test --package-path apps/tippi --filter ModelStoreTests
```

Expected: 所有案例 PASS；checksum mismatch 後無 partial 路徑。

- [ ] **Step 7: Commit**

```bash
git add apps/tippi/Sources/TippiCore/ModelChecksum.swift apps/tippi/Sources/TippiCore/ModelStore.swift apps/tippi/Sources/TippiCore/KeywordModelManifest.swift apps/tippi/Sources/TippiCore/KeywordModelInstaller.swift apps/tippi/Tests/TippiCoreTests/KeywordModelInstallerTests.swift apps/tippi/Tests/TippiCoreTests/ModelStoreTests.swift
git commit -m "feat: install verified keyword model"
```

---

### Task 5: 舊 sandbox 模型安全遷移

**Files:**
- Create: `apps/tippi/Sources/TippiCore/ModelDirectoryMigrator.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/ModelDirectoryMigratorTests.swift`
- Modify: `apps/tippi/Sources/TippiCore/ModelStore.swift`

**Interfaces:**
- `ModelDirectoryMigrating.migrateIfNeeded() async throws`.
- `ModelDirectoryMigrator` 預設 source 是舊 container `Models`，destination 是非 sandbox `Models`。

- [ ] **Step 1: 寫四個遷移失敗測試**

```swift
@Test func movesLegacyModelsWhenDestinationIsAbsent() async throws
@Test func nonEmptyDestinationIsNeverOverwritten() async throws
@Test func moveFailureCopiesThroughStagingAndRemovesSourceOnlyAfterPromotion() async throws
@Test func copyFailureKeepsTheOnlyLegacyModelCopy() async throws
```

copy fallback 測試注入永遠 throw 的 `moveItem` closure；copy failure 測試在 source 內建立無讀權限項目，並在 defer 恢復權限。

- [ ] **Step 2: 跑測試確認失敗**

Run: `swift test --package-path apps/tippi --filter ModelDirectoryMigratorTests`

Expected: 編譯失敗，migrator 尚不存在。

- [ ] **Step 3: 實作 migrator**

```swift
public protocol ModelDirectoryMigrating: Sendable {
    func migrateIfNeeded() async throws
}

public actor ModelDirectoryMigrator: ModelDirectoryMigrating {
    public typealias MoveItem = @Sendable (URL, URL) throws -> Void

    private let source: URL
    private let destination: URL
    private let fileManager: FileManager
    private let moveItem: MoveItem

    public init(
        source: URL = Self.defaultLegacyRoot(),
        destination: URL = ModelStore.defaultRootDirectory(),
        fileManager: FileManager = .default,
        moveItem: @escaping MoveItem = {
            try FileManager.default.moveItem(at: $0, to: $1)
        }
    ) {
        self.source = source
        self.destination = destination
        self.fileManager = fileManager
        self.moveItem = moveItem
    }

    private static func defaultLegacyRoot(
        fileManager: FileManager = .default
    ) -> URL {
        fileManager.homeDirectoryForCurrentUser
            .appending(path: "Library/Containers/com.wcamon.tippi/Data/Library/Application Support/Tippi/Models", directoryHint: .isDirectory)
    }
}
```

`migrateIfNeeded`：

1. source 不存在則 return。
2. destination 存在且內容非空則 return。
3. 建立 destination parent，移除空 destination。
4. 先 `moveItem(source, destination)`。
5. move 失敗時 copy source 到 sibling `.Models.migration.partial`；對每個 regular file 比較 relative path、file size 與 `ModelChecksum.sha256`，驗證後 move staging 至 destination，最後才 remove source。
6. 任一 copy/verify/promote 失敗都清 staging 且保留 source。

`ModelStore.defaultRootDirectory(fileManager:)` 從 private 改為 public static，預設參數 `.default`。

- [ ] **Step 4: 跑遷移與完整 Swift 測試**

Run: `swift test --package-path apps/tippi`

Expected: 四個新案例與全部既有案例 PASS。

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/ModelDirectoryMigrator.swift apps/tippi/Sources/TippiCore/ModelStore.swift apps/tippi/Tests/TippiCoreTests/ModelDirectoryMigratorTests.swift
git commit -m "feat: migrate legacy Tippi models"
```

---

### Task 6: Accessibility、CGEvent 與注入 coordinator

**Files:**
- Create: `apps/tippi/Sources/TippiCore/TextInjector.swift`
- Create: `apps/tippi/Sources/TippiCore/TextInjectionCoordinator.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/TextInjectionCoordinatorTests.swift`

**Interfaces:**
- `@MainActor TextInjecting`: trust check/request、`inject(_:)`、`submit()`。
- `@MainActor FrontmostApplicationProviding`.
- `TextInjectionCoordinator.consume(_:)` 與 `submit(_:)`。

- [ ] **Step 1: 寫 coordinator 失敗測試**

用 fake injector/target provider 覆蓋：

```swift
@Test func appendOnlyUpdatesInjectOnlyNewSuffix() throws
@Test func divergentPrefixThrowsWithoutBackspaceOrSubmit() throws
@Test func submitInjectsRemainingSuffixBeforeReturn() throws
@Test func duplicateSubmitIsIgnoredUntilReset() throws
@Test func tippiFrontmostDoesNotInjectOrSubmit() throws
@Test func unicodeTextIsPassedWithoutClipboardTransformation() throws
```

核心 assertion：

```swift
#expect(injector.events == [
    .text("你好"),
    .text("，世界"),
    .returnKey,
])
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `swift test --package-path apps/tippi --filter TextInjectionCoordinatorTests`

Expected: 編譯失敗，protocol/coordinator 尚不存在。

- [ ] **Step 3: 實作平台介面**

```swift
public struct TargetApplication: Equatable, Sendable {
    public let name: String
    public let bundleIdentifier: String?
}

@MainActor
public protocol FrontmostApplicationProviding: AnyObject {
    func current() -> TargetApplication?
}

@MainActor
public protocol TextInjecting: AnyObject {
    func isTrusted(prompt: Bool) -> Bool
    func inject(_ text: String) throws
    func submit() throws
}
```

`CGEventTextInjector.isTrusted(prompt:)` 呼叫：

```swift
let options = [
    kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: prompt
] as CFDictionary
return AXIsProcessTrustedWithOptions(options)
```

`inject(_:)` 將 `Array(text.utf16)` 設到 virtual key 0 的 key-down / key-up：

```swift
units.withUnsafeBufferPointer { buffer in
    down.keyboardSetUnicodeString(
        stringLength: buffer.count,
        unicodeString: buffer.baseAddress
    )
    up.keyboardSetUnicodeString(
        stringLength: buffer.count,
        unicodeString: buffer.baseAddress
    )
}
down.post(tap: .cghidEventTap)
up.post(tap: .cghidEventTap)
```

`submit()` 使用 key code 36，依序 post down/up。`FrontmostApplicationProvider` 從 `NSWorkspace.shared.frontmostApplication` 取 localizedName 與 bundleIdentifier。

- [ ] **Step 4: 實作 coordinator**

```swift
public enum TextInjectionEvent: Equatable {
    case noChange
    case waitingForTarget
    case injected(text: String, target: String)
    case submitted(text: String, target: String)
    case duplicateCommandIgnored
}

public enum TextInjectionError: Error, LocalizedError, Equatable {
    case divergentPrefix(previous: String, current: String)
    case eventCreationFailed
}
```

`consume` 必須先檢查非 Tippi 目標，再驗證 `fullText.hasPrefix(injectedPrefix)`；只有成功 post 後才更新 prefix。`submit` 先檢查 `commandInFlight`，再取得非 Tippi 目標、呼叫相同 suffix helper，接著 `injector.submit()`，最後保持 latch 為 true；controller 完成模型 reset 後呼叫：

```swift
public func resetTurn() {
    injectedPrefix = ""
    commandInFlight = false
}
```

同時提供：

```swift
public func isTrusted(prompt: Bool) -> Bool
public func currentTarget() -> TargetApplication?
```

Coordinator 的 production signature 固定為：

```swift
@MainActor
public final class TextInjectionCoordinator {
    public init(
        injector: any TextInjecting,
        targetProvider: any FrontmostApplicationProviding,
        ownBundleIdentifier: String
    ) {
        self.injector = injector
        self.targetProvider = targetProvider
        self.ownBundleIdentifier = ownBundleIdentifier
    }
}
```

`CGEventTextInjector` 與 `FrontmostApplicationProvider` 都提供 `public init() {}`。

- [ ] **Step 5: 跑測試**

Run: `swift test --package-path apps/tippi --filter TextInjectionCoordinatorTests`

Expected: 六個案例 PASS；fake event list 不含 delete、paste 或第二次 return。

- [ ] **Step 6: Commit**

```bash
git add apps/tippi/Sources/TippiCore/TextInjector.swift apps/tippi/Sources/TippiCore/TextInjectionCoordinator.swift apps/tippi/Tests/TippiCoreTests/TextInjectionCoordinatorTests.swift
git commit -m "feat: add safe cross-app text injection"
```

---

### Task 7: Controller 雙模式與音訊 fan-out

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/TippiState.swift`
- Modify: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- `RecordingMode.transcription / .voiceInput`.
- `VoiceInputPreparationState.notPrepared / downloading / loading / ready / failed`.
- `toggleRecording(mode:)`, `prepareVoiceInput()`, permission refresh/request。

- [ ] **Step 1: 擴充 fakes 並寫失敗測試**

`FakeCatcher` 加入 `finish(before:)`，`FakeKeywordSpotter` 可 script detection/error，並測：

```swift
@Test func transcriptionModeNeverStartsOrPushesKeywordSpotter() async throws
@Test func voiceModeFansEachAudioChunkToAsrAndKws() async throws
@Test func commandCutsOffInjectsSubmitsAndResetsInOrder() async throws
@Test func duplicateCommandDoesNotSubmitTwice() async throws
@Test func voiceStopDoesNotInjectOrSubmitTrailingText() async throws
@Test func kwsFailureStopsAudioAndFailsClosed() async throws
@Test func otherTabsRecordingButtonIsDisabledByActiveMode() async throws
@Test func prepareRunsLegacyMigrationBeforeModelInstaller() async throws
```

命令順序 test 的共享 event log 必須等於：

```swift
[
    "asr.start",
    "kws.start",
    "audio.start",
    "asr.push",
    "kws.push",
    "asr.finishBefore:960",
    "inject:你好",
    "submit",
    "asr.start",
    "kws.reset",
]
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `swift test --package-path apps/tippi --filter TranscriptionControllerTests`

Expected: 新模式、KWS dependency 與新方法不存在而失敗。

- [ ] **Step 3: 定義狀態與 dependencies**

`TippiState.swift` 增加：

```swift
public enum RecordingMode: Equatable, Sendable {
    case transcription
    case voiceInput
}

public enum VoiceInputPreparationState: Equatable, Sendable {
    case notPrepared
    case downloading(Double)
    case loading
    case ready
    case failed(String)
}
```

Controller 新增 observable properties：

```swift
public private(set) var activeMode: RecordingMode?
public private(set) var voiceInputPreparation: VoiceInputPreparationState = .notPrepared
public private(set) var accessibilityTrusted = false
public private(set) var targetApplicationName: String?
public private(set) var lastInjectedText = ""
public private(set) var voiceInputMessage = "請先切到目標輸入框"
```

及 dependencies：

```swift
@ObservationIgnored private let modelMigrator: any ModelDirectoryMigrating
@ObservationIgnored private let keywordInstaller: any KeywordModelInstalling
@ObservationIgnored private let keywordFactory: KeywordSpotterFactory
@ObservationIgnored private let injectionCoordinator: TextInjectionCoordinator
@ObservationIgnored private var keywordSpotter: (any KeywordSpotting)?
```

Controller initializer 固定為：

```swift
public init(
    modelInstaller: any ModelBundleInstalling,
    audio: any AudioRecording,
    catcherFactory: @escaping CatcherFactory,
    modelMigrator: any ModelDirectoryMigrating,
    keywordInstaller: any KeywordModelInstalling,
    keywordFactory: @escaping KeywordSpotterFactory,
    injectionCoordinator: TextInjectionCoordinator
) {
    self.modelInstaller = modelInstaller
    self.audio = audio
    self.catcherFactory = catcherFactory
    self.modelMigrator = modelMigrator
    self.keywordInstaller = keywordInstaller
    self.keywordFactory = keywordFactory
    self.injectionCoordinator = injectionCoordinator
}
```

既有 unit tests 全部提供明確 fake dependency，不使用 production singleton。

- [ ] **Step 4: 實作準備與權限**

`prepare()` 的 do 區塊第一行：

```swift
try await modelMigrator.migrateIfNeeded()
```

`prepareVoiceInput()`：

```swift
public func prepareVoiceInput() async {
    switch voiceInputPreparation {
    case .notPrepared, .failed:
        break
    default:
        return
    }
    voiceInputPreparation = .downloading(0)
    do {
        try await modelMigrator.migrateIfNeeded()
        let directory = try await keywordInstaller.installIfNeeded { [weak self] value in
            Task { @MainActor in
                self?.voiceInputPreparation = .downloading(value)
            }
        }
        voiceInputPreparation = .loading
        keywordSpotter = try await keywordFactory(directory)
        voiceInputPreparation = .ready
        refreshAccessibility(prompt: false)
    } catch {
        voiceInputPreparation = .failed(error.localizedDescription)
    }
}
```

另加：

```swift
public func refreshAccessibility(prompt: Bool) {
    accessibilityTrusted = injectionCoordinator.isTrusted(prompt: prompt)
    targetApplicationName = injectionCoordinator.currentTarget()?.name
}
```

- [ ] **Step 5: 實作錄音模式互斥**

將 `toggleRecording()` 換成：

```swift
public func toggleRecording(mode: RecordingMode) async {
    switch state {
    case .ready:
        guard activeMode == nil else { return }
        if mode == .voiceInput {
            guard voiceInputPreparation == .ready, accessibilityTrusted else { return }
        }
        await startRecording(mode: mode)
    case .recording where activeMode == mode:
        await stopRecording()
    default:
        break
    }
}

public func isRecording(_ mode: RecordingMode) -> Bool {
    state == .recording && activeMode == mode
}

public func canToggle(_ mode: RecordingMode) -> Bool {
    if state == .recording { return activeMode == mode }
    guard state == .ready, activeMode == nil else { return false }
    return mode == .transcription ||
        (voiceInputPreparation == .ready && accessibilityTrusted)
}
```

- [ ] **Step 6: 實作 fan-out 與命令 transaction**

開始 voice mode 時依序 `catcher.start()`、`keywordSpotter.start()`、`audio.start()`；任一步失敗都清理先前已啟動元件。

每個 chunk：

```swift
private func processVoiceInput(
    _ samples: [Float],
    catcher: any CatcherServing,
    keywordSpotter: any KeywordSpotting
) async throws {
    let update = try await catcher.push(samples)
    let detection = try await keywordSpotter.push(samples)
    if let detection {
        guard detection.keyword == "TIPPI_GO" else { return }
        let final = try await catcher.finish(before: detection.startMs)
        let event = try injectionCoordinator.submit(final.text)
        applyInjectionEvent(event)
        try await catcher.start()
        try await keywordSpotter.reset()
        injectionCoordinator.resetTurn()
        return
    }
    if let update {
        applyInjectionEvent(try injectionCoordinator.consume(update.text))
    }
}
```

`submit` 回傳 `.waitingForTarget` 時不送 Enter；controller 顯示「請切到目標輸入框後重說 Tippi Go」，仍 reset 本輪以免 deferred send。

`applyInjectionEvent` 使用同一份 UI 狀態 mapping：

```swift
private func applyInjectionEvent(_ event: TextInjectionEvent) {
    switch event {
    case .noChange:
        break
    case .waitingForTarget:
        targetApplicationName = "Tippi"
        voiceInputMessage = "請切到目標輸入框"
    case let .injected(text, target):
        targetApplicationName = target
        lastInjectedText = text
        voiceInputMessage = "已嘗試注入至 \(target)"
    case let .submitted(text, target):
        targetApplicationName = target
        if !text.isEmpty { lastInjectedText = text }
        voiceInputMessage = "已送出"
    case .duplicateCommandIgnored:
        break
    }
}
```

停止時：

```swift
switch activeMode {
case .transcription:
    apply(try await catcher.finish())
case .voiceInput:
    _ = try await catcher.finish()
    try await keywordSpotter?.reset()
    injectionCoordinator.resetTurn()
case nil:
    break
}
activeMode = nil
state = .ready
```

`startRecording(mode:)` 在啟動任何 async service 前先設定 `activeMode = mode`，catch 區塊與
`handleStreamFailure` 都必須停止 audio、finish continuation、cancel task、將
`activeMode = nil` 並 `injectionCoordinator.resetTurn()`，確保失敗後另一分頁可以重試。

- [ ] **Step 7: 跑 controller 與完整 Swift 測試**

Run:

```bash
swift test --package-path apps/tippi --filter TranscriptionControllerTests
swift test --package-path apps/tippi
```

Expected: 命令順序、互斥、failure 與既有轉錄行為全部 PASS。

- [ ] **Step 8: Commit**

```bash
git add apps/tippi/Sources/TippiCore/TippiState.swift apps/tippi/Sources/TippiCore/TranscriptionController.swift apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "feat: coordinate transcription and voice input"
```

---

### Task 8: 語音輸入 UI、App composition 與非 sandbox 打包

**Files:**
- Create: `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`
- Delete: `apps/tippi/Sources/TippiApp/VoiceInputPlaceholderView.swift`
- Modify: `apps/tippi/Sources/TippiApp/ContentView.swift`
- Modify: `apps/tippi/Sources/TippiApp/TranscriptionTabView.swift`
- Modify: `apps/tippi/Sources/TippiApp/TippiApp.swift`
- Modify: `apps/tippi/Resources/Tippi.entitlements`
- Create: `apps/tippi/Resources/THIRD_PARTY_NOTICES.md`
- Modify: `apps/tippi/scripts/build-app.sh`
- Modify: `apps/tippi/scripts/verify-app.sh`

**Interfaces:**
- Voice tab 自動 `prepareVoiceInput()`，顯示 permission/model/target/activity。
- App composition 注入 production migrator、installer、KWS factory、CGEvent coordinator。

- [ ] **Step 1: 先更新 build verification，確認舊 bundle 不符合**

在 `verify-app.sh` 增加：

```zsh
ENTITLEMENTS=$(codesign -d --entitlements - ${APP} 2>&1)
if print -- "${ENTITLEMENTS}" | grep -q 'com.apple.security.app-sandbox'; then
    print -u2 "Tippi must not be sandboxed for cross-app injection"
    exit 1
fi
test -f ${APP}/Contents/Resources/THIRD_PARTY_NOTICES.md
```

Run: `apps/tippi/scripts/build-app.sh`

Expected: FAIL，舊 entitlement 仍含 app sandbox 或 notice 尚未打包。

- [ ] **Step 2: 組合 production dependencies**

`TippiApp.init()` 建立：

```swift
let injector = CGEventTextInjector()
let coordinator = TextInjectionCoordinator(
    injector: injector,
    targetProvider: FrontmostApplicationProvider(),
    ownBundleIdentifier: "com.wcamon.tippi"
)
let keywordInstaller = KeywordModelInstaller()
let modelMigrator = ModelDirectoryMigrator()
```

再傳入 controller：

```swift
modelMigrator: modelMigrator,
keywordInstaller: keywordInstaller,
keywordFactory: { directory in
    try KeywordSpotterClient(modelDirectory: directory)
},
injectionCoordinator: coordinator
```

- [ ] **Step 3: 實作 VoiceInputTabView**

畫面固定包含：

```swift
Text("語音輸入")
Text("口令：Tippi Go")
Label(permissionText, systemImage: permissionSymbol)
Text(controller.targetApplicationName.map { "目前目標：\($0)" } ?? "目前沒有可用目標")
Text(controller.lastInjectedText.isEmpty ? "尚未注入文字" : controller.lastInjectedText)
Button(controller.isRecording(.voiceInput) ? "停止" : "開始語音輸入") {
    Task { await controller.toggleRecording(mode: .voiceInput) }
}
.disabled(!controller.canToggle(.voiceInput))
```

未授權區同時提供：

```swift
Button("要求輔助使用權限") {
    controller.refreshAccessibility(prompt: true)
}
Button("開啟系統設定") {
    NSWorkspace.shared.open(
        URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
    )
}
```

View：

```swift
.task { await controller.prepareVoiceInput() }
.onChange(of: scenePhase) {
    if scenePhase == .active {
        controller.refreshAccessibility(prompt: false)
    }
}
```

模型下載顯示 `ProgressView(value:)`；failed 狀態顯示錯誤與「重試」按鈕。錄音中若目標為 Tippi，顯示「請切到目標輸入框」。

- [ ] **Step 4: 接上雙分頁與錄音互斥**

`ContentView`：

```swift
VoiceInputTabView(controller: controller)
    .tabItem { Label("語音輸入", systemImage: "keyboard") }
```

刪除 placeholder 檔。`TranscriptionTabView` 所有錄音判斷改成 `.transcription`：

```swift
controller.isRecording(.transcription)
await controller.toggleRecording(mode: .transcription)
.disabled(!controller.canToggle(.transcription))
```

- [ ] **Step 5: 移除 sandbox 並打包 notice**

`Tippi.entitlements` 刪除：

```xml
<key>com.apple.security.app-sandbox</key>
<true/>
```

`THIRD_PARTY_NOTICES.md` 至少包含 sherpa-onnx 名稱、Apache-2.0、repo URL 與模型 release URL。

`build-app.sh` 建立 Resources 並複製：

```zsh
mkdir -p ${CONTENTS}/MacOS ${CONTENTS}/Frameworks ${CONTENTS}/Resources
cp apps/tippi/Resources/THIRD_PARTY_NOTICES.md ${CONTENTS}/Resources/
```

- [ ] **Step 6: 建置與驗證**

Run:

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
apps/tippi/scripts/build-app.sh
apps/tippi/scripts/verify-app.sh
otool -L apps/tippi/build/Tippi.app/Contents/Frameworks/libcatcher_ffi.dylib
```

Expected: build/verify PASS；codesign entitlements 不含 sandbox；App bundle 有 notice；`otool` 不出現需要另外複製的 sherpa/onnxruntime dylib。

- [ ] **Step 7: Commit**

```bash
git add apps/tippi/Sources/TippiApp/VoiceInputTabView.swift apps/tippi/Sources/TippiApp/VoiceInputPlaceholderView.swift apps/tippi/Sources/TippiApp/ContentView.swift apps/tippi/Sources/TippiApp/TranscriptionTabView.swift apps/tippi/Sources/TippiApp/TippiApp.swift apps/tippi/Resources/Tippi.entitlements apps/tippi/Resources/THIRD_PARTY_NOTICES.md apps/tippi/scripts/build-app.sh apps/tippi/scripts/verify-app.sh
git commit -m "feat: add voice input tab and packaging"
```

---

### Task 9: 真實 App 驗收與操作文件

**Files:**
- Modify: `README.md`

**Interfaces:**
- 使用者可依 README 完成權限、聚焦、語音輸入、`Tippi Go` 送出與模型位置確認。

- [ ] **Step 1: 更新 README 使用說明**

新增「Voice Input」段落，明確寫：

```text
1. 打開「語音輸入」分頁，等待 Catcher 與 Tippi Go 模型就緒。
2. 授予 Tippi「系統設定 → 隱私權與安全性 → 輔助使用」權限。
3. 按「開始語音輸入」，切到目標 App 並點進輸入框。
4. 說出內容；最後說「Tippi Go」送出。
5. Tippi Go 不會進入輸入框。停止按鈕不會自動送出未完成內容。
```

並記錄模型位置 `~/Library/Application Support/Tippi/Models` 與 v1 不會自動聚焦目標。

- [ ] **Step 2: 執行完整自動驗證**

Run:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
apps/tippi/scripts/build-app.sh
apps/tippi/scripts/verify-app.sh
git diff --check
```

Expected: 全數 PASS；real-model ignored tests除外。

- [ ] **Step 3: 執行 real KWS 測試**

先準備 pinned real model：

```bash
KWS_ARCHIVE=/tmp/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
KWS_ROOT=/tmp/tippi-kws-model-check
curl -fL --retry 2 -o "${KWS_ARCHIVE}" https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2
test "$(shasum -a 256 "${KWS_ARCHIVE}" | cut -d ' ' -f 1)" = "68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6"
rm -rf "${KWS_ROOT}"
mkdir -p "${KWS_ROOT}"
tar -xjf "${KWS_ARCHIVE}" -C "${KWS_ROOT}"
```

再執行：

```bash
SHERPA_KWS_MODEL=/tmp/tippi-kws-model-check/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 cargo test -p catcher-ffi --test kws_ffi -- --ignored --nocapture
```

Expected: positive fixture 偵測 `TIPPI_GO`，三組 unrelated speech 不觸發。

- [ ] **Step 4: 執行真實注入矩陣**

Launch:

```bash
open apps/tippi/build/Tippi.app
```

逐項記錄 PASS/FAIL：

```text
TextEdit：繁中、英文、中英混合、emoji
Chrome textarea
Chrome contenteditable
ChatGPT 網頁
ChatGPT 桌面 App（本機有安裝時）
內容 + Tippi Go：口令不出現、Enter 恰好一次
送出後第二段訊息仍可運作
Tippi 前景：不注入、不送出
焦點移到非文字目標：不崩潰，UI 保持可停止
轉錄分頁錄音時語音輸入按鈕不可啟動，反向亦同
舊 sandbox Models 存在時只遷移，不重新下載大型 ASR/diar 模型
```

若 Accessibility 尚未由使用者授權，完成其他自動驗證，將此一項明確標為需使用者授權的人工 gate；不得用剪貼簿 fallback 讓測試假通過。

- [ ] **Step 5: Commit README**

```bash
git add README.md
git commit -m "docs: explain Tippi voice input"
```

- [ ] **Step 6: 最終工作樹檢查**

Run:

```bash
git status --short
git log --oneline -10
```

Expected: 工作樹乾淨；九個 task 的 commits 依序存在，沒有模型 archive、ONNX 或 `/tmp` 檔案被追蹤。

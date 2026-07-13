# Tippi Voice Input Holdback Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incorrect KWS-relative cutoff with a sample-count-derived 1.5-second holdback so `Tippi Go` never leaks into the target field and long utterances no longer collapse to an earlier prefix.

**Architecture:** Rust exposes a non-destructive timed-token snapshot before an utterance-relative cutoff. Swift counts the exact 16 kHz PCM samples delivered to both ASR and KWS, releases only text older than 1.5 seconds, and uses the same safe cutoff when a command is detected; the sherpa-onnx timestamp remains diagnostic-only.

**Tech Stack:** Rust 2024, Catcher C ABI, Nemotron MLX timed tokens, Swift 6, Swift Observation/SwiftUI, Swift Testing, sherpa-onnx 1.13.4, macOS Accessibility/CGEvent.

## Global Constraints

- Voice input audio is mono Float32 PCM at exactly 16,000 Hz.
- `VoiceInputTiming.holdbackMs` is exactly `1_500`; v1 does not expose a delay setting.
- Stable cutoff is `max(0, floor(receivedSampleCount * 1000 / 16000) - 1500)`.
- KWS `startMs` stays in the ABI and Swift type for diagnostics but must never drive ASR cutoff.
- Stable token filtering remains strict: `token.frame * 80 < cutoff_ms`.
- No clipboard, Backspace, AXValue writes, target focusing, third model, or text-based command fallback.
- An empty safe prefix never sends Return.
- Stop never injects or submits the held tail.
- Every command completion, duplicate-command discard, stop, failure, and retry resets the ASR/KWS/coordinator sample timeline.
- Existing model paths, hashes, sandbox migration, static sherpa linkage, and app bundle rules remain unchanged.

## File Map

- `crates/catcher-ffi/src/lib.rs`: owns the non-destructive token snapshot buffer and C ABI implementation.
- `crates/catcher-ffi/include/catcher.h`: canonical public C declaration and lifetime contract.
- `apps/tippi/Sources/CCatcher/include/catcher.h`: Swift module copy of the same C declaration.
- `crates/catcher-ffi/tests/ffi_lifecycle.rs`: null safety and real-ASR non-mutation coverage.
- `crates/catcher-ffi/tests/kws_ffi.rs`: real-model evidence that KWS timestamps are chunk-relative.
- `apps/tippi/Sources/TippiCore/CatcherClient.swift`: async Swift bridge for `catcher_text_before`.
- `apps/tippi/Sources/TippiCore/VoiceInputTiming.swift`: single source of truth for sample-rate and holdback arithmetic.
- `apps/tippi/Sources/TippiCore/TextInjectionCoordinator.swift`: empty-submit event and Return suppression.
- `apps/tippi/Sources/TippiCore/TranscriptionController.swift`: sample clock, stable snapshot cadence, command cutoff, resets, and failure context.
- `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`: pause guidance and runtime-failure wording.
- `apps/tippi/Tests/TippiCoreTests/CatcherClientTests.swift`: bridge interface coverage.
- `apps/tippi/Tests/TippiCoreTests/TextInjectionCoordinatorTests.swift`: no-text command coverage.
- `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`: holdback, timestamp independence, ordering, reset, and error coverage.
- `README.md`: 1.5-second latency and pause-before-command instructions.

---

### Task 1: Add the non-destructive stable-text C ABI

**Files:**
- Modify: `crates/catcher-ffi/src/lib.rs`
- Modify: `crates/catcher-ffi/include/catcher.h`
- Modify: `apps/tippi/Sources/CCatcher/include/catcher.h`
- Modify: `crates/catcher-ffi/tests/ffi_lifecycle.rs`
- Modify: `crates/catcher-ffi/tests/kws_ffi.rs`

**Interfaces:**
- Consumes: `CatcherHandle.timed_tokens: Vec<TimedToken>`, `ASR_FRAME_MS = 80`, the existing tokenizer/OpenCC path, and the real KWS fixture/model environment.
- Produces: `catcher_text_before(catcher_handle_t *, uint64_t) -> const char *`, whose borrowed pointer is valid until the next mutating/snapshot call.

- [ ] **Step 1: Write failing Rust tests**

Add a pure strict-boundary test inside `crates/catcher-ffi/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ids_before_ms_use_a_strict_eighty_ms_boundary() {
        let tokens = vec![
            TimedToken { id: 10, frame: 0 },
            TimedToken { id: 11, frame: 1 },
            TimedToken { id: 12, frame: 2 },
        ];

        assert_eq!(token_ids_before_ms(&tokens, 0), Vec::<u32>::new());
        assert_eq!(token_ids_before_ms(&tokens, 80), vec![10]);
        assert_eq!(token_ids_before_ms(&tokens, 160), vec![10, 11]);
    }
}
```

Extend `null_arguments_report_errors_without_unwinding` in `ffi_lifecycle.rs`:

First add `catcher_text_before` to the existing `use catcher_ffi::{ ... }` import list, then add:

```rust
assert!(catcher_text_before(ptr::null_mut(), 1_500).is_null());
assert!(!catcher_last_error(ptr::null()).is_null());
```

In `c_abi_transcribes_reference_wav_exactly`, after `catcher_finish(handle)`, verify snapshot reads do not replace the full transcript:

```rust
let full = CStr::from_ptr(catcher_text(handle)).to_str().unwrap().to_owned();
assert_eq!(
    CStr::from_ptr(catcher_text_before(handle, 0)).to_bytes(),
    b""
);
assert_eq!(CStr::from_ptr(catcher_text(handle)).to_str().unwrap(), full);
assert_eq!(
    CStr::from_ptr(catcher_text_before(handle, u64::MAX))
        .to_str()
        .unwrap(),
    full
);
assert_eq!(CStr::from_ptr(catcher_text(handle)).to_str().unwrap(), full);
```

Add an ignored real-model diagnostic to `kws_ffi.rs`:

```rust
#[test]
#[ignore = "requires SHERPA_KWS_MODEL"]
fn reported_timestamp_is_not_an_absolute_long_stream_offset() {
    let model = CString::new(std::env::var("SHERPA_KWS_MODEL").unwrap()).unwrap();
    let handle = unsafe { catcher_kws_create(model.as_ptr()) };
    assert!(!handle.is_null());
    let spoken = read_wav(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/tippi-go.wav"
    ));

    unsafe {
        for prefix_seconds in [1_usize, 5, 10, 20] {
            assert_eq!(catcher_kws_start(handle), CATCHER_OK);
            let mut samples = vec![0.0; prefix_seconds * 16_000];
            samples.extend_from_slice(&spoken);
            samples.extend(vec![0.0; 16_000]);
            assert!(feed_until_detected(handle, &samples));
            let reported = catcher_kws_start_ms(handle);
            assert!(
                reported < prefix_seconds as u64 * 1_000,
                "{prefix_seconds}s prefix unexpectedly produced absolute {reported}ms"
            );
        }
        catcher_kws_destroy(handle);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo test -p catcher-ffi token_ids_before_ms_use_a_strict_eighty_ms_boundary
cargo test -p catcher-ffi --test ffi_lifecycle null_arguments_report_errors_without_unwinding
```

Expected: compile failure because `token_ids_before_ms` and `catcher_text_before` do not exist.

- [ ] **Step 3: Implement the token filter and handle-owned snapshot buffer**

Add this stored property immediately after `text: CString` in `CatcherHandle`:

```rust
text_before_cutoff: CString,
```

Initialize it immediately after `text: empty_c_string()` in `catcher_create`:

```rust
text_before_cutoff: empty_c_string(),
```

Clear it immediately after `handle.text = empty_c_string()` in `catcher_start`:

```rust
handle.text_before_cutoff = empty_c_string();
```

Add the strict helper and exported function:

```rust
fn token_ids_before_ms(tokens: &[TimedToken], cutoff_ms: u64) -> Vec<u32> {
    tokens
        .iter()
        .filter(|token| token.frame.saturating_mul(ASR_FRAME_MS) < cutoff_ms)
        .map(|token| token.id)
        .collect()
}

/// Returns a non-destructive transcript snapshot containing only tokens
/// strictly before `cutoff_ms`.
///
/// # Safety
///
/// `handle` must be null or a live pointer returned by `catcher_create`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn catcher_text_before(
    handle: *mut CatcherHandle,
    cutoff_ms: u64,
) -> *const c_char {
    if handle.is_null() {
        set_global_error("catcher handle is null");
        return ptr::null();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let handle = unsafe { &mut *handle };
        let ids = token_ids_before_ms(&handle.timed_tokens, cutoff_ms);
        let decoded = handle
            .tokenizer
            .decode(&ids, true)
            .map_err(|error| error.to_string())?;
        handle.text_before_cutoff = safe_c_string(&opencc::to_traditional(&decoded));
        Ok::<*const c_char, String>(handle.text_before_cutoff.as_ptr())
    }));
    match result {
        Ok(Ok(pointer)) => pointer,
        Ok(Err(error)) => {
            unsafe { &mut *handle }.last_error = safe_c_string(&error);
            ptr::null()
        }
        Err(payload) => {
            unsafe { &mut *handle }.last_error = safe_c_string(&panic_message(payload));
            ptr::null()
        }
    }
}
```

In both headers, place this declaration and contract immediately after `catcher_text`:

```c
/*
 * Returns a non-destructive UTF-8 transcript snapshot containing timed
 * tokens whose frame start is strictly earlier than cutoff_ms. This call
 * does not flush, discard tokens, change session state, or replace the full
 * catcher_text transcript. The borrowed pointer is valid until the next
 * mutating call, catcher_text_before call, or catcher_destroy.
 * Returns NULL and records catcher_last_error on failure.
 */
const char *catcher_text_before(catcher_handle_t *handle, uint64_t cutoff_ms);
```

- [ ] **Step 4: Run Rust and real-model tests**

Run:

```bash
cargo fmt --all -- --check
cargo test -p catcher-ffi
NEMOTRON_MLX_ARTIFACT="$HOME/Library/Application Support/Tippi/Models/catcher-asr-mlx-int8" \
  cargo test -p catcher-ffi --test ffi_lifecycle c_abi_transcribes_reference_wav_exactly -- --ignored --nocapture
SHERPA_KWS_MODEL="$HOME/Library/Application Support/Tippi/Models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20" \
  cargo test -p catcher-ffi --test kws_ffi reported_timestamp_is_not_an_absolute_long_stream_offset -- --ignored --nocapture
```

Expected: all commands pass; the KWS diagnostic detects all four padded commands while proving the reported timestamp is smaller than the preceding stream duration.

- [ ] **Step 5: Commit**

```bash
git add crates/catcher-ffi/src/lib.rs crates/catcher-ffi/include/catcher.h \
  apps/tippi/Sources/CCatcher/include/catcher.h \
  crates/catcher-ffi/tests/ffi_lifecycle.rs crates/catcher-ffi/tests/kws_ffi.rs
git commit -m "feat: expose stable transcript snapshots"
```

---

### Task 2: Bridge stable snapshots into Swift

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/CatcherClient.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/CatcherClientTests.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Consumes: `catcher_text_before(catcher_handle_t *, uint64_t)` from Task 1.
- Produces: `CatcherServing.text(before:) async throws -> String`; the controller fake can script stable snapshots and records `textBefore:<cutoff>`.

- [ ] **Step 1: Write the failing Swift interface test**

Create `CatcherClientTests.swift`:

```swift
import Testing
@testable import TippiCore

private actor SnapshotCatcher: CatcherServing {
    func start() async throws {}
    func push(_ samples: [Float]) async throws -> TranscriptUpdate? { nil }
    func finish() async throws -> TranscriptUpdate {
        TranscriptUpdate(text: "", segments: [], warning: nil)
    }
    func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate {
        TranscriptUpdate(text: "", segments: [], warning: nil)
    }
    func text(before cutoffMs: UInt64) async throws -> String {
        "stable:\(cutoffMs)"
    }
}

@Test
func catcherServingExposesStableSnapshot() async throws {
    let service: any CatcherServing = SnapshotCatcher()
    #expect(try await service.text(before: 1_500) == "stable:1500")
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi --filter catcherServingExposesStableSnapshot
```

Expected: compile failure because `CatcherServing` and `CatcherClient` do not yet expose `text(before:)`.

- [ ] **Step 3: Implement the bridge and update the controller fake**

Add to `CatcherServing` and `CatcherClient`:

```swift
public protocol CatcherServing: Sendable {
    func start() async throws
    func push(_ samples: [Float]) async throws -> TranscriptUpdate?
    func text(before cutoffMs: UInt64) async throws -> String
    func finish() async throws -> TranscriptUpdate
    func finish(before cutoffMs: UInt64) async throws -> TranscriptUpdate
}
```

```swift
public func text(before cutoffMs: UInt64) async throws -> String {
    guard let pointer = catcher_text_before(owner.pointer, cutoffMs) else {
        throw CatcherClientError.operationFailed(currentError())
    }
    return String(cString: pointer)
}
```

Extend `FakeCatcher` in `TranscriptionControllerTests.swift`:

```swift
private var stableTexts: [String] = []
private var textBeforeError: TestFailure?

func text(before cutoffMs: UInt64) async throws -> String {
    record("textBefore:\(cutoffMs)")
    if let textBeforeError { throw textBeforeError }
    return stableTexts.isEmpty ? "" : stableTexts.removeFirst()
}

func failTextBefore(with error: TestFailure) {
    textBeforeError = error
}
```

Replace its `script` helper with:

```swift
func script(
    pushes: [TranscriptUpdate?] = [],
    stableTexts: [String] = [],
    finish: TranscriptUpdate = TranscriptUpdate(text: "", segments: [], warning: nil),
    finishBefore: TranscriptUpdate? = nil
) {
    pushUpdates = pushes
    self.stableTexts = stableTexts
    finishUpdate = finish
    finishBeforeUpdate = finishBefore ?? finish
}
```

- [ ] **Step 4: Run all Swift tests**

Run:

```bash
cargo build -p catcher-ffi --release
swift test --package-path apps/tippi
```

Expected: the new interface test and all existing tests pass; no other `CatcherServing` conformer is missing the method.

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/CatcherClient.swift \
  apps/tippi/Tests/TippiCoreTests/CatcherClientTests.swift \
  apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "feat: bridge stable transcript snapshots into Swift"
```

---

### Task 3: Prevent empty command submission

**Files:**
- Modify: `apps/tippi/Sources/TippiCore/TextInjectionCoordinator.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/TextInjectionCoordinatorTests.swift`
- Modify: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`

**Interfaces:**
- Consumes: existing `TextInjectionCoordinator.submit(_:)` and `TextInjectionEvent` handling.
- Produces: `TextInjectionEvent.nothingToSubmit`; empty safe text performs no injection and no Return.

- [ ] **Step 1: Write the failing no-text test**

Add to `TextInjectionCoordinatorTests.swift`:

```swift
@MainActor
@Test
func emptyTurnDoesNotPressReturn() throws {
    let (coordinator, injector, _) = makeCoordinator()

    #expect(try coordinator.submit("") == .nothingToSubmit)
    #expect(injector.events.isEmpty)
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
swift test --package-path apps/tippi --filter emptyTurnDoesNotPressReturn
```

Expected: compile failure because `.nothingToSubmit` does not exist.

- [ ] **Step 3: Implement the event and empty guard**

Add the event:

```swift
public enum TextInjectionEvent: Equatable, Sendable {
    case noChange
    case waitingForTarget
    case injected(text: String, target: String)
    case submitted(text: String, target: String)
    case nothingToSubmit
    case duplicateCommandIgnored
}
```

At the start of `submit`, after the duplicate-command guard and before target lookup, add:

```swift
guard !fullText.isEmpty || !injectedPrefix.isEmpty else {
    return .nothingToSubmit
}
```

Add the exhaustive controller event handling:

```swift
case .nothingToSubmit:
    lastInjectedText = ""
    voiceInputMessage = "沒有可送出的文字"
```

- [ ] **Step 4: Run coordinator and controller tests**

Run:

```bash
swift test --package-path apps/tippi --filter emptyTurnDoesNotPressReturn
swift test --package-path apps/tippi
```

Expected: all selected tests pass; the fake injector records neither text nor Return for an empty turn.

- [ ] **Step 5: Commit**

```bash
git add apps/tippi/Sources/TippiCore/TextInjectionCoordinator.swift \
  apps/tippi/Tests/TippiCoreTests/TextInjectionCoordinatorTests.swift \
  apps/tippi/Sources/TippiCore/TranscriptionController.swift
git commit -m "fix: suppress empty voice command submissions"
```

---

### Task 4: Drive injection and command cutoff from the sample clock

**Files:**
- Create: `apps/tippi/Sources/TippiCore/VoiceInputTiming.swift`
- Modify: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- Modify: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Consumes: `CatcherServing.text(before:)`, `TextInjectionEvent.nothingToSubmit`, 16 kHz audio chunks, and KWS detection as a command-only signal.
- Produces: `VoiceInputTiming.stableCutoffMs(receivedSampleCount:)`; `TranscriptionController.failedMode`; reset-safe holdback flow.

- [ ] **Step 1: Write failing timing and controller tests**

Add timing arithmetic coverage:

```swift
@Test(arguments: zip(
    [UInt64(0), 23_999, 24_000, 40_000],
    [UInt64(0), 0, 0, 1_000]
))
func stableCutoffUsesSixteenKilohertzSampleClock(
    sampleCount: UInt64,
    expected: UInt64
) {
    #expect(VoiceInputTiming.stableCutoffMs(receivedSampleCount: sampleCount) == expected)
}
```

Add a silence-release regression:

```swift
@MainActor
@Test
func heldTextBecomesInjectableAsSilenceAdvancesTheClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        pushes: [TranscriptUpdate(text: "你好", segments: [], warning: nil), nil],
        stableTexts: ["", "你好"]
    )
    await fixture.keywordSpotter.script([nil, nil])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:0") }
    #expect(fixture.injector.injected.isEmpty)

    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:500") }
    #expect(fixture.injector.injected == ["你好"])
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

Replace the old command cutoff assertion with a parameterized regression that ignores KWS time:

```swift
@MainActor
@Test(arguments: [UInt64(0), 320, 960])
func commandCutoffUsesSampleClockInsteadOfKwsTimestamp(startMs: UInt64) async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "你好", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: startMs)
    ])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { log.snapshot().contains("kws.reset") }

    #expect(log.snapshot().contains("asr.finishBefore:1000"))
    if startMs != 1_000 {
        #expect(!log.snapshot().contains("asr.finishBefore:\(startMs)"))
    }
    #expect(fixture.injector.injected == ["你好"])
    #expect(fixture.injector.submitCount == 1)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

Add reset coverage:

```swift
@MainActor
@Test
func commandResetStartsTheNextTurnAtZeroMilliseconds() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(
        pushes: [nil],
        stableTexts: [""],
        finishBefore: TranscriptUpdate(text: "第一輪", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 320),
        nil,
    ])
    await prepareVoiceFixture(fixture)
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { fixture.injector.submitCount == 1 }
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:0") }

    #expect(log.snapshot().filter { $0 == "asr.finishBefore:1000" }.count == 1)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

Add empty-safe-prefix command coverage:

```swift
@MainActor
@Test
func commandInsideInitialHoldbackDoesNotPressReturn() async throws {
    let fixture = makeFixture()
    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 960)
    ])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { fixture.controller.voiceInputMessage == "沒有可送出的文字" }

    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

Add this reset helper to `FakeKeywordSpotter`:

```swift
func clearPushError() { pushError = nil }
```

Add failure-and-retry reset coverage:

```swift
@MainActor
@Test
func streamFailureAndRetryResetTheSampleClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.keywordSpotter.failPush(with: .keyword)
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { fixture.controller.state == .failed("keyword failed") }

    await fixture.keywordSpotter.clearPushError()
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 320)
    ])
    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "", segments: [], warning: nil)
    )
    await fixture.controller.prepare()
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.finishBefore:0") }

    #expect(fixture.injector.submitCount == 0)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

Add stop-and-restart reset coverage:

```swift
@MainActor
@Test
func stopAndRestartResetTheSampleClock() async throws {
    let log = EventLog()
    let fixture = makeFixture(log: log)
    await fixture.catcher.script(pushes: [nil], stableTexts: [""])
    await fixture.keywordSpotter.script([nil])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { log.snapshot().contains("asr.textBefore:1000") }
    await fixture.controller.toggleRecording(mode: .voiceInput)

    await fixture.catcher.script(
        finishBefore: TranscriptUpdate(text: "", segments: [], warning: nil)
    )
    await fixture.keywordSpotter.script([
        KeywordDetection(keyword: "TIPPI_GO", startMs: 960)
    ])
    log.removeAll()

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 16_000))
    try await waitUntil { log.snapshot().contains("asr.finishBefore:0") }

    #expect(fixture.injector.submitCount == 0)
    await fixture.controller.toggleRecording(mode: .voiceInput)
}
```

Add stable-snapshot failure coverage:

```swift
@MainActor
@Test
func stableSnapshotFailureStopsVoiceInputWithoutSubmitting() async throws {
    let fixture = makeFixture()
    await fixture.catcher.failTextBefore(with: .inference)
    await fixture.keywordSpotter.script([nil])
    await prepareVoiceFixture(fixture)

    await fixture.controller.toggleRecording(mode: .voiceInput)
    await fixture.audio.emit([Float](repeating: 0, count: 40_000))
    try await waitUntil { fixture.controller.state == .failed("inference failed") }

    #expect(fixture.controller.failedMode == .voiceInput)
    #expect(fixture.controller.activeMode == nil)
    #expect(fixture.injector.injected.isEmpty)
    #expect(fixture.injector.submitCount == 0)
    #expect(await fixture.audio.snapshot() == ["start", "stop"])
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
swift test --package-path apps/tippi --filter stableCutoffUsesSixteenKilohertzSampleClock
swift test --package-path apps/tippi --filter commandCutoffUsesSampleClockInsteadOfKwsTimestamp
```

Expected: the timing test cannot compile because `VoiceInputTiming` does not exist; the controller test still logs `finishBefore:<KWS startMs>`.

- [ ] **Step 3: Implement the timing helper**

Create `VoiceInputTiming.swift`:

```swift
public enum VoiceInputTiming {
    public static let sampleRate: UInt64 = 16_000
    public static let holdbackMs: UInt64 = 1_500

    public static func stableCutoffMs(receivedSampleCount: UInt64) -> UInt64 {
        let audioEndMs = receivedSampleCount * 1_000 / sampleRate
        return audioEndMs > holdbackMs ? audioEndMs - holdbackMs : 0
    }
}
```

- [ ] **Step 4: Implement sample accounting, snapshot cadence, and resets**

Add controller state:

```swift
public private(set) var failedMode: RecordingMode?
@ObservationIgnored private var receivedSampleCount: UInt64 = 0
```

At the first line of `prepare()`, clear preparation failure context:

```swift
failedMode = nil
```

Immediately after `guard let catcher else { return }` in `startRecording(mode:)`, clear the previous runtime failure:

```swift
failedMode = nil
```

When voice input starts, reset the timeline:

```swift
case .voiceInput:
    lastInjectedText = ""
    voiceInputMessage = "請切到目標輸入框"
    resetVoiceTurn()
    targetApplicationName = injectionCoordinator.currentTarget()?.name
```

Replace `processVoiceInput` with:

```swift
private func processVoiceInput(
    _ samples: [Float],
    catcher: any CatcherServing,
    keywordSpotter: any KeywordSpotting
) async throws {
    receivedSampleCount += UInt64(samples.count)
    _ = try await catcher.push(samples)
    let detection = try await keywordSpotter.push(samples)
    let cutoffMs = VoiceInputTiming.stableCutoffMs(
        receivedSampleCount: receivedSampleCount
    )

    if let detection {
        guard detection.keyword == "TIPPI_GO" else { return }
        if suppressImmediateDuplicateCommand {
            try await catcher.start()
            try await keywordSpotter.reset()
            resetVoiceTurn()
            return
        }

        let final = try await catcher.finish(before: cutoffMs)
        let event = try injectionCoordinator.submit(final.text)
        applyInjectionEvent(event)
        if event == .waitingForTarget {
            voiceInputMessage = "請切到目標輸入框後重說 Tippi Go"
        }
        try await catcher.start()
        try await keywordSpotter.reset()
        resetVoiceTurn()
        suppressImmediateDuplicateCommand = true
        return
    }

    suppressImmediateDuplicateCommand = false
    let stableText = try await catcher.text(before: cutoffMs)
    applyInjectionEvent(try injectionCoordinator.consume(stableText))
}
```

Add the single reset helper and replace every voice-input `injectionCoordinator.resetTurn()` cleanup call with it:

```swift
private func resetVoiceTurn() {
    receivedSampleCount = 0
    injectionCoordinator.resetTurn()
}
```

In the `stopRecording` catch block, set the failed mode before clearing `activeMode`:

```swift
failedMode = mode
if mode == .voiceInput {
    try? await keywordSpotter?.reset()
    resetVoiceTurn()
}
suppressImmediateDuplicateCommand = false
activeMode = nil
state = .failed(error.localizedDescription)
```

Replace `cleanupAfterStartFailure` with:

```swift
private func cleanupAfterStartFailure(
    _ error: any Error,
    catcher: any CatcherServing,
    catcherStarted: Bool,
    keywordSpotter: (any KeywordSpotting)?
) async {
    let failedRecordingMode = activeMode
    await audio.stop()
    audioContinuation?.finish()
    audioContinuation = nil
    let task = audioTask
    audioTask = nil
    task?.cancel()
    await task?.value
    if catcherStarted {
        _ = try? await catcher.finish()
    }
    try? await keywordSpotter?.reset()
    if failedRecordingMode == .voiceInput {
        resetVoiceTurn()
    } else {
        injectionCoordinator.resetTurn()
    }
    suppressImmediateDuplicateCommand = false
    activeMode = nil
    failedMode = failedRecordingMode
    state = .failed(error.localizedDescription)
}
```

Replace `handleStreamFailure` with:

```swift
private func handleStreamFailure(
    _ error: any Error,
    catcher: any CatcherServing,
    keywordSpotter: (any KeywordSpotting)?
) async {
    let failedRecordingMode = activeMode
    state = .finishing
    await audio.stop()
    audioContinuation?.finish()
    audioContinuation = nil
    audioTask?.cancel()
    audioTask = nil
    _ = try? await catcher.finish()
    try? await keywordSpotter?.reset()
    if failedRecordingMode == .voiceInput {
        resetVoiceTurn()
    } else {
        injectionCoordinator.resetTurn()
    }
    suppressImmediateDuplicateCommand = false
    activeMode = nil
    failedMode = failedRecordingMode
    state = .failed(error.localizedDescription)
}
```

For prepare/install failures `failedMode` stays `nil`. Every successful command, duplicate discard, stop, and voice-input failure calls `resetVoiceTurn()` exactly once.

- [ ] **Step 5: Update existing exact-order expectations**

For non-command voice chunks, include `asr.textBefore:<sample-derived cutoff>` after `kws.push`. For the command test, replace the old `asr.finishBefore:960` expectation with `asr.finishBefore:1000`. Keep the required command order:

```text
asr.push
kws.push
asr.finishBefore:1000
inject:你好
submit
asr.start
kws.reset
```

Ensure existing stop, duplicate-command, Tippi-frontmost, and KWS-failure tests expect the new `textBefore` calls only on non-command chunks.

- [ ] **Step 6: Run controller and full Swift tests**

Run:

```bash
swift test --package-path apps/tippi --filter heldTextBecomesInjectableAsSilenceAdvancesTheClock
swift test --package-path apps/tippi
```

Expected: all holdback regressions and the full Swift suite pass with no flaky asynchronous assertions.

- [ ] **Step 7: Commit**

```bash
git add apps/tippi/Sources/TippiCore/VoiceInputTiming.swift \
  apps/tippi/Sources/TippiCore/TranscriptionController.swift \
  apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift
git commit -m "fix: hold back unstable voice input text"
```

---

### Task 5: Update voice-input guidance and runtime error copy

**Files:**
- Modify: `apps/tippi/Sources/TippiApp/VoiceInputTabView.swift`
- Modify: `README.md`

**Interfaces:**
- Consumes: `TranscriptionController.failedMode` and the fixed 1.5-second timing behavior from Task 4.
- Produces: user-facing pause guidance and a runtime error title that does not claim an already-loaded model failed preparation.

- [ ] **Step 1: Update SwiftUI copy**

Change the header subtitle to:

```swift
Text("內容說完後短暫停頓，再說 Tippi Go 送出。")
```

Change the active recording hint to:

```swift
if controller.isRecording(.voiceInput) {
    return "文字約延遲 1.5 秒；短暫停頓後說 Tippi Go。"
}
```

In the `.failed(message)` model-state branch, choose title and retry label from failure context:

```swift
let runtimeFailure = controller.failedMode == .voiceInput
Label(
    runtimeFailure ? "語音輸入已停止" : "語音辨識模型準備失敗",
    systemImage: "exclamationmark.triangle.fill"
)
.foregroundStyle(.red)
Text(message)
    .font(.caption)
    .foregroundStyle(.secondary)
    .lineLimit(3)
Button(runtimeFailure ? "重新準備語音輸入" : "重試語音辨識模型") {
    Task { await controller.prepare() }
}
.buttonStyle(.borderedProminent)
```

- [ ] **Step 2: Update README behavior and limitation**

Replace the Voice Input instructions with wording that preserves the existing numbered flow and adds these exact facts:

```text
- Live text is intentionally held for about 1.5 seconds before injection.
- After finishing the content, pause briefly (about 0.5 seconds), then say Tippi Go.
- The pause keeps the final content outside the command safety window.
- If content runs directly into the command, Tippi prefers dropping a very short tail over leaking command words.
- sherpa-onnx timestamps are diagnostic-only; cutoff follows the shared 16 kHz sample clock.
```

Keep the existing model location, Accessibility, no-clipboard, mode-exclusion, and stop-does-not-submit documentation unchanged.

- [ ] **Step 3: Verify copy and build the app**

Run:

```bash
rg -n "1.5|短暫停頓|語音輸入已停止|diagnostic-only" \
  README.md apps/tippi/Sources/TippiApp/VoiceInputTabView.swift
swift test --package-path apps/tippi
apps/tippi/scripts/build-app.sh
apps/tippi/scripts/verify-app.sh
```

Expected: required copy is present, Swift tests pass, and the signed bundle verifies without the app-sandbox entitlement.

- [ ] **Step 4: Commit**

```bash
git add apps/tippi/Sources/TippiApp/VoiceInputTabView.swift README.md
git commit -m "docs: explain voice input safety delay"
```

---

### Task 6: Full regression and real cross-App acceptance

**Files:**
- Verify only; do not commit model archives, ONNX files, generated app contents, or `/tmp` diagnostics.

**Interfaces:**
- Consumes: the completed Rust/Swift/app changes from Tasks 1–5 and the installed local ASR/KWS models.
- Produces: evidence that automated tests, real-model behavior, bundle constraints, and TextEdit injection satisfy the approved spec.

- [ ] **Step 1: Run the complete automated suite**

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

Expected: every command passes; tests requiring external model variables remain ignored in the ordinary workspace run.

- [ ] **Step 2: Run real ASR and KWS regressions**

Run:

```bash
NEMOTRON_MLX_ARTIFACT="$HOME/Library/Application Support/Tippi/Models/catcher-asr-mlx-int8" \
  cargo test -p catcher-ffi --test ffi_lifecycle c_abi_transcribes_reference_wav_exactly -- --ignored --nocapture
SHERPA_KWS_MODEL="$HOME/Library/Application Support/Tippi/Models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20" \
  cargo test -p catcher-ffi --test kws_ffi -- --ignored --nocapture
```

Expected: the real ASR snapshot test passes without mutating full text; positive KWS, reset, unrelated-audio negatives, and long-prefix timestamp diagnostics pass.

- [ ] **Step 3: Verify linkage, entitlements, and packaged notices**

Run:

```bash
codesign -d --entitlements - apps/tippi/build/Tippi.app
otool -L apps/tippi/build/Tippi.app/Contents/Frameworks/libcatcher_ffi.dylib
test -f apps/tippi/build/Tippi.app/Contents/Resources/THIRD_PARTY_NOTICES.md
```

Expected: no `com.apple.security.app-sandbox`; no external sherpa/onnxruntime dylib; third-party notice exists.

- [ ] **Step 4: Run the TextEdit acceptance matrix**

Launch:

```bash
open -na apps/tippi/build/Tippi.app
```

With Accessibility already enabled, record PASS/FAIL for each item:

```text
TextEdit 中文：內容 -> 停頓約 0.5 秒 -> Tippi Go
TextEdit English: content -> short pause -> Tippi Go
TextEdit 中英混合與 emoji
文字在說出後約 1.5 秒才注入
Tippi Go 與主 ASR 近似拼字均未出現
Return 恰好一次
送出後第二輪重新從 0 ms 運作
命令出現在前 1.5 秒且無內容時不送空白 Return
停止按鈕不注入 holdback 尾段
Tippi 前景不注入、不送出
非文字焦點不崩潰且仍可停止
```

If the user approves broader target testing, repeat the content/pause/command sequence in Chrome textarea, Chrome contenteditable, ChatGPT web, and ChatGPT desktop.

- [ ] **Step 5: Final repository check**

Run:

```bash
git status --short
git log --oneline -12
git ls-files | rg '\.(onnx|tar\.bz2)$' && exit 1 || true
```

Expected: worktree clean; the five implementation commits are present after the two holdback design commits; no model binary or archive is tracked.

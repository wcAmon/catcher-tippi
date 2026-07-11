# Catcher and Tippi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an incremental Catcher Rust + MLX-C library and a native macOS Tippi app that downloads the public model, records through an explicit toggle, and displays live partial text.

**Architecture:** Catcher buffers mono 16 kHz Float32 samples and advances the existing cache-aware encoder/RNNT only when an exact model chunk is available; a panic-safe C ABI exposes one serialized session to Swift. Tippi is a Swift Package executable bundled as a `.app`; it uses URLSession/CryptoKit for first-run model installation, AVAudioEngine/AVAudioConverter for microphone input, and an observable controller for UI state.

**Tech Stack:** Rust 1.85+, mlx-rs/MLX-C, C ABI, Swift 6.2, SwiftUI, AVFoundation, URLSession, CryptoKit, Swift Package Manager, macOS 15+, Xcode 26.

## Global Constraints

- Runtime inference must remain local and require no Python, PyTorch, NeMo, or ONNX Runtime.
- The public model source is `wcamon/catcher-asr-mlx-int8`; Tippi must never embed a Hugging Face token.
- Microphone audio must reach Catcher as mono Float32 at exactly 16,000 Hz.
- The recording control is an explicit on/off toggle; partial text appears while it is on and final text remains after it is off.
- Model weights load once per app process; each recording creates fresh encoder/RNNT caches.
- All FFI functions catch Rust panics and never unwind into Swift.
- Production behavior is added only after a directly relevant failing test.
- The first release targets arm64 macOS 15 or newer and is locally ad-hoc signed, not App Store packaged or notarized.

---

### Task 1: Incremental Catcher streaming state

**Files:**
- Modify: `crates/nemotron-mlx/src/model/stream.rs`
- Modify: `crates/nemotron-mlx/src/model/mod.rs`
- Create: `crates/nemotron-mlx/tests/incremental_stream.rs`
- Modify: `crates/nemotron-mlx/tests/real_checkpoint.rs`

**Interfaces:**
- Produces: `StreamingTranscriber::push_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<u32>>`.
- Produces: `StreamingTranscriber::finish(&mut self) -> ModelResult<Vec<u32>>`.
- Produces: `StreamingTranscriber::reset(&mut self) -> ModelResult<()>`.
- Preserves: `transcribe_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<u32>>` as `push_samples` plus `finish`.

- [x] **Step 1: Write the failing threshold and lifecycle tests**

  Add tests around a small extracted `AudioChunkScheduler` so `4_039` samples emit no chunk, sample `4_040` emits the first `[start=0,length=4_040,center=true]` chunk, later complete windows use `5_520` samples and `center=false`, and `finish` emits at most one padded tail. Add errors for push-after-finish and duplicate finish.

- [x] **Step 2: Run the new test and verify RED**

  Run: `cargo test -p nemotron-mlx --test incremental_stream`

  Expected: compilation fails because `AudioChunkScheduler`, `push_samples`, `finish`, and `reset` do not exist.

- [x] **Step 3: Implement the scheduler and incremental methods**

  Store accumulated samples, `mel_frame_index`, `first_processed`, and `finished`. Process only complete chunks during `push_samples`; at `finish`, pad the first chunk for a short utterance or the one remaining subsequent chunk. Reconstruct the encoder, decoder, and caches in `reset` without reloading `Artifact` tensors by moving their creation into reusable model fields or a private session-state constructor.

- [x] **Step 4: Prove split-input equivalence**

  Add a gated real-checkpoint test that pushes the existing WAV in irregular blocks `[127, 1024, 333, 4096, ...]`, calls `finish`, and asserts exact token equality with both the one-shot Catcher result and the official reference IDs.

- [x] **Step 5: Run tests and commit**

  Run: `cargo test -p nemotron-mlx --test incremental_stream`

  Run: `NEMOTRON_MLX_ARTIFACT=... cargo test -p nemotron-mlx --test real_checkpoint -- --ignored --test-threads=1`

  Commit: `feat: make Catcher transcription incremental`

### Task 2: Panic-safe Catcher C ABI

**Files:**
- Create: `crates/catcher-ffi/Cargo.toml`
- Create: `crates/catcher-ffi/src/lib.rs`
- Create: `crates/catcher-ffi/include/catcher.h`
- Create: `crates/catcher-ffi/tests/ffi_lifecycle.rs`
- Modify: `Cargo.toml`

**Interfaces:**
- Produces opaque `catcher_handle_t`.
- Produces C functions `catcher_create`, `catcher_start`, `catcher_push_audio`, `catcher_finish`, `catcher_text`, `catcher_last_error`, and `catcher_destroy`.
- Produces status codes `CATCHER_OK=0`, `CATCHER_NO_UPDATE=1`, `CATCHER_INVALID_ARGUMENT=-1`, `CATCHER_INVALID_STATE=-2`, and `CATCHER_RUNTIME_ERROR=-3`.

- [ ] **Step 1: Write a failing ABI lifecycle test**

  The Rust integration test calls the exported `extern "C"` functions exactly as Swift will: null create arguments fail, a valid fake/session harness starts, zero-length audio is accepted, text pointers contain valid NUL-terminated UTF-8, finish locks the session, restart clears text, and destroy accepts null.

- [ ] **Step 2: Run the test and verify RED**

  Run: `cargo test -p catcher-ffi --test ffi_lifecycle`

  Expected: Cargo reports that package `catcher-ffi` does not exist.

- [ ] **Step 3: Implement the crate and header**

  Build `crate-type = ["cdylib", "staticlib", "rlib"]`. Store `StreamingTranscriber`, `Tokenizer`, cumulative token IDs, current `CString`, and last-error `CString` in the handle. Wrap every exported body in `catch_unwind(AssertUnwindSafe(...))`; validate pointer/count pairs before slice construction; never expose a temporary string.

- [ ] **Step 4: Add real C-link smoke coverage**

  Compile a tiny C program against `include/catcher.h` and the release dylib, call `catcher_destroy(NULL)`, and verify process exit zero. Inspect `otool -L` to require only Apple/system dynamic dependencies.

- [ ] **Step 5: Run tests and commit**

  Run: `cargo test -p catcher-ffi`

  Run: `cargo build -p catcher-ffi --release`

  Commit: `feat: expose Catcher through a C ABI`

### Task 3: Tippi model installation and state machine

**Files:**
- Create: `apps/tippi/Package.swift`
- Create: `apps/tippi/Sources/CCatcher/include/catcher.h`
- Create: `apps/tippi/Sources/CCatcher/module.modulemap`
- Create: `apps/tippi/Sources/TippiCore/ModelManifest.swift`
- Create: `apps/tippi/Sources/TippiCore/ModelStore.swift`
- Create: `apps/tippi/Sources/TippiCore/TippiState.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/ModelStoreTests.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/TippiStateTests.swift`

**Interfaces:**
- Produces: `ModelStore.installIfNeeded(progress:) async throws -> URL`.
- Produces: `ModelFile(name: String, sha256: String, required: Bool)` for all runtime files.
- Produces: `TippiState` cases `modelMissing`, `downloading(Double)`, `loading`, `ready`, `recording`, `finishing`, and `failed(String)`.

- [ ] **Step 1: Write failing Swift tests**

  Use injected `ModelDownloading` and temporary directories. Assert that progress is monotonic, checksum mismatch removes the staging directory, a complete verified download is atomically moved to the final path, an existing valid model skips network access, and invalid state transitions are rejected.

- [ ] **Step 2: Run tests and verify RED**

  Run: `swift test --package-path apps/tippi`

  Expected: SwiftPM fails because `Package.swift` and TippiCore types do not exist.

- [ ] **Step 3: Implement model installation**

  Use public `https://huggingface.co/wcamon/catcher-asr-mlx-int8/resolve/main/<file>?download=true` URLs. Stream downloads to a sibling `.partial` directory, compute CryptoKit SHA-256, compare `weights.safetensors`, `manifest.json`, `config.json`, and `tokenizer.json` against pinned values, and copy the remaining tokenizer/config/license notices before atomic promotion.

- [ ] **Step 4: Implement the state reducer**

  Keep allowed transitions explicit: missing→downloading→loading→ready; ready→recording→finishing→ready; any operational state→failed; failed→missing or ready through retry after checking the model.

- [ ] **Step 5: Run tests and commit**

  Run: `swift test --package-path apps/tippi`

  Commit: `feat: add Tippi model installation state`

### Task 4: Microphone capture, live transcription, and SwiftUI

**Files:**
- Create: `apps/tippi/Sources/TippiCore/CatcherClient.swift`
- Create: `apps/tippi/Sources/TippiCore/AudioRecorder.swift`
- Create: `apps/tippi/Sources/TippiCore/TranscriptionController.swift`
- Create: `apps/tippi/Sources/TippiApp/TippiApp.swift`
- Create: `apps/tippi/Sources/TippiApp/ContentView.swift`
- Create: `apps/tippi/Tests/TippiCoreTests/TranscriptionControllerTests.swift`

**Interfaces:**
- Produces: `CatcherServing.start()`, `push(_ samples: [Float]) -> String?`, `finish() -> String`.
- Produces: `AudioRecording.start(onSamples:) async throws` and `stop()`.
- Produces: `@MainActor TranscriptionController` with published `state`, `text`, `downloadProgress`, and `isRecording`.

- [ ] **Step 1: Write controller tests and verify RED**

  With fake audio and Catcher clients, assert toggle-on clears old text and starts both services; pushed blocks update partial text; toggle-off stops audio before calling finish; final text persists; and permission/inference failures enter `failed` without leaving recording active.

- [ ] **Step 2: Implement CatcherClient**

  Wrap the C handle in a serial actor. Convert model/language paths with `withCString`, copy `catcher_text` immediately into Swift `String`, and destroy the handle in `deinit` through a small owned reference type.

- [ ] **Step 3: Implement AudioRecorder**

  Request AVAudioApplication microphone permission, install an AVAudioEngine tap, convert input to `AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false)`, and copy channel data before leaving the tap callback.

- [ ] **Step 4: Implement controller and UI**

  Create a single-window SwiftUI app. Show product name, state label, download progress, a scrollable selectable transcript, error/retry affordance, and a large accessible toggle whose label is `Start Recording` or `Stop Recording`. Disable it before `ready`; use color and text together rather than color alone.

- [ ] **Step 5: Run tests and commit**

  Run: `swift test --package-path apps/tippi`

  Commit: `feat: add Tippi live recording UI`

### Task 5: Build a runnable Tippi.app and validate end to end

**Files:**
- Create: `apps/tippi/Resources/Info.plist`
- Create: `apps/tippi/Resources/Tippi.entitlements`
- Create: `apps/tippi/scripts/build-app.sh`
- Create: `apps/tippi/scripts/run-reference.sh`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-11-catcher-tippi.md`

**Interfaces:**
- Produces: `apps/tippi/build/Tippi.app` containing the Swift executable and `Contents/Frameworks/libcatcher_ffi.dylib`.
- Produces: an ad-hoc signed local development app with `NSMicrophoneUsageDescription` and audio-input entitlement.

- [ ] **Step 1: Write a failing bundle verification script**

  The script requires `Tippi.app/Contents/MacOS/Tippi`, the embedded Catcher dylib, Info.plist microphone text, `@executable_path/../Frameworks` resolution, valid ad-hoc signing, and an arm64 Mach-O executable. Run it before the build script exists and confirm failure.

- [ ] **Step 2: Implement the build script**

  Build `catcher-ffi --release`, build the Swift package in release mode, assemble the standard macOS bundle directories, copy the dylib, normalize its install name to `@rpath/libcatcher_ffi.dylib`, copy Info.plist, and sign nested code before signing the app.

- [ ] **Step 3: Run automated verification**

  Run: `cargo fmt --check`

  Run: `cargo clippy --workspace --all-targets -- -D warnings`

  Run: `cargo test --workspace`

  Run: `swift test --package-path apps/tippi`

  Run: `apps/tippi/scripts/build-app.sh`

  Run: `codesign --verify --deep --strict apps/tippi/build/Tippi.app`

- [ ] **Step 4: Run real-model integration**

  Link or download the public artifact, feed `tests/fixtures/hello-streaming.wav` through the C ABI in irregular callback-sized blocks, and require the exact final text `Hello, this is a streaming speech recognition test`.

- [ ] **Step 5: Launch and document**

  Launch with `open apps/tippi/build/Tippi.app`, verify first-run download UI and recording toggle, then document model storage, microphone permission recovery, build/run commands, limitations, and the public artifact URL.

- [ ] **Step 6: Commit**

  Commit: `feat: ship the Tippi macOS app`

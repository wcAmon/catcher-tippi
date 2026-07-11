# Catcher + Tippi

**Catcher** is an end-to-end Rust speech-to-text runtime for
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
using MLX-C/Metal on Apple Silicon. **Tippi** is its native macOS SwiftUI app:
turn recording on, speak, watch partial text appear, and turn recording off to
flush the final transcript.

Catcher performs log-mel extraction, cache-aware 24-layer FastConformer
inference, language prompting, greedy RNNT decoding, and tokenizer decoding
without Python, PyTorch, NeMo, or ONNX Runtime. Rust owns the model topology,
caches, audio frontend, control flow, CLI, and C ABI; MLX-C/Metal executes the
accelerated tensor kernels.

## Requirements

- arm64 Apple Silicon Mac running macOS 15 or newer;
- Xcode 26 or newer and its Command Line Tools;
- Xcode Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`);
- Rust 1.85 or newer.

## Download the public INT8 model

The public Catcher artifact is approximately 629 MiB and retains the upstream
OpenMDW-1.1 license and NVIDIA origin notices:

```sh
hf download wcamon/catcher-asr-mlx-int8 \
  --local-dir catcher-asr-mlx-int8
```

Tippi performs this download automatically on first launch, verifies pinned
SHA-256 hashes, and installs the model atomically under its sandboxed Application
Support directory. The app contains no Hugging Face token.

## Build and run Tippi

```sh
apps/tippi/scripts/build-app.sh
open apps/tippi/build/Tippi.app
```

The build script compiles the release Catcher dylib, builds the Swift package,
creates the standard `.app` bundle, embeds the dylib, normalizes `@rpath`, adds
microphone/network sandbox entitlements, ad-hoc signs nested code and the app,
then verifies the complete bundle.

Inside Tippi:

1. Wait for the first-run model download and model-load status to reach Ready.
2. Select **Start Recording** and grant microphone permission when macOS asks.
3. Speak while partial text updates in the transcript area.
4. Select **Stop Recording** to flush and retain the final text.

If microphone access is denied, use Tippi's **Microphone Settings** action or
open System Settings → Privacy & Security → Microphone.

## Catcher CLI

```sh
cargo build -p nemotron-cli --release

target/release/catcher transcribe \
  --model catcher-asr-mlx-int8 \
  --audio speech.wav \
  --language en-US \
  --lookahead 3
```

The CLI accepts mono 16 kHz PCM or float WAV. Tippi accepts the Mac's native
microphone format and converts it to mono Float32 16 kHz with AVAudioConverter.
Supported Catcher lookahead values are `0`, `3`, `6`, and `13`; default `3`
corresponds to 320 ms algorithmic latency.

## Catcher C ABI

The canonical header is `crates/catcher-ffi/include/catcher.h`. A loaded handle
can be reused across utterances:

```c
catcher_handle_t *handle = catcher_create(model_path, "auto", 3);
catcher_start(handle);
catcher_push_audio(handle, samples, sample_count);
catcher_finish(handle);
const char *text = catcher_text(handle);
catcher_destroy(handle);
```

Calls are serialized per handle. Returned UTF-8 text is owned by Catcher and is
valid until the next mutating call. Every exported function validates pointers,
catches Rust panics, and never unwinds across C/Swift.

## Validation

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
swift test --package-path apps/tippi
apps/tippi/scripts/build-app.sh

apps/tippi/scripts/run-reference.sh /path/to/catcher-asr-mlx-int8
```

The real-model tests push the 4.151875-second reference WAV through one-shot,
irregular incremental Rust blocks, and the C ABI. All paths require the same
non-blank RNNT token IDs as the official Transformers reference and decode to:

> Hello, this is a streaming speech recognition test

The earlier CLI release measurement was 2.64 seconds (real-time factor 0.64)
with 702.5 MiB maximum resident memory on the development Apple Silicon Mac.
Measurements are hardware- and workload-specific.

## Current limitations

- Apple Silicon macOS only;
- greedy RNNT search with at most ten emitted symbols per encoder frame;
- one active utterance per Catcher handle;
- no global shortcut, text injection, transcript history, App Store signing, or
  notarization in the first Tippi release;
- the INT8 artifact has exact-token reference coverage but not yet a complete
  multilingual WER evaluation.

Model weights remain governed by OpenMDW-1.1. Catcher and Tippi are independent
community software and are not affiliated with or endorsed by NVIDIA.

See the [Catcher/Tippi design](docs/plans/2026-07-11-catcher-tippi-design.md) and
[implementation plan](docs/superpowers/plans/2026-07-11-catcher-tippi.md).

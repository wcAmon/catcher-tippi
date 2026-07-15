# Catcher + Tippi

**Catcher** is an end-to-end Rust speech-to-text runtime for
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
using MLX-C/Metal on Apple Silicon. **Tippi** is its native macOS SwiftUI app:
record speaker-attributed transcripts, or stream recognized text into the
frontmost app and say `幫我送出` to press Return.

Catcher performs log-mel extraction, cache-aware 24-layer FastConformer
inference, language prompting, greedy RNNT decoding, and tokenizer decoding
without Python, PyTorch, NeMo, or ONNX Runtime. Rust owns the model topology,
caches, audio frontend, control flow, CLI, and C ABI; MLX-C/Metal executes the
accelerated tensor kernels.

## Windows CPU app

A native Windows x64 WPF version now lives in `apps/tippi-windows`. It uses a
pinned INT4 ONNX conversion of the original NVIDIA Nemotron 3.5 ASR model and a
CPU-only ONNX Runtime GenAI build. Speaker attribution uses small Pyannote INT8
and NVIDIA TitaNet-S ONNX models through sherpa-onnx after recording stops, so
CUDA and a discrete GPU are not required.
The self-contained build includes .NET and the Visual C++ runtime:

```powershell
.\apps\tippi-windows\scripts\build.ps1
.\artifacts\Tippi-win-x64\Tippi.exe
```

See [`apps/tippi-windows/README.md`](apps/tippi-windows/README.md) for Windows
requirements, model storage, features, limitations, and tests. The original
MLX/Metal implementation and macOS app remain unchanged.

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
SHA-256 hashes, and installs the model atomically under its Application Support
directory. The app contains no Hugging Face token.

## Download the public diarization model

The public Catcher diarization artifact is approximately 121 MiB and retains
the upstream NVIDIA Open Model License and NVIDIA origin notices:

```sh
hf download wcamon/catcher-diar-mlx-int8 \
  --local-dir catcher-diar-mlx-int8
```

The diarization model is currently used by the `catcher diarize` CLI only;
Tippi app integration arrives in a later phase.

## Build and run Tippi

```sh
apps/tippi/scripts/build-app.sh
open apps/tippi/build/Tippi.app
```

The build script compiles the release Catcher dylib, builds the Swift package,
creates the standard `.app` bundle, embeds the dylib, normalizes `@rpath`, adds
microphone/network entitlements, ad-hoc signs nested code and the app, then
verifies the complete bundle. Tippi is intentionally not sandboxed because
cross-app text injection uses macOS Accessibility-authorized keyboard events.

On first launch, Tippi downloads both the ASR model
(`wcamon/catcher-asr-mlx-int8`, ≈628 MiB) and the diarization model
(`wcamon/catcher-diar-mlx-int8`, ≈121 MiB) behind a single merged progress
bar, verifies pinned SHA-256 hashes, and installs them atomically under its
Application Support directory. The app contains no Hugging Face token.

The window has two tabs: **轉錄** (Transcription), described below, and
**語音輸入** (Voice Input), which injects live speech recognition into the
currently focused control in another app.

Inside the **轉錄** tab:

1. Wait for the first-run model download and model-load status to reach Ready.
2. Select **Start Recording** and grant microphone permission when macOS asks.
3. Speak while the transcript renders as a speaker-attributed message list:
   each message is tagged with its speaker (unnamed speakers show as
   說話者 N), the most recent message grows in place as partial text arrives,
   and a new message starts whenever the speaker changes. Click a speaker's
   name (hover shows an underline and a tooltip) to rename them; the rename
   applies retroactively across the whole list.
4. Select **Stop Recording** to flush and retain the final text. Starting a
   new recording resets speaker names and messages.

Use **複製全部** to copy the whole transcript, or **匯出…** to save it as
either a UTF-8 `.txt` file or a `.json` file, picked by the extension chosen
in the save panel. The `.txt` format is the same line-per-message text, e.g.
`[03:24] 小明：今天先討論這個。`. The `.json` format is a
`{"messages": [...] }` document with snake_case keys — `speaker`, `name`,
`start_ms`, `end_ms`, `text`, `final` — where `name` is each message's
display name (renamed or the default 說話者 N). If the export write fails
(e.g. a read-only destination), Tippi shows a "匯出失敗" alert with the
underlying error instead of failing silently. Each message also offers
**複製此則** to copy just that line.

Once recording is stopped and there is at least one message, **清除** clears
all messages, speaker renames, and the diarizer warning banner after a
confirmation dialog ("清除全部訊息?"); it stays disabled while downloading,
loading, or recording, and while the transcript is empty.

If diarization hits a runtime error, Tippi shows a non-blocking banner
("說話者分離已暫停,文字繼續轉寫") and keeps transcribing; the next recording
automatically attempts to rebuild the diarizer (`catcher_start`'s rebuild
semantics — see the C ABI section below) and clears the banner on success.

If microphone access is denied, use Tippi's **Microphone Settings** action or
open System Settings → Privacy & Security → Microphone.

## Voice Input

Voice Input sends the same 16 kHz microphone stream to two local models. The
main Catcher ASR model produces the text to inject. A second, smaller offline
[`sherpa-onnx`](https://github.com/k2-fsa/sherpa-onnx) keyword-spotting model
detects the fixed `幫我送出` command without sending audio to a cloud service.
Tippi downloads and verifies the pinned keyword model when this tab is first
prepared. Existing valid installations update the generated command files in
place without downloading the model archive again.

To use it:

1. 打開「語音輸入」分頁，等待語音辨識與「幫我送出」口令模型就緒。
2. 授予 Tippi「系統設定 → 隱私權與安全性 → 輔助使用」權限。
3. 按「開始語音輸入」，切到目標 App 並點進輸入框。
4. 說完內容後短暫停頓（約 0.5 秒），再說「幫我送出」。
5. 口令不會進入輸入框；停止按鈕不會自動送出未完成內容。

- Live text is intentionally held for about 2 seconds before injection.
- After finishing the content, pause briefly (about 0.5 seconds), then say `幫我送出`.
- The pause keeps the final content outside the command safety window.
- If content runs directly into the command, Tippi prefers dropping a very short tail over leaking command words.
- sherpa-onnx timestamps are diagnostic-only; cutoff follows the shared 16 kHz sample clock.

Tippi injects text as Unicode keyboard events and presses Return exactly once
when the command is accepted; it does not use or replace the clipboard. Keep
the intended target app and input field focused while speaking. Version 1 does
not find or focus a text field automatically, and a command spoken while Tippi
itself is frontmost is discarded rather than deferred. Switch back to the
target input field and say 「幫我送出」 again.

Downloaded models live at:

```text
~/Library/Application Support/Tippi/Models
```

This directory contains the Catcher ASR, diarization, and sherpa-onnx keyword
model subdirectories. When upgrading from the earlier sandboxed build, Tippi
migrates an existing model directory from
`~/Library/Containers/com.wcamon.tippi/Data/Library/Application Support/Tippi/Models`
when the new destination is absent or empty, so the large Catcher models do not
need to be downloaded again. It does not overwrite a non-empty destination.

The **轉錄** and **語音輸入** modes share one microphone recorder and cannot
record at the same time. Voice Input is append-only within each spoken turn: it
does not move the target cursor, select text, or correct edits made in the
target field while recognition is active.

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

`catcher diarize` runs the Streaming Sortformer speaker diarizer over a WAV
file and prints a speaker timeline:

```sh
target/release/catcher diarize \
  --model catcher-diar-mlx-int8 \
  --audio meeting.wav
```

Pass `--diar-model` to `catcher transcribe` to fuse the two models in one
streaming pass and print who said what, instead of a flat transcript:

```sh
target/release/catcher transcribe \
  --model catcher-asr-mlx-int8 \
  --diar-model catcher-diar-mlx-int8 \
  --audio meeting.wav
```

```
[00:00] 說話者1：喂你好
[00:02] 說話者2：你好請問是哪位
[00:06] 說話者1：我是想跟您確認一下訂單的狀態
...
```

`--json` with `--diar-model` prints the same `SpeakerSegment` array shape as
the C ABI's `catcher_segments` (`speaker`, `start_ms`, `end_ms`, `text`,
`final`) instead of the flat-transcript JSON object used without
`--diar-model`.

All Chinese text produced anywhere in Catcher — the plain transcript, `--json`
output, and per-speaker segments alike — is normalized to Taiwan-standard
Traditional Chinese (OpenCC `s2twp`); simplified input is converted, and
already-Traditional input passes through unchanged. The streaming diarizer's
low-latency buffering (6-frame chunk + 7-frame right context at 80 ms/frame)
adds about 1.04 s of algorithmic latency before a frame's speaker label is
finalized; ASR tokens that arrive before that ride a tentative trailing
segment until enough diarization context lands (or `catcher_finish`/EOF
flushes it).

## Catcher C ABI

The canonical header is `crates/catcher-ffi/include/catcher.h`. A loaded handle
can be reused across utterances:

```c
catcher_handle_t *handle = catcher_create(model_path, NULL, "auto", 3);
catcher_start(handle);
catcher_push_audio(handle, samples, sample_count);
catcher_finish(handle);
const char *text = catcher_text(handle);
catcher_destroy(handle);
```

The second argument is an optional diarization model path; passing `NULL`
(as above) gives ASR-only transcription, while passing a Sortformer
diarization artifact directory instead enables speaker attribution and
populates `catcher_segments`.

Calls are serialized per handle. Returned UTF-8 text is owned by Catcher and is
valid until the next mutating call. Every exported function validates pointers,
catches Rust panics, and never unwinds across C/Swift. Two more accessors
follow the same borrowed-pointer lifetime as `catcher_text`:
`catcher_segments` returns the current speaker segments as a UTF-8 JSON array
(`"[]"` when no diarization model was supplied), and `catcher_warning`
returns a non-fatal diarization warning, or `NULL` when there is none.

If the diarizer degrades to disabled after a runtime error, `catcher_start`
attempts to rebuild it in place from the same `diar_model_path` given to
`catcher_create` before starting the next utterance: a successful rebuild
clears the warning and resumes diarization, while a failed rebuild is
non-fatal and leaves the handle in ASR-only mode with a new warning
describing the reload failure. `catcher_start` never fails outright for this
reason alone.

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
- no global shortcut, transcript history, custom voice command, command
  sensitivity UI, App Store signing, or notarization in the first Tippi
  release;
- Voice Input requires Accessibility permission, a compatible focused text
  control, and the fixed `幫我送出` command; it does not automatically focus or
  inspect the destination field;
- the INT8 artifact has exact-token reference coverage but not yet a complete
  multilingual WER evaluation.

The transcription model weights remain governed by OpenMDW-1.1 and the
diarization model weights by the NVIDIA Open Model License. Catcher and Tippi
are independent community software and are not affiliated with or endorsed by
NVIDIA.

See the [Catcher/Tippi design](docs/plans/2026-07-11-catcher-tippi-design.md) and
[implementation plan](docs/superpowers/plans/2026-07-11-catcher-tippi.md).

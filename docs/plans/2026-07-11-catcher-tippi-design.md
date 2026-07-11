# Catcher and Tippi Design

**Date:** 2026-07-11

## Goal

Turn the verified Rust + MLX-C Nemotron streaming ASR runtime into **Catcher**, a
reusable incremental speech-to-text library, and build **Tippi**, a native macOS
SwiftUI app with an explicit recording toggle and live partial transcription.

## Product boundary

Catcher owns model loading, the 16 kHz log-mel frontend, FastConformer and RNNT
caches, incremental token generation, tokenizer decoding, and final flushing.
Tippi owns model download, microphone permission, audio capture and conversion,
recording state, UI presentation, and persistence of the last transcript.

The first Tippi release is intentionally small. It does not inject text into
other apps, register a global shortcut, keep transcript history, or support
non-macOS platforms. The primary flow is:

1. On first launch, download the public Catcher artifact from
   `wcamon/catcher-asr-mlx-int8` into Application Support.
2. Verify the declared SHA-256 values before making the model active.
3. Load Catcher once and display a ready state.
4. When the user turns recording on, start a fresh streaming session.
5. Convert microphone audio to mono Float32 16 kHz and push it to Catcher.
6. Update the visible transcript whenever Catcher emits new non-blank tokens.
7. When recording turns off, flush the final padded chunk and display final text.

## Catcher architecture

`StreamingTranscriber` will become an incremental state machine rather than a
single consumed-buffer call. It maintains a pending audio buffer and exact model
frame cursor. The first inference begins after 4,040 samples; later inference
uses the checkpoint's 5,520-sample window and retained STFT overlap. `finish()`
pads only the final incomplete window, emits remaining tokens, and prevents
further pushes. The existing one-shot API becomes a convenience wrapper around
`push_samples()` plus `finish()`, preserving its exact-token regression test.

A new `catcher-ffi` crate exports an opaque session handle through a small C ABI:

- `catcher_create(model_path, language, lookahead)` loads one model/session.
- `catcher_start(handle)` clears streaming state for a new utterance.
- `catcher_push_audio(handle, samples, count)` processes available full chunks.
- `catcher_finish(handle)` flushes the final partial chunk.
- `catcher_text(handle)` returns the current UTF-8 transcript snapshot.
- `catcher_last_error(handle)` reports structured failures.
- `catcher_destroy(handle)` releases all Rust and MLX resources.

The ABI never transfers ownership of Swift buffers to Rust. Returned strings are
owned by the handle and remain valid until the next mutating call. Catcher calls
are serialized on Tippi's inference queue because one MLX session and its caches
are not concurrently mutable.

## Tippi architecture

Tippi is a sandbox-compatible SwiftUI macOS application. `ModelStore` downloads
required files with URLSession, publishes byte progress, writes into a temporary
directory, verifies CryptoKit SHA-256 values, and atomically promotes a complete
model into `~/Library/Application Support/Tippi/Models/catcher-asr-mlx-int8`.
Interrupted downloads can restart without exposing a partial model as ready.

`AudioRecorder` requests microphone permission and installs an AVAudioEngine
input tap. AVAudioConverter converts the hardware format (commonly 48 kHz) to
mono Float32 16 kHz. Converted buffers are handed to an actor-backed
`TranscriptionController`, which serializes C ABI calls off the main actor.

The main window has a title, status line, download progress when needed, a large
scrollable transcript, and one recording toggle. Recording states are explicit:
`modelMissing`, `downloading`, `loading`, `ready`, `recording`, `finishing`, and
`failed`. The toggle is disabled until Catcher is ready. Stopping recording does
not discard the transcript. Starting again clears it and starts new caches.

## Errors and recovery

Download, checksum, disk-space, model-load, microphone-permission, audio-format,
and inference failures are shown in the window with a retry action. Tippi never
embeds a Hugging Face token because the artifact is public. A failed checksum
deletes the temporary download. A denied microphone permission links to System
Settings. Catcher rejects null pointers, invalid UTF-8, pushes before start,
pushes after finish, and duplicate finish calls without unwinding across FFI.

## Testing

Rust unit tests cover incremental chunk thresholds, split-input equivalence,
finish padding, reset behavior, and C ABI lifecycle/error handling. The existing
real-checkpoint test must continue to produce token IDs identical to the official
Transformers reference.

Swift tests use protocol-backed download, checksum, audio, and Catcher clients.
They cover model promotion only after validation, recording-state transitions,
partial-text updates, finish behavior, and recoverable errors. A macOS integration
test loads the local artifact and feeds the existing reference WAV through the C
ABI. Final verification builds the release Rust library and Tippi app with
`xcodebuild`, runs both test suites, launches the app, and checks microphone UI
and transcript updates manually on the development Mac.

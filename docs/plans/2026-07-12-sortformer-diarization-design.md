# Tippi Speaker Diarization Design (Streaming Sortformer v2.1)

- Date: 2026-07-12
- Status: approved for planning
- Scope: speaker diarization via NVIDIA Streaming Sortformer v2.1 on MLX,
  speaker-segmented messages in Tippi, Simplified→Traditional Chinese output,
  speaker naming, transcript export and copy.

## As-built corrections (Phase 1)

The Phase 1 implementation revealed several checkpoint facts that differ from
assumptions elsewhere in this document:

- The mel frontend uses **128 mel features**, not 80.
- The exported preprocessor config is `normalize: "NA"` — the pipeline
  consumes raw (unnormalized) log-mel frames, not per-feature-normalized ones.
- `xscaling: true`: the encoder multiplies subsampled features by `sqrt(512)`
  after `pre_encode`, before the first Conformer block.
- The mel window function is a Hann window built with the symmetric
  (`periodic=False`) convention, matching `torch.hann_window`.
- Tensor storage is dispatched per-tensor between INT8 (affine, group size
  128) and F16 based on shape, not by module: the 192-wide Transformer stack
  stays F16 because its input dimension (192) is not a multiple of the
  group size — this is a divisibility constraint, not an accuracy fallback.
- `sortformer-mlx` implements its own mel frontend and Conformer encoder
  layers; it only reuses `nemotron-mlx` primitives (quantized matmul,
  artifact I/O, tensor helpers), not `nemotron-mlx`'s ASR-specific encoder.

## Goals

1. Live speaker segmentation while recording: the transcript renders as
   per-speaker messages that grow in real time, not one text blob.
2. Diarization runs locally on MLX-C/Metal, converted from
   `nvidia/diar_streaming_sortformer_4spk-v2.1` (no Python, PyTorch, NeMo, or
   ONNX Runtime at user runtime).
3. All Chinese output is Traditional Chinese using OpenCC **s2twp**
   (Taiwan standard glyphs plus Taiwan phrase conversion, e.g. 软件→軟體,
   信息→資訊).
4. Speakers can be renamed within a recording session; renames apply to all of
   that speaker's messages immediately. Names reset on the next recording.
   No cross-session voice re-identification (out of scope; a future speaker
   embedding model could add it).
5. The transcript can be copied to the clipboard and exported to a `.txt`
   file. Both include speaker names and `[mm:ss]` start timestamps.

## Non-goals

- More than 4 concurrent speakers (Sortformer architectural limit).
- Cross-recording speaker identity (no voice fingerprinting).
- Bundling models in the app: both models remain first-launch downloads with
  pinned SHA-256 hashes; app bundle size is unchanged.

## Key facts driving the design

- Sortformer v2.1: 17-layer NEST (Fast-Conformer) encoder + 18-layer
  Transformer (hidden 192), 117M parameters, sigmoid output of 4 speaker
  activation probabilities per **80 ms frame**. Streaming uses AOSC
  (Arrival-Order Speaker Cache) + FIFO; low-latency setting (chunk 6 frames,
  right context 7) gives ~1.04 s algorithmic latency.
- Nemotron ASR frames are also 80 ms, so ASR token timestamps and diarization
  frames align 1:1 with no resampling.
- The v2.1 Hugging Face repo ships only a `.nemo` checkpoint (no safetensors),
  so conversion needs a one-time developer-side Python export step, following
  the existing `tools/*.py` reference-script pattern.
- Quantization: INT8 affine group-128, same machinery as the ASR artifact
  (`Int8Affine`, `quantized_matmul`). ~125 MB download versus ~470 MB F32.
  Fall back to FP16 (~235 MB) only if INT8 measurably degrades diarization
  against the NeMo reference.
- License: NVIDIA Open Model License; the published artifact keeps the license
  and origin notices, matching the ASR artifact's treatment of OpenMDW-1.1.

## Architecture (approved: fusion in Rust)

Rust owns both engines and the fusion logic; Swift stays a thin native UI.
Alternative (fusion in Swift with two independent FFIs) was rejected: the
fusion state machine is the hardest part to test and does not belong on the
MainActor.

```
tools/export_sortformer_weights.py   .nemo → f32 safetensors (dev machine, one-time)
crates/sortformer-mlx                Sortformer inference engine
  ├─ reuses nemotron-mlx MLX-C backend, log-mel frontend, Fast-Conformer layers
  ├─ model/   17-layer NEST encoder + 18-layer Transformer + 4-way sigmoid head
  ├─ stream/  AOSC + FIFO streaming state, low-latency preset
  └─ weights/ f32 safetensors → MLX INT8 artifact conversion
crates/sortformer-convert            CLI shell (mirrors nemotron-convert)
crates/nemotron-mlx (extended)
  ├─ StreamingTranscriber emits (token id, frame index) instead of token ids
  ├─ fusion/  token timestamps × speaker probabilities → SpeakerSegment stream
  └─ opencc/  s2twp conversion, dictionaries embedded in the binary
crates/catcher-ffi                   C ABI v2 (see below)
```

Published artifact: `wcamon/catcher-diar-mlx-int8` on Hugging Face with pinned
SHA-256 manifest, LICENSE, and NOTICE files.

## Fusion algorithm (Rust `fusion` module)

Inputs: ASR `(token, frame)` stream and Sortformer `P(speaker s | frame t)`
stream, both on the 80 ms frame grid.

1. For each token, look at a ±3 frame window around its emission frame, smooth
   the four speaker probabilities, and attribute the token to the most active
   speaker above the activity threshold. If no speaker is active (silence,
   overlap ambiguity), inherit the previous token's speaker.
2. Group consecutive same-speaker tokens into
   `SpeakerSegment { speaker: u8, start_ms: u64, end_ms: u64, text: String, is_final: bool }`.
3. Anti-flicker: a speaker turn shorter than 0.5 s merges into its neighbor.
4. Latency handling: diarization trails ASR by ~1 s. Tokens newer than the
   diarization horizon stay in a tentative tail segment (current speaker,
   `is_final = false`). Segments finalize once diarization catches up; stopping
   the recording flushes and finalizes everything.
5. s2twp conversion applies when segment text is produced. The tentative tail
   is reconverted in full on each update (conversion is idempotent and cheap).
6. Degradation: if the diarization engine fails at runtime, transcription
   continues, all text attributes to speaker 0, and the FFI surfaces a warning
   state so the UI can show it.

## C ABI v2 (`crates/catcher-ffi/include/catcher.h`)

```c
catcher_handle_t *catcher_create(const char *asr_model_path,
                                 const char *diar_model_path,   /* NULL = ASR only */
                                 const char *language,
                                 uint32_t lookahead);
/* catcher_start / catcher_push_audio / catcher_finish / catcher_text unchanged */
const char *catcher_segments(const catcher_handle_t *h);  /* UTF-8 JSON array */
```

`catcher_segments` returns
`[{"speaker":0,"start_ms":1040,"end_ms":5200,"text":"…","final":true},…]`,
following the existing return-a-C-string ABI style; Swift decodes with
`Codable`. Passing `diar_model_path = NULL` preserves current single-stream
behavior and is the degradation path. `catcher_text` keeps returning the plain
concatenated transcript (now s2twp-converted) so the CLI and existing tests
stay meaningful.

## Swift side

- **ModelStore**: generalize to install multiple artifacts (the `files:`
  injection point already exists; add a second pinned manifest constant for
  the diarization artifact). Download progress merges by byte count across
  both artifacts.
- **TranscriptionController**: replace `text: String` with
  `messages: [Message]` where
  `Message { speaker: Int, start: Duration, text: String, isFinal: Bool }`,
  driven by polling `catcher_segments` on each push result. Speaker naming is
  a `[Int: String]` dictionary; unnamed speakers display as 說話者 N. Renames
  re-render all messages immediately and reset when a new recording starts.
- **ContentView**: the transcript pane becomes a message list. Each message
  shows a speaker name (click to rename via popover text field), a `[mm:ss]`
  start timestamp, and the text. Each speaker gets a stable accent color. The
  last bubble grows live; a speaker change opens a new bubble.
- **Export and copy**: toolbar gains "複製全部" (clipboard) and "匯出…"
  (`NSSavePanel`, satisfying the sandbox via user-selected file write, saved
  as UTF-8 `.txt`). Per-message copy via context menu. Line format:
  `[03:24] 小明：今天先討論這個。` — identical for copy and export.

## Testing

- **Conversion**: golden manifest tests (tensor names, shapes, quantization
  layout) mirroring `weight_manifest.rs`; the Python export script's output
  inventory is locked by fixture.
- **sortformer-mlx numerics**: a `tools/` reference script generates NeMo
  frame-level probabilities for fixture WAVs; Rust tests assert per-frame
  parity within tolerance for the INT8 artifact. Streaming equivalence tests
  (chunked ≡ offline under the same cache config) mirror
  `incremental_stream.rs`.
- **fusion**: pure unit tests over synthetic probability grids and token
  timings — attribution, inheritance on silence, anti-flicker merging,
  tentative→final transitions, flush on stop.
- **opencc**: fixture conversions (软件→軟體, 信息→資訊, 里面→裡面, 鼠标→滑鼠)
  plus idempotency on already-Traditional text.
- **Swift**: controller tests with a fake segment stream covering message
  state transitions, renaming, and reset; export/copy formatting tests.
- **FFI**: lifecycle test extended for dual-model create, NULL diar path, and
  segments JSON decoding.

## Delivery phases (each gets its own implementation plan)

1. **Engine**: `tools/export_sortformer_weights.py`, `sortformer-mlx` offline
   inference passing numeric parity, `sortformer-convert` CLI, published
   `wcamon/catcher-diar-mlx-int8` artifact. Verifiable: CLI prints a speaker
   activity timeline for a WAV.
2. **Streaming + fusion**: AOSC streaming, token timestamps through
   `StreamingTranscriber`, fusion module, s2twp, FFI v2. Verifiable: CLI
   prints who-said-what-when in Traditional Chinese for a WAV.
3. **App**: dual-artifact ModelStore, message-list UI, speaker naming,
   export/copy. Verifiable: full Tippi experience end to end.

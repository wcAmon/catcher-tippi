# Nemotron 3.5 ASR Rust + MLX-C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-Python-runtime Rust inference engine that runs Nemotron 3.5 ASR streaming inference on Apple Silicon through MLX-C Metal kernels with weight-only INT8 quantization.

**Architecture:** Rust owns audio preprocessing, model topology, streaming caches, RNNT control flow, and the CLI. A narrow backend module wraps MLX-C through `mlx-rs` initially; quantized linear and reshaped 1x1 convolution weights use affine INT8 group-128 MLX matmul, while numerically sensitive and unsupported operations remain FP16/FP32.

**Tech Stack:** Rust 1.95, Cargo, `mlx-rs`/MLX-C/MLX Metal, safetensors, serde, clap, hound, rustfft, rubato, thiserror, approx.

## Global Constraints

- Runtime must not require Python, PyTorch, NeMo, or ONNX Runtime.
- Target is arm64 macOS on Apple Silicon using MLX Metal acceleration.
- Model weights remain external files and must retain OpenMDW-1.1 notices when redistributed.
- Quantized weights use affine INT8 with group size 128; FP16/FP32 exceptions must be explicit in the manifest.
- Production code is added only after a directly relevant test has failed for the expected missing behavior.
- First release uses RNNT greedy search and supports a maximum of ten emitted symbols per encoder frame.

---

### Task 1: Cargo workspace and MLX INT8 proof

**Files:**
- Create: `Cargo.toml`
- Create: `crates/nemotron-mlx/Cargo.toml`
- Create: `crates/nemotron-mlx/src/lib.rs`
- Create: `crates/nemotron-mlx/src/backend/mod.rs`
- Create: `crates/nemotron-mlx/tests/quantized_matmul.rs`

**Interfaces:**
- Produces: `backend::is_metal_available() -> Result<bool>` and `backend::quantized_linear_reference(input, weight, group_size) -> Result<Vec<f32>>`.
- Produces: an MLX-backed INT8 matmul proof that later `QuantizedLinear` uses.

- [x] Write a test with a deterministic `[2, 128]` input and `[64, 128]` weight matrix that expects Metal availability and output within an explicit tolerance of an FP32 Rust reference.
- [x] Run `cargo test -p nemotron-mlx --test quantized_matmul` and verify it fails because the backend API does not exist.
- [x] Add the minimal workspace, backend error type, MLX device check, affine INT8 quantization call, and quantized matmul call.
- [x] Re-run the test and require a passing numerical comparison with no warnings.
- [x] Commit the workspace and proof as `feat: prove MLX INT8 matmul from Rust`.

### Task 2: Weight manifest and validation

**Files:**
- Create: `crates/nemotron-mlx/src/weights/mod.rs`
- Create: `crates/nemotron-mlx/src/weights/manifest.rs`
- Create: `crates/nemotron-mlx/tests/weight_manifest.rs`

**Interfaces:**
- Produces: `TensorSpec { name, shape, storage }`, `Storage::{Int8Affine { group_size }, F16, F32, I32}`.
- Produces: `ModelManifest::nemotron_3_5()` and `ModelManifest::validate(&TensorIndex)`.

- [x] Write failing tests for the configured model dimensions, group-size divisibility, a missing tensor, and an incorrect tensor shape.
- [x] Run `cargo test -p nemotron-mlx --test weight_manifest` and verify failures are caused by missing manifest types.
- [x] Implement the manifest and validation errors with exact tensor names derived from the Hugging Face config and safetensors index.
- [x] Run the manifest test and the full crate tests until all pass without warnings.
- [x] Commit as `feat: validate Nemotron weight manifests`.

### Task 3: Rust weight converter and MLX artifact loader

**Files:**
- Create: `crates/nemotron-convert/Cargo.toml`
- Create: `crates/nemotron-convert/src/main.rs`
- Create: `crates/nemotron-mlx/src/weights/convert.rs`
- Create: `crates/nemotron-mlx/src/weights/load.rs`
- Create: `crates/nemotron-mlx/tests/weight_roundtrip.rs`

**Interfaces:**
- Consumes: `ModelManifest`, source safetensors tensor views.
- Produces: `convert_model(source, destination, QuantizationConfig { bits: 8, group_size: 128 })`.
- Produces: an artifact containing packed weights, FP16 scales, FP16 affine biases, FP16 exceptions, tokenizer/config files, and `manifest.json`.

- [x] Write a failing round-trip test using a small deterministic safetensors fixture with one matrix and one depthwise convolution.
- [x] Verify the test fails because converter and loader APIs are missing.
- [x] Implement checked safetensors reading, MLX INT8 packing, FP16 exception conversion, atomic artifact writing, and artifact loading.
- [x] Verify reconstructed fixture values meet the declared tolerance and corrupt manifests are rejected.
- [x] Commit as `feat: convert safetensors to MLX INT8 artifacts`.

### Task 4: Audio frontend

**Files:**
- Create: `crates/nemotron-mlx/src/audio/mod.rs`
- Create: `crates/nemotron-mlx/src/audio/log_mel.rs`
- Create: `crates/nemotron-mlx/tests/log_mel.rs`
- Create: `tests/fixtures/log_mel_reference.json`

**Interfaces:**
- Produces: `LogMelFrontend::nemotron()` and explicit `LogMelFrontend::new(...)`.
- Produces: `LogMelFrontend::extract(&[f32], center: bool) -> Vec<Vec<f32>>`; the streaming scheduler supplies the documented overlap for subsequent uncentered chunks.

- [x] Add failing tests for frame count, silence output, chunk-boundary equivalence, and the official feature-extractor reference fixture.
- [x] Verify the tests fail because the frontend is absent.
- [x] Implement centered/streaming STFT buffering, Slaney mel filters, log guard, and exact frame accounting in Rust.
- [x] Verify chunked and one-shot features match within tolerance.
- [x] Commit as `feat: add streaming Nemotron log-mel frontend`.

### Task 5: Quantized model primitives

**Files:**
- Create: `crates/nemotron-mlx/src/model/mod.rs`
- Create: `crates/nemotron-mlx/src/model/layers.rs`
- Create: `crates/nemotron-mlx/src/model/cache.rs`
- Create: `crates/nemotron-mlx/tests/model_layers.rs`

**Interfaces:**
- Produces: `QuantizedLinear`, `PointwiseConv1d`, `DepthwiseConv1d`, `LayerNorm`, `Tensor3`, and `CausalConv1dCache`; Conv2D subsampling remains part of Task 6 because its cache shape is encoder-specific.
- Consumes: arrays loaded by Task 3 and delegates operations only through the backend module.

- [x] Write failing deterministic tests for each primitive, including cache update behavior across two chunks.
- [x] Verify failures are caused by missing layer types.
- [x] Implement minimal MLX-backed primitives, using INT8 matmul for matrix/pointwise weights and FP16 convolution for depthwise weights.
- [x] Compare all primitive outputs to stored FP32 reference vectors.
- [x] Commit as `feat: add MLX Nemotron model primitives`.

### Task 6: Cache-aware FastConformer encoder and language prompt

**Files:**
- Create: `crates/nemotron-mlx/src/model/encoder.rs`
- Create: `crates/nemotron-mlx/src/model/prompt.rs`
- Create: `crates/nemotron-mlx/tests/encoder_streaming.rs`
- Create: `tests/fixtures/encoder_reference.json`

**Interfaces:**
- Produces: `StreamingEncoder::encode_chunk(features, prompt, &mut EncoderCache)`.
- Produces: `LanguagePrompt::from_code(&str)` with `auto`, `zh-CN`, and all model-card locale aliases.

- [x] Add failing tests for prompt mapping, first/subsequent chunk shapes, cache lengths, and reference encoder output.
- [x] Verify failures are caused by missing encoder behavior.
- [x] Implement Conv2D subsampling, 24 FastConformer blocks, relative-position attention, convolution modules, left/right context masking, and prompt projection.
- [x] Verify first and subsequent chunk outputs against the official reference at valid positions within the declared INT8 tolerances.
- [x] Commit the initial cache/subsampling work as `feat: add FastConformer streaming caches and subsampling`; complete the encoder in the release commit.

### Task 7: RNNT prediction, joint network, and greedy decoder

**Files:**
- Create: `crates/nemotron-mlx/src/model/rnnt.rs`
- Create: `crates/nemotron-mlx/src/tokenizer.rs`
- Create: `crates/nemotron-mlx/tests/rnnt_decode.rs`

**Interfaces:**
- Produces: `RnntDecoder::decode_frames(encoded, &mut RnntState) -> Vec<u32>`.
- Produces: `Tokenizer::decode(&[u32], strip_language_tag) -> String`.

- [x] Write failing tests for blank advancement, non-blank LSTM updates, ten-symbol frame limit, language-tag stripping, and a reference token sequence.
- [x] Verify the failures identify the absent decoder API.
- [x] Implement the two-layer LSTM gates, joint ReLU/head, greedy control flow, and tokenizer decode.
- [x] Verify deterministic token IDs and state tensors match reference fixtures.
- [x] Commit as `feat: implement RNNT decoding`.

### Task 8: CLI, end-to-end validation, and release build

**Files:**
- Create: `crates/nemotron-cli/Cargo.toml`
- Create: `crates/nemotron-cli/src/main.rs`
- Create: `crates/nemotron-mlx/tests/end_to_end.rs`
- Create: `README.md`
- Modify: `Cargo.toml`

**Interfaces:**
- Produces: `nemotron-mlx transcribe --model <dir> --audio <wav> --language <code> --lookahead <0|3|6|13>`.
- Produces: final JSON or human-readable text output.

- [x] Write a failing CLI integration test for argument validation and a gated end-to-end test for a downloaded model fixture.
- [x] Verify the CLI test fails because the binary does not exist.
- [x] Implement WAV input, exact streaming scheduling, final text/JSON output, structured errors, and tokenizer discovery.
- [x] Run `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and a release transcription benchmark on Apple Silicon.
- [x] Build with LTO, `panic = "abort"`, and symbol stripping; report executable, model, peak-memory, latency, and real-time-factor measurements.
- [x] Commit as `feat: ship Nemotron MLX streaming CLI`.

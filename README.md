# nemotron-mlx

Rust host runtime for `nvidia/nemotron-3.5-asr-streaming-0.6b`, using MLX-C/Metal on Apple Silicon.

## Current status

Implemented and tested:

- affine weight-only INT8 (`group_size=128`) MLX matmul;
- exact 655-tensor checkpoint manifest and atomic safetensors conversion;
- FP16 storage for depthwise/subsampling convolution, norm, and bias tensors;
- 16 kHz preemphasis, centered/streaming STFT, and 128-bin Slaney log-mel frontend;
- quantized linear and pointwise convolution, FP16 depthwise convolution, and LayerNorm;
- all 121 language-prompt aliases and the two-layer prompt projector;
- checkpoint-accurate streaming chunk plans for lookahead 0, 3, 6, and 13;
- RNNT embedding, two-layer LSTM prediction cache, encoder/joint projectors, blank control, greedy search, and minimal BPE Metaspace decoding.

The cache-aware 24-layer FastConformer encoder and final WAV transcription CLI are still in development. The repository does not yet claim end-to-end transcription correctness.

## Requirements

- arm64 Apple Silicon Mac;
- Xcode Command Line Tools;
- Xcode Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`);
- Rust 1.95 or newer.

No Python, PyTorch, NeMo, ONNX Runtime, or Python MLX package is required at runtime. `mlx-rs` is used as a narrow Rust wrapper over MLX-C; MLX itself contains C++ and Metal kernels.

## Build and test

```sh
cargo test --workspace
cargo build --workspace --release
```

The stripped release converter currently measures about 9.5 MiB on arm64 macOS and dynamically uses only Apple/system libraries (`Metal`, `Accelerate`, `Foundation`, libc++, libobjc, and libSystem).

## Convert weights

Download the original `model.safetensors`, retain the model's OpenMDW-1.1 license and notices, then run:

```sh
target/release/nemotron-convert \
  --source /path/to/model.safetensors \
  --output /path/to/nemotron-mlx-int8
```

The output contains packed MLX INT8 matrices, FP16 scale/affine metadata, FP16 exception tensors, and `manifest.json`. Tokenizer/config/license copying will be added with the end-to-end CLI milestone.

## Important checkpoint details

- Model blank ID: `13087` (use this for RNNT frame advancement).
- Prompt IDs: `zh-CN=4`, `zh-TW=5`, `auto=101`.
- Default lookahead: 3 subsampled frames (320 ms algorithmic latency).
- First default chunk: 25 mel frames / 4040 samples.
- Subsequent default chunks: 32 mel frames / 5520 samples, including the required STFT overlap.
- The `tokenizer.json` added-token entry for `<blank>` is 13088, outside the model's 0..13087 logits; the runtime deliberately follows `config.json` and filters model blank ID 13087.

See [the implementation plan](docs/superpowers/plans/2026-07-10-nemotron-mlx.md) and [design notes](docs/plans/2026-07-10-nemotron-mlx-design.md).

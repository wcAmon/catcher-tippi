# nemotron-mlx

An end-to-end Rust host runtime for
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
using MLX-C/Metal on Apple Silicon.

The runtime performs WAV decoding, log-mel extraction, cache-aware 24-layer
FastConformer inference, language prompting, greedy RNNT decoding, and tokenizer
decoding without Python, PyTorch, NeMo, or ONNX Runtime. Matrix and pointwise
weights use affine weight-only INT8; numerically sensitive convolution, norm,
bias, and scale tensors remain FP16/FP32 as declared by the artifact manifest.

## Requirements

- arm64 Apple Silicon Mac;
- Xcode Command Line Tools;
- Xcode Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`);
- Rust 1.85 or newer.

`mlx-rs` is used as a narrow Rust wrapper over MLX-C. MLX itself contains C++
and Metal kernels, but the inference topology, cache control, audio frontend,
RNNT loop, and CLI are Rust.

## Build

```sh
cargo build --workspace --release
```

## Convert the official checkpoint

Download the Hugging Face repository, retain its OpenMDW-1.1 license and
notices, then convert `model.safetensors`:

```sh
target/release/nemotron-convert \
  --source /path/to/model.safetensors \
  --output /path/to/nemotron-mlx-int8 \
  --group-size 128
```

The converter accepts group sizes 32, 64, and 128. Group 128 is the default and
smallest supported artifact. Recognized tokenizer, configuration, README, and
license companion files are copied from the checkpoint directory into the
artifact. The tested group-128 artifact is 629 MiB on disk (659.6 MB decimal).

## Transcribe

The first CLI release accepts mono 16 kHz PCM or float WAV files:

```sh
target/release/nemotron-mlx transcribe \
  --model /path/to/nemotron-mlx-int8 \
  --audio /path/to/audio.wav \
  --language en-US \
  --lookahead 3
```

Use `--json` for a final JSON object containing text, token IDs, language, and
lookahead. Supported checkpoint lookahead values are `0`, `3`, `6`, and `13`.
The default `3` corresponds to 320 ms algorithmic latency.

## Validation

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

NEMOTRON_MLX_ARTIFACT=/path/to/nemotron-mlx-int8 \
  cargo test -p nemotron-mlx --test real_checkpoint -- \
  --ignored --test-threads=1
NEMOTRON_MLX_ARTIFACT=/path/to/nemotron-mlx-int8 \
  cargo test -p nemotron-cli --test cli -- --ignored --test-threads=1
```

The gated end-to-end fixture compares Rust streaming inference against the
official Transformers implementation and requires identical non-blank token
IDs. The reference WAV decodes to:

> Hello, this is a streaming speech recognition test

On the development Apple Silicon Mac, the stripped 10.3 MiB CLI transcribed the
4.151875-second fixture in 2.64 seconds (real-time factor 0.64) with 702.5 MiB
maximum resident memory. These are single-run local measurements, not a general
hardware guarantee. The stripped converter is 9.6 MiB.

## Current limitations

- Apple Silicon macOS only;
- mono 16 kHz WAV input only; no resampling or microphone capture yet;
- greedy RNNT search with at most ten emitted symbols per encoder frame;
- one complete utterance per `StreamingTranscriber` session;
- final output only; incremental partial-result callbacks are not exposed yet;
- model weights are external and remain governed by NVIDIA's OpenMDW-1.1 terms.

Important checkpoint constants: model blank ID `13087`; prompt IDs `zh-CN=4`,
`zh-TW=5`, and `auto=101`. The tokenizer's added `<blank>` entry is 13088,
outside the model logits, so the runtime deliberately follows `config.json` and
filters model blank ID 13087.

See [the implementation plan](docs/superpowers/plans/2026-07-10-nemotron-mlx.md)
and [design notes](docs/plans/2026-07-10-nemotron-mlx-design.md).

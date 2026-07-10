# Nemotron 3.5 ASR Rust + MLX-C Design

## Objective

Build a macOS Apple Silicon streaming inference engine for
`nvidia/nemotron-3.5-asr-streaming-0.6b` with no Python, PyTorch, NeMo, or ONNX
Runtime dependency at runtime. Rust owns audio processing, model topology,
stream state, RNNT decoding, error handling, and the CLI. MLX-C supplies the
official MLX array and Metal execution API.

## Precision and size policy

The release format uses affine weight-only INT8 with a group size of 128 and
FP16 scales/biases. Activations, normalization, attention scores, convolution
state, and recurrent state remain FP16 or FP32 where numerically required.
Linear layers, attention projections, feed-forward layers, prompt projection,
embedding/output matrices, and 1x1 pointwise convolutions use MLX
`quantized_matmul`. The small depthwise and subsampling convolutions remain
FP16. This quantizes approximately 99.9% of model weights while preserving the
operators for which MLX does not expose an INT8 convolution path.

The expected artifact is approximately 660 MB: about 637 MB of packed INT8
weights, about 20 MB of group scales/biases, and a few MB of FP16 weights and
metadata. The model remains an external artifact rather than being embedded in
the executable.

## Architecture

The project is a Cargo workspace with a safe MLX-facing library and a CLI.
During the first implementation phase it uses the maintained `mlx-rs` wrapper,
which builds against MLX-C. The model-facing interfaces do not expose
`mlx-rs` types outside the backend module, allowing a later replacement with a
smaller generated binding to the official MLX-C headers without changing the
model code.

Audio is decoded and resampled to mono 16 kHz in Rust. A Rust log-mel frontend
produces 128-bin features. The MLX backend runs causal Conv2D subsampling,
24 cache-aware FastConformer blocks, the language prompt projection, the RNNT
prediction network, and the joint network. Rust stores the attention,
convolution, and LSTM cache ownership and performs greedy RNNT control flow.

## Validation

Each layer is developed test-first. CPU reference calculations validate
quantized matrix multiplication and frontend features. Model conversion checks
every expected tensor name, shape, dtype, and byte count before writing an
artifact. Layer outputs are compared with reference vectors exported from the
official Transformers implementation. End-to-end tests compare transcripts
for offline and chunked streaming inference, followed by Mandarin CER and
real-time-factor measurements on the target Mac.

## Explicit non-goals for the first usable release

- Training or fine-tuning.
- Full-integer activations.
- Speaker diarization, timestamps, or endpoint detection.
- Intel Mac, iOS, Linux, Windows, or non-Apple GPU support.
- Beam search; the first decoder is RNNT greedy search with the model's
  ten-symbol-per-frame limit.


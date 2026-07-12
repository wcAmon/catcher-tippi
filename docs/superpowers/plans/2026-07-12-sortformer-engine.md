# Sortformer Engine (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert NVIDIA Streaming Sortformer v2.1 to an MLX INT8 artifact and implement offline diarization inference in Rust that matches NeMo reference outputs, exposed through a `catcher diarize` CLI.

**Architecture:** A one-time Python tool unpacks the `.nemo` checkpoint into F32 safetensors plus ground-truth `config.json`/`inventory.json` fixtures. A new `sortformer-mlx` crate builds its conversion plan from the inventory (reusing `nemotron_mlx::weights::convert_tensors`), then implements the full-context forward pass: NeMo mel frontend → 17-layer NEST Fast-Conformer encoder → 512→192 projection → 18-layer Transformer (no positional embeddings) → 4-way sigmoid head. Every model stage is gated by numeric parity tests against NeMo-generated reference checkpoints.

**Tech Stack:** Rust 2024 (1.85+), mlx-rs 0.25.3, rustfft, serde; Python (dev-only): torch, safetensors, pyyaml, nemo_toolkit (reference generation only).

## Global Constraints

- Workspace: edition `2024`, `rust-version = "1.85"`, license `Apache-2.0 OR MIT` (crate code; the model artifact itself carries the NVIDIA Open Model License).
- Source model: `nvidia/diar_streaming_sortformer_4spk-v2.1` (117M params, 4 speakers, 80 ms output frames, 16 kHz input).
- Quantization: MLX affine INT8, group size 128, format version 1 — reuse `nemotron_mlx::weights` machinery; do NOT fork it.
- Published artifact name: `wcamon/catcher-diar-mlx-int8` (Hugging Face), NVIDIA license and notices preserved.
- No Python/PyTorch/NeMo/ONNX at user runtime; Python only in `tools/` on the developer machine.
- All commits: imperative conventional-commit subject, and end the message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Fixture files live in top-level `tests/fixtures/` (existing repo convention); crate tests reach them via `concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/…")`.
- Ground-truth rule: tensor names, shapes, and config values come from the exported `inventory.json`/`config.json`, never from guesses. If a name in this plan disagrees with the exported inventory, the inventory wins — update the code and fixtures, not the checkpoint.

## As-built corrections — the fixtures override the 80-mel/normalized-frontend expectations below

- The mel frontend uses 128 mel features, not 80.
- `normalize: "NA"` in the exported preprocessor config: the pipeline consumes raw log-mel frames, not per-feature-normalized ones.
- `xscaling: true`: subsampled features are multiplied by `sqrt(512)` after `pre_encode`, before the first Conformer block.
- The Hann window is built with the symmetric (`periodic=False`) convention.
- Storage is dispatched per-tensor (INT8 group-128 vs. F16) by shape; the 192-wide Transformer stack stays F16 because 192 is not a multiple of 128, not as an accuracy fallback.
- `sortformer-mlx` implements its own mel frontend and Conformer layers, reusing only `nemotron-mlx` primitives (not its ASR encoder).

---

### Task 1: Sortformer weight export tool

**Files:**
- Create: `tools/export_sortformer_weights.py`

**Interfaces:**
- Produces: `OUTPUT_DIR/model.safetensors` (all F32 tensors, original NeMo names), `OUTPUT_DIR/config.json` (raw `model_config.yaml` converted to JSON), `OUTPUT_DIR/inventory.json` (`{tensor_name: [dims…]}`). Tasks 2, 4, 5 consume these.

- [ ] **Step 1: Write the export script**

```python
"""Unpack a NeMo Sortformer checkpoint into F32 safetensors plus metadata.

Usage: python tools/export_sortformer_weights.py MODEL.nemo OUTPUT_DIR
Requires: pip install torch safetensors pyyaml   (no NeMo needed)
"""

import json
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

SKIP_SUFFIXES = ("num_batches_tracked",)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_sortformer_weights.py MODEL.nemo OUTPUT_DIR")
    nemo_path = Path(sys.argv[1])
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(nemo_path) as tar:
            tar.extractall(tmp, filter="data")
        tmp_path = Path(tmp)
        config_path = next(tmp_path.rglob("model_config.yaml"))
        ckpt_path = next(tmp_path.rglob("*.ckpt"))
        config = yaml.safe_load(config_path.read_text())
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    tensors: dict[str, torch.Tensor] = {}
    inventory: dict[str, list[int]] = {}
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor) or name.endswith(SKIP_SUFFIXES):
            continue
        converted = tensor.detach().to(torch.float32).contiguous()
        tensors[name] = converted
        inventory[name] = list(converted.shape)

    save_file(tensors, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    parameters = sum(t.numel() for t in tensors.values())
    print(f"exported {len(tensors)} tensors, {parameters:,} parameters")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Download the checkpoint and run the export**

```bash
hf download nvidia/diar_streaming_sortformer_4spk-v2.1 \
  diar_streaming_sortformer_4spk-v2.1.nemo --local-dir /tmp/sortformer-src
python3 -m venv /tmp/sortformer-venv && /tmp/sortformer-venv/bin/pip install torch safetensors pyyaml
/tmp/sortformer-venv/bin/python tools/export_sortformer_weights.py \
  /tmp/sortformer-src/diar_streaming_sortformer_4spk-v2.1.nemo /tmp/sortformer-f32
```

Expected: prints `exported N tensors, ~117,000,000 parameters` (parameter count within 116M–119M). If the count is far off, the checkpoint layout differs from expectations — stop and inspect `inventory.json` before continuing.

- [ ] **Step 3: Check the metadata fixtures into the repo**

```bash
cp /tmp/sortformer-f32/config.json tests/fixtures/sortformer_config.json
cp /tmp/sortformer-f32/inventory.json tests/fixtures/sortformer_inventory.json
```

Inspect `tests/fixtures/sortformer_config.json` and record (in the commit message body) the actual values of: `preprocessor.features` (expect 80), `preprocessor.window_stride` (expect 0.01), `encoder.d_model` (expect 512), `encoder.n_layers` (expect 17), `encoder.subsampling_factor` (expect 8), transformer layer count (expect 18) and hidden size (expect 192), `max_num_of_spks` (expect 4). These feed Task 4's assertions.

- [ ] **Step 4: Commit**

```bash
git add tools/export_sortformer_weights.py tests/fixtures/sortformer_config.json tests/fixtures/sortformer_inventory.json
git commit -m "feat: add Sortformer weight export tool and metadata fixtures"
```

---

### Task 2: sortformer-mlx crate with inventory-driven conversion

**Files:**
- Create: `crates/sortformer-mlx/Cargo.toml`
- Create: `crates/sortformer-mlx/src/lib.rs`
- Create: `crates/sortformer-mlx/src/weights.rs`
- Modify: `Cargo.toml` (workspace members)
- Test: `crates/sortformer-mlx/tests/weights.rs`

**Interfaces:**
- Consumes: `nemotron_mlx::weights::{convert_tensors, copy_model_companion_files, Storage, TensorSpec, TensorTransform, Artifact, ArtifactError, ArtifactResult}` (all already `pub`).
- Produces: `sortformer_mlx::weights::specs_from_inventory(inventory: &BTreeMap<String, Vec<usize>>) -> Vec<TensorSpec>`, `sortformer_mlx::weights::convert_model(source: &Path, output: &Path, group_size: usize) -> ArtifactResult<()>`, `sortformer_mlx::weights::MODEL_ID: &str`.

- [ ] **Step 1: Create the crate and register it in the workspace**

`crates/sortformer-mlx/Cargo.toml`:

```toml
[package]
name = "sortformer-mlx"
version = "0.1.0"
edition.workspace = true
license.workspace = true
rust-version.workspace = true

[dependencies]
half = "2"
mlx-rs = { version = "0.25.3", default-features = false, features = ["accelerate", "metal", "safetensors"] }
nemotron-mlx = { path = "../nemotron-mlx" }
rustfft = "6"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"

[dev-dependencies]
approx = "0.5"
hound = "3"
```

`crates/sortformer-mlx/src/lib.rs`:

```rust
//! Rust model runtime for NVIDIA Streaming Sortformer diarization on MLX.

pub mod weights;
```

Root `Cargo.toml` members line becomes:

```toml
members = ["crates/catcher-ffi", "crates/nemotron-cli", "crates/nemotron-convert", "crates/nemotron-mlx", "crates/sortformer-convert", "crates/sortformer-mlx"]
```

(`sortformer-convert` is created in Task 3; adding the member now is fine because Task 3 lands before the workspace next builds in CI, but if you build in between, temporarily keep only `sortformer-mlx` and add `sortformer-convert` in Task 3.) Use the two-step: add only `crates/sortformer-mlx` now, extend in Task 3.

- [ ] **Step 2: Write failing tests for the storage policy and inventory plan**

`crates/sortformer-mlx/tests/weights.rs`:

```rust
use std::collections::BTreeMap;

use nemotron_mlx::weights::{Storage, TensorTransform};
use sortformer_mlx::weights::specs_from_inventory;

fn inventory(entries: &[(&str, &[usize])]) -> BTreeMap<String, Vec<usize>> {
    entries
        .iter()
        .map(|(name, shape)| (name.to_string(), shape.to_vec()))
        .collect()
}

#[test]
fn large_matrices_quantize_to_int8_group_128() {
    let specs = specs_from_inventory(&inventory(&[(
        "encoder.layers.0.self_attn.linear_q.weight",
        &[512, 512],
    )]));
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].storage, Storage::Int8Affine { group_size: 128 });
    assert_eq!(specs[0].transform, TensorTransform::Identity);
    assert_eq!(specs[0].artifact_shape, vec![512, 512]);
}

#[test]
fn pointwise_convolutions_squeeze_then_quantize() {
    let specs = specs_from_inventory(&inventory(&[(
        "encoder.layers.0.conv.pointwise_conv1.weight",
        &[1024, 512, 1],
    )]));
    assert_eq!(specs[0].storage, Storage::Int8Affine { group_size: 128 });
    assert_eq!(
        specs[0].transform,
        TensorTransform::SqueezeTrailingUnitDimensions
    );
    assert_eq!(specs[0].artifact_shape, vec![1024, 512]);
}

#[test]
fn narrow_and_odd_tensors_stay_f16() {
    let specs = specs_from_inventory(&inventory(&[
        ("transformer_encoder.layers.0.first_sub_layer.query_net.weight", &[192, 192]),
        ("encoder.layers.0.conv.depthwise_conv.weight", &[512, 1, 9]),
        ("encoder.layers.0.self_attn.pos_bias_u", &[8, 64]),
        ("encoder.layers.0.norm_out.bias", &[512]),
        ("encoder.pre_encode.conv.0.weight", &[256, 1, 3, 3]),
    ]));
    assert_eq!(specs.len(), 5);
    for spec in &specs {
        assert_eq!(spec.storage, Storage::F16, "tensor {}", spec.name);
        assert_eq!(spec.transform, TensorTransform::Identity, "tensor {}", spec.name);
    }
}

#[test]
fn real_inventory_produces_a_plan_covering_every_tensor() {
    let inventory: BTreeMap<String, Vec<usize>> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_inventory.json"
    )))
    .unwrap();
    let specs = specs_from_inventory(&inventory);
    assert_eq!(specs.len(), inventory.len());
    let parameters: usize = specs
        .iter()
        .map(|spec| spec.source_shape.iter().product::<usize>())
        .sum();
    assert!(
        (110_000_000..125_000_000).contains(&parameters),
        "unexpected parameter count {parameters}"
    );
    let int8 = specs
        .iter()
        .filter(|spec| matches!(spec.storage, Storage::Int8Affine { .. }))
        .count();
    assert!(int8 > 100, "expected the conformer stack quantized, got {int8}");
}
```

- [ ] **Step 2a: Run the tests to verify they fail**

Run: `cargo test -p sortformer-mlx --test weights`
Expected: FAIL — `specs_from_inventory` not found.

- [ ] **Step 3: Implement `weights.rs`**

```rust
//! Inventory-driven conversion plan and artifact helpers for Sortformer.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_rs::Array;
use nemotron_mlx::weights::{
    convert_tensors, copy_model_companion_files, ArtifactResult, Storage, TensorSpec,
    TensorTransform,
};

/// Upstream checkpoint identity recorded in the artifact manifest.
pub const MODEL_ID: &str = "nvidia/diar_streaming_sortformer_4spk-v2.1";

const GROUP_SIZE: usize = 128;
const MIN_QUANTIZED_ROWS: usize = 8;

/// Builds the conversion plan for every tensor in an exported inventory.
///
/// Rank-2 `.weight` matrices (after squeezing trailing unit dimensions) whose
/// input dimension is a multiple of the group size quantize to affine INT8;
/// everything else (norms, biases, depthwise/2-D convolutions, attention
/// position biases, and the 192-wide transformer stack) stays FP16.
pub fn specs_from_inventory(inventory: &BTreeMap<String, Vec<usize>>) -> Vec<TensorSpec> {
    inventory
        .iter()
        .map(|(name, shape)| spec_for(name, shape))
        .collect()
}

fn spec_for(name: &str, shape: &[usize]) -> TensorSpec {
    let squeezable = shape.len() > 2 && shape[2..].iter().all(|dimension| *dimension == 1);
    let matrix: &[usize] = if squeezable { &shape[..2] } else { shape };
    let quantize = name.ends_with(".weight")
        && matrix.len() == 2
        && matrix[1] % GROUP_SIZE == 0
        && matrix[1] > 0
        && matrix[0] >= MIN_QUANTIZED_ROWS;
    if quantize {
        TensorSpec {
            name: name.to_string(),
            source_shape: shape.to_vec(),
            artifact_shape: matrix.to_vec(),
            storage: Storage::Int8Affine {
                group_size: GROUP_SIZE,
            },
            transform: if squeezable {
                TensorTransform::SqueezeTrailingUnitDimensions
            } else {
                TensorTransform::Identity
            },
        }
    } else {
        TensorSpec {
            name: name.to_string(),
            source_shape: shape.to_vec(),
            artifact_shape: shape.to_vec(),
            storage: Storage::F16,
            transform: TensorTransform::Identity,
        }
    }
}

/// Converts the exported F32 safetensors into an MLX release artifact.
pub fn convert_model(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
    group_size: usize,
) -> ArtifactResult<()> {
    let source = source.as_ref();
    let inventory = read_inventory(source)?;
    let mut specs = specs_from_inventory(&inventory);
    if group_size != GROUP_SIZE {
        for spec in &mut specs {
            if let Storage::Int8Affine { .. } = spec.storage {
                if spec.artifact_shape[1] % group_size == 0 {
                    spec.storage = Storage::Int8Affine { group_size };
                } else {
                    spec.storage = Storage::F16;
                }
            }
        }
    }
    convert_tensors(source, output.as_ref(), MODEL_ID, &specs)?;
    copy_model_companion_files(source, output)
}

fn read_inventory(source: &Path) -> ArtifactResult<BTreeMap<String, Vec<usize>>> {
    let arrays = Array::load_safetensors(source)?;
    Ok(arrays
        .iter()
        .map(|(name, array)| {
            (
                name.clone(),
                array.shape().iter().map(|value| *value as usize).collect(),
            )
        })
        .collect())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p sortformer-mlx --test weights`
Expected: 4 passed. If `narrow_and_odd_tensors_stay_f16` fails because a fixture-listed tensor name differs from the plan's synthetic names, keep the synthetic tests generic (they test the *policy*) and check the real names only through `real_inventory_produces_a_plan_covering_every_tensor`.

- [ ] **Step 5: Convert the real checkpoint and record the artifact size**

```bash
cargo run -p sortformer-convert --release -- --help 2>/dev/null || true  # not built yet; use a scratch runner:
cat > /tmp/convert_sortformer.rs <<'EOF'
fn main() {
    sortformer_mlx::weights::convert_model(
        "/tmp/sortformer-f32/model.safetensors",
        "/tmp/catcher-diar-mlx-int8",
        128,
    )
    .unwrap();
}
EOF
```

Skip the scratch runner: Task 3 delivers the CLI; conversion of the real checkpoint is verified there. This step only confirms the crate builds: `cargo build -p sortformer-mlx`. Expected: success.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/sortformer-mlx
git commit -m "feat: add sortformer-mlx crate with inventory-driven conversion plan"
```

---

### Task 3: sortformer-convert CLI

**Files:**
- Create: `crates/sortformer-convert/Cargo.toml`
- Create: `crates/sortformer-convert/src/main.rs`
- Modify: `Cargo.toml` (add `crates/sortformer-convert` to members)
- Test: `crates/sortformer-convert/tests/cli.rs`

**Interfaces:**
- Consumes: `sortformer_mlx::weights::convert_model(source, output, group_size)`.
- Produces: `sortformer-convert` binary with `--source`, `--output`, `--group-size` arguments; the converted `/tmp/catcher-diar-mlx-int8` artifact used by all later real-checkpoint tests via env var `SORTFORMER_MLX_ARTIFACT`.

- [ ] **Step 1: Write failing CLI tests**

`crates/sortformer-convert/tests/cli.rs` (mirrors `nemotron-convert/tests/cli.rs`):

```rust
use std::process::Command;

#[test]
fn help_describes_source_output_and_group_size_arguments() {
    let output = Command::new(env!("CARGO_BIN_EXE_sortformer-convert"))
        .arg("--help")
        .output()
        .expect("run converter help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("--source"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--group-size"));
    assert!(stdout.contains("Sortformer"));
}

#[test]
fn missing_source_fails_without_panicking() {
    let output_path =
        std::env::temp_dir().join(format!("sortformer-convert-cli-test-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_sortformer-convert"))
        .args([
            "--source",
            "missing.safetensors",
            "--output",
            output_path.to_str().unwrap(),
        ])
        .output()
        .expect("run converter");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("conversion failed"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
```

- [ ] **Step 2: Run tests to verify they fail to build**

Run: `cargo test -p sortformer-convert --test cli`
Expected: FAIL — package does not exist yet.

- [ ] **Step 3: Implement the CLI**

`crates/sortformer-convert/Cargo.toml`:

```toml
[package]
name = "sortformer-convert"
version = "0.1.0"
edition.workspace = true
license.workspace = true
rust-version.workspace = true

[dependencies]
clap = { version = "4", features = ["derive"] }
sortformer-mlx = { path = "../sortformer-mlx" }
```

`crates/sortformer-convert/src/main.rs`:

```rust
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use sortformer_mlx::weights::convert_model;

/// Convert exported Sortformer v2.1 F32 safetensors to an MLX INT8 artifact.
#[derive(Debug, Parser)]
#[command(name = "sortformer-convert", version, about)]
struct Arguments {
    /// Exported model.safetensors from tools/export_sortformer_weights.py.
    #[arg(long)]
    source: PathBuf,

    /// New output directory for weights.safetensors and manifest.json.
    #[arg(long)]
    output: PathBuf,

    /// Affine INT8 values per scale/bias group.
    #[arg(long, default_value_t = 128)]
    group_size: usize,
}

fn main() -> ExitCode {
    let arguments = Arguments::parse();
    match convert_model(&arguments.source, &arguments.output, arguments.group_size) {
        Ok(()) => {
            println!("wrote MLX INT8 artifact to {}", arguments.output.display());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("conversion failed: {error}");
            ExitCode::FAILURE
        }
    }
}
```

Add `"crates/sortformer-convert"` to the root `Cargo.toml` members list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p sortformer-convert --test cli`
Expected: 2 passed.

- [ ] **Step 5: Convert the real checkpoint**

```bash
cargo run -p sortformer-convert --release -- \
  --source /tmp/sortformer-f32/model.safetensors \
  --output /tmp/catcher-diar-mlx-int8
du -sh /tmp/catcher-diar-mlx-int8
export SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8
```

Expected: success message; directory size roughly 120–160 MB (INT8 conformer + FP16 transformer). `config.json` must be present inside the output (copied companion file).

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/sortformer-convert
git commit -m "feat: add sortformer-convert CLI"
```

---

### Task 4: Config parsing and NeMo mel frontend

**Files:**
- Create: `crates/sortformer-mlx/src/config.rs`
- Create: `crates/sortformer-mlx/src/audio.rs`
- Modify: `crates/sortformer-mlx/src/lib.rs`
- Test: `crates/sortformer-mlx/tests/config.rs`
- Test: `crates/sortformer-mlx/tests/audio.rs`

**Interfaces:**
- Consumes: `tests/fixtures/sortformer_config.json` (Task 1).
- Produces:
  - `sortformer_mlx::config::SortformerConfig` with `pub fn load(model_dir: &Path) -> Result<Self, ConfigError>` and `pub fn from_json(json: &str) -> Result<Self, ConfigError>`; fields `sample_rate: usize`, `n_mels: usize`, `window_seconds: f64`, `hop_seconds: f64`, `n_fft: usize`, `preemphasis: f32`, `encoder_layers: usize`, `encoder_dim: usize`, `encoder_heads: usize`, `conv_kernel: usize`, `subsampling_factor: usize`, `subsampling_channels: usize`, `transformer_layers: usize`, `transformer_dim: usize`, `transformer_inner_dim: usize`, `transformer_heads: usize`, `num_speakers: usize`.
  - `sortformer_mlx::audio::MelFrontend` with `pub fn new(config: &SortformerConfig) -> Self` and `pub fn extract_normalized(&self, audio: &[f32]) -> Vec<Vec<f32>>` (per-frame `n_mels` vectors, log mel with zero-guard, preemphasis, Hann window, Slaney mel filterbank, then per-feature mean/variance normalization over the whole utterance).

- [ ] **Step 1: Write failing config tests**

`crates/sortformer-mlx/tests/config.rs`:

```rust
use sortformer_mlx::config::SortformerConfig;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/sortformer_config.json"
));

#[test]
fn real_config_parses_with_expected_architecture() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    assert_eq!(config.sample_rate, 16_000);
    assert_eq!(config.n_mels, 80);
    assert_eq!(config.encoder_layers, 17);
    assert_eq!(config.encoder_dim, 512);
    assert_eq!(config.subsampling_factor, 8);
    assert_eq!(config.transformer_layers, 18);
    assert_eq!(config.transformer_dim, 192);
    assert_eq!(config.num_speakers, 4);
    assert!((config.hop_seconds - 0.01).abs() < 1e-9);
}

#[test]
fn output_frame_duration_is_80_ms() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frame_ms = config.hop_seconds * config.subsampling_factor as f64 * 1_000.0;
    assert!((frame_ms - 80.0).abs() < 1e-6);
}
```

If any assertion contradicts the real fixture (Task 1 recorded the actual values), fix the *assertion* to the fixture's value and propagate the corrected value through later tasks — the fixture is ground truth.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p sortformer-mlx --test config`
Expected: FAIL — module `config` not found.

- [ ] **Step 3: Implement `config.rs`**

The raw NeMo YAML (as JSON) nests these values under `preprocessor`, `encoder`, `transformer_encoder` / `sortformer_modules` sections whose exact key names come from the fixture. Implement with serde against the fixture's real key paths; the expected NeMo names are shown below — verify each against `tests/fixtures/sortformer_config.json` and adjust to the file:

```rust
//! Model configuration parsed from the exported NeMo `config.json`.

use std::fs;
use std::path::Path;

/// Errors loading or interpreting the model configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The configuration file could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The configuration JSON is invalid or missing required keys.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, serde::Deserialize)]
struct RawConfig {
    preprocessor: RawPreprocessor,
    encoder: RawEncoder,
    transformer_encoder: RawTransformer,
    sortformer_modules: RawSortformerModules,
}

#[derive(Debug, serde::Deserialize)]
struct RawPreprocessor {
    sample_rate: usize,
    features: usize,
    window_size: f64,
    window_stride: f64,
    #[serde(default)]
    n_fft: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
struct RawEncoder {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    conv_kernel_size: usize,
    subsampling_factor: usize,
    subsampling_conv_channels: usize,
}

#[derive(Debug, serde::Deserialize)]
struct RawTransformer {
    num_layers: usize,
    hidden_size: usize,
    inner_size: usize,
    num_attention_heads: usize,
}

#[derive(Debug, serde::Deserialize)]
struct RawSortformerModules {
    num_spks: usize,
}

/// Validated architecture and audio-frontend parameters.
#[derive(Debug, Clone)]
pub struct SortformerConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub window_seconds: f64,
    pub hop_seconds: f64,
    pub n_fft: usize,
    pub preemphasis: f32,
    pub encoder_layers: usize,
    pub encoder_dim: usize,
    pub encoder_heads: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    pub transformer_layers: usize,
    pub transformer_dim: usize,
    pub transformer_inner_dim: usize,
    pub transformer_heads: usize,
    pub num_speakers: usize,
}

impl SortformerConfig {
    /// Reads `config.json` from a converted artifact directory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json(&fs::read_to_string(model_dir.as_ref().join("config.json"))?)
    }

    /// Parses the exported NeMo configuration JSON.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig = serde_json::from_str(json)?;
        Ok(Self {
            sample_rate: raw.preprocessor.sample_rate,
            n_mels: raw.preprocessor.features,
            window_seconds: raw.preprocessor.window_size,
            hop_seconds: raw.preprocessor.window_stride,
            n_fft: raw.preprocessor.n_fft.unwrap_or(512),
            preemphasis: 0.97,
            encoder_layers: raw.encoder.n_layers,
            encoder_dim: raw.encoder.d_model,
            encoder_heads: raw.encoder.n_heads,
            conv_kernel: raw.encoder.conv_kernel_size,
            subsampling_factor: raw.encoder.subsampling_factor,
            subsampling_channels: raw.encoder.subsampling_conv_channels,
            transformer_layers: raw.transformer_encoder.num_layers,
            transformer_dim: raw.transformer_encoder.hidden_size,
            transformer_inner_dim: raw.transformer_encoder.inner_size,
            transformer_heads: raw.transformer_encoder.num_attention_heads,
            num_speakers: raw.sortformer_modules.num_spks,
        })
    }
}
```

Register in `lib.rs`: add `pub mod audio;` and `pub mod config;`.

- [ ] **Step 4: Run config tests to verify they pass**

Run: `cargo test -p sortformer-mlx --test config`
Expected: 2 passed (after aligning serde paths with the real fixture).

- [ ] **Step 5: Write a failing frontend shape/normalization test**

`crates/sortformer-mlx/tests/audio.rs`:

```rust
use sortformer_mlx::audio::MelFrontend;
use sortformer_mlx::config::SortformerConfig;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/sortformer_config.json"
));

#[test]
fn one_second_of_audio_yields_100_normalized_frames() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frontend = MelFrontend::new(&config);
    let audio: Vec<f32> = (0..16_000)
        .map(|index| (index as f32 * 0.02).sin() * 0.3)
        .collect();
    let frames = frontend.extract_normalized(&audio);
    assert!((98..=101).contains(&frames.len()), "frames {}", frames.len());
    assert!(frames.iter().all(|frame| frame.len() == config.n_mels));
    // Per-feature normalization: each mel bin is zero-mean unit-variance over time.
    for bin in 0..config.n_mels {
        let count = frames.len() as f32;
        let mean: f32 = frames.iter().map(|frame| frame[bin]).sum::<f32>() / count;
        let variance: f32 =
            frames.iter().map(|frame| (frame[bin] - mean).powi(2)).sum::<f32>() / (count - 1.0);
        assert!(mean.abs() < 1e-3, "bin {bin} mean {mean}");
        assert!((variance - 1.0).abs() < 2e-2, "bin {bin} variance {variance}");
    }
}
```

- [ ] **Step 6: Run it to verify it fails, then implement `audio.rs`**

Run: `cargo test -p sortformer-mlx --test audio` → FAIL (module missing).

Implement the NeMo `AudioToMelSpectrogramPreprocessor` pipeline (study `crates/nemotron-mlx/src/audio/log_mel.rs` for the repo's rustfft framing style, but this is a separate implementation because NeMo adds preemphasis and per-feature normalization):

```rust
//! NeMo-compatible log-mel frontend with per-feature normalization.

use rustfft::{num_complex::Complex, FftPlanner};

use crate::config::SortformerConfig;

/// Offline mel-spectrogram extractor matching NeMo preprocessing.
#[derive(Debug)]
pub struct MelFrontend {
    sample_rate: usize,
    window_length: usize,
    hop_length: usize,
    n_fft: usize,
    preemphasis: f32,
    filterbank: Vec<Vec<f32>>, // [n_mels][n_fft / 2 + 1]
}

const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24, NeMo's log zero guard.

impl MelFrontend {
    /// Builds the frontend from the model configuration.
    pub fn new(config: &SortformerConfig) -> Self {
        let window_length = (config.window_seconds * config.sample_rate as f64).round() as usize;
        let hop_length = (config.hop_seconds * config.sample_rate as f64).round() as usize;
        Self {
            sample_rate: config.sample_rate,
            window_length,
            hop_length,
            n_fft: config.n_fft,
            preemphasis: config.preemphasis,
            filterbank: slaney_mel_filterbank(
                config.n_mels,
                config.n_fft,
                config.sample_rate,
                0.0,
                config.sample_rate as f32 / 2.0,
            ),
        }
    }

    /// Extracts log-mel frames and applies per-feature normalization.
    pub fn extract_normalized(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let mut frames = self.extract(audio);
        normalize_per_feature(&mut frames);
        frames
    }

    fn extract(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        // Preemphasis: y[t] = x[t] - k * x[t-1].
        let mut signal = Vec::with_capacity(audio.len());
        let mut previous = 0.0f32;
        for &sample in audio {
            signal.push(sample - self.preemphasis * previous);
            previous = sample;
        }
        // Center-padded framing (reflect), Hann window, rfft power, mel, log.
        let pad = self.n_fft / 2;
        let padded = reflect_pad(&signal, pad);
        let window: Vec<f32> = (0..self.window_length)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * index as f32 / self.window_length as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let bins = self.n_fft / 2 + 1;
        let frame_count = if signal.is_empty() {
            0
        } else {
            signal.len() / self.hop_length + 1
        };
        let mut output = Vec::with_capacity(frame_count);
        for frame in 0..frame_count {
            let start = frame * self.hop_length;
            let mut buffer = vec![Complex::new(0.0f32, 0.0f32); self.n_fft];
            for (index, weight) in window.iter().enumerate() {
                let sample = padded.get(start + index).copied().unwrap_or(0.0);
                buffer[index] = Complex::new(sample * weight, 0.0);
            }
            fft.process(&mut buffer);
            let power: Vec<f32> = buffer[..bins].iter().map(|value| value.norm_sqr()).collect();
            let mel: Vec<f32> = self
                .filterbank
                .iter()
                .map(|filter| {
                    let energy: f32 =
                        filter.iter().zip(&power).map(|(weight, value)| weight * value).sum();
                    (energy + LOG_ZERO_GUARD).ln()
                })
                .collect();
            output.push(mel);
        }
        output
    }
}

fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let mut padded = Vec::with_capacity(signal.len() + 2 * pad);
    for index in (1..=pad).rev() {
        padded.push(signal.get(index).copied().unwrap_or(0.0));
    }
    padded.extend_from_slice(signal);
    for index in 0..pad {
        let source = signal.len().saturating_sub(2 + index);
        padded.push(signal.get(source).copied().unwrap_or(0.0));
    }
    padded
}

fn normalize_per_feature(frames: &mut [Vec<f32>]) {
    if frames.len() < 2 {
        return;
    }
    let bins = frames[0].len();
    let count = frames.len() as f32;
    for bin in 0..bins {
        let mean: f32 = frames.iter().map(|frame| frame[bin]).sum::<f32>() / count;
        let variance: f32 = frames
            .iter()
            .map(|frame| (frame[bin] - mean).powi(2))
            .sum::<f32>()
            / (count - 1.0);
        let std = variance.sqrt() + 1e-5;
        for frame in frames.iter_mut() {
            frame[bin] = (frame[bin] - mean) / std;
        }
    }
}

fn slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    fn hz_to_mel(hz: f32) -> f32 {
        // Slaney scale: linear below 1 kHz, logarithmic above.
        if hz < 1_000.0 {
            hz * 3.0 / 200.0
        } else {
            15.0 + (hz / 1_000.0).ln() / (6.4f32.ln() / 27.0)
        }
    }
    fn mel_to_hz(mel: f32) -> f32 {
        if mel < 15.0 {
            mel * 200.0 / 3.0
        } else {
            1_000.0 * ((mel - 15.0) * (6.4f32.ln() / 27.0)).exp()
        }
    }
    let bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let edges: Vec<f32> = (0..n_mels + 2)
        .map(|index| {
            mel_to_hz(mel_min + (mel_max - mel_min) * index as f32 / (n_mels + 1) as f32)
        })
        .collect();
    let bin_hz: Vec<f32> = (0..bins)
        .map(|index| index as f32 * sample_rate as f32 / n_fft as f32)
        .collect();
    (0..n_mels)
        .map(|mel| {
            let (lower, center, upper) = (edges[mel], edges[mel + 1], edges[mel + 2]);
            let norm = 2.0 / (upper - lower); // Slaney area normalization.
            bin_hz
                .iter()
                .map(|&hz| {
                    let weight = if hz <= lower || hz >= upper {
                        0.0
                    } else if hz <= center {
                        (hz - lower) / (center - lower)
                    } else {
                        (upper - hz) / (upper - center)
                    };
                    weight * norm
                })
                .collect()
        })
        .collect()
}
```

- [ ] **Step 7: Run audio tests to verify they pass**

Run: `cargo test -p sortformer-mlx --test audio`
Expected: PASS. (Exact NeMo parity is verified in Task 6 against the captured `features` checkpoint; this test locks shape and normalization semantics.)

- [ ] **Step 8: Commit**

```bash
git add crates/sortformer-mlx
git commit -m "feat: add Sortformer config parsing and NeMo mel frontend"
```

---

### Task 5: NeMo reference generator

**Files:**
- Create: `tools/generate_sortformer_reference.py`
- Create (generated): `tests/fixtures/sortformer_reference.json`

**Interfaces:**
- Consumes: the `.nemo` checkpoint and `tests/fixtures/hello-streaming.wav` (existing fixture, mono 16 kHz).
- Produces: `tests/fixtures/sortformer_reference.json` with, for the fixture WAV, summaries (`shape`, `mean`, `rms`, `first` 64 values) of: `features` (normalized mel), `pre_encode` (subsampling output), `encoder_layer_{0..16}`, `encoder_out`, `projected` (512→192), `transformer_layer_{0..17}`, and full flattened `probabilities` (T×4). Tasks 6–8 consume it.

- [ ] **Step 1: Write the reference generator**

```python
"""Capture NeMo Sortformer intermediate activations for Rust parity tests.

Usage: python tools/generate_sortformer_reference.py MODEL.nemo WAV OUTPUT_JSON
Requires: pip install "nemo_toolkit[asr]" soundfile
Run with --dump-structure first to print the module tree and confirm the
attribute names used below match this NeMo version; adjust if they differ.
"""

import json
import sys
from pathlib import Path

import soundfile
import torch
from nemo.collections.asr.models import SortformerEncLabelModel


def summary(tensor):
    flat = tensor.detach().float().flatten()
    return {
        "shape": list(tensor.shape),
        "mean": flat.mean().item(),
        "rms": flat.square().mean().sqrt().item(),
        "first": flat[:64].tolist(),
    }


def main() -> None:
    dump_structure = "--dump-structure" in sys.argv
    arguments = [value for value in sys.argv[1:] if value != "--dump-structure"]
    if len(arguments) != 3 and not (dump_structure and len(arguments) >= 1):
        raise SystemExit(
            "usage: generate_sortformer_reference.py MODEL.nemo WAV OUTPUT_JSON"
        )
    model = SortformerEncLabelModel.restore_from(arguments[0], map_location="cpu").eval()
    if dump_structure:
        print(model)
        return
    wav_path, output_path = Path(arguments[1]), Path(arguments[2])

    audio, sample_rate = soundfile.read(wav_path, dtype="float32")
    assert sample_rate == 16_000 and audio.ndim == 1
    signal = torch.tensor(audio)[None, :]
    length = torch.tensor([signal.shape[1]])

    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value

        return hook

    handles = [model.encoder.pre_encode.register_forward_hook(capture("pre_encode"))]
    handles += [
        layer.register_forward_hook(capture(f"encoder_layer_{index}"))
        for index, layer in enumerate(model.encoder.layers)
    ]
    handles += [
        layer.register_forward_hook(capture(f"transformer_layer_{index}"))
        for index, layer in enumerate(model.transformer_encoder.layers)
    ]

    with torch.no_grad():
        features, feature_length = model.preprocessor(
            input_signal=signal, length=length
        )
        captured["features"] = features  # [1, n_mels, frames]
        encoded, encoded_length = model.encoder(
            audio_signal=features, length=feature_length
        )
        captured["encoder_out"] = encoded  # [1, d_model, frames/8]
        embeddings = encoded.transpose(1, 2)  # [1, frames/8, d_model]
        projected = model.sortformer_modules.encoder_proj(embeddings)
        captured["projected"] = projected
        transformed = model.transformer_encoder(
            encoder_states=projected,
            encoder_mask=torch.ones(projected.shape[:2]),
        )
        probabilities = model.sortformer_modules.forward_speaker_sigmoids(transformed)
        captured["probabilities"] = probabilities  # [1, frames/8, 4]

    for handle in handles:
        handle.remove()

    payload = {name: summary(tensor) for name, tensor in captured.items()}
    payload["probabilities_full"] = probabilities.flatten().tolist()
    payload["wav"] = wav_path.name
    output_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {output_path} with {probabilities.shape[1]} frames")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run structure dump, align attribute names, then generate**

```bash
/tmp/sortformer-venv/bin/pip install "nemo_toolkit[asr]" soundfile
/tmp/sortformer-venv/bin/python tools/generate_sortformer_reference.py \
  /tmp/sortformer-src/diar_streaming_sortformer_4spk-v2.1.nemo --dump-structure
```

Confirm from the printed tree: the preprocessor/encoder attribute names, the projection module (`sortformer_modules.encoder_proj`), the transformer attribute (`transformer_encoder`), and the sigmoid head method (`forward_speaker_sigmoids`). If any differ in this NeMo version, edit the script accordingly (the script is dev tooling; correctness is defined by it running and producing the captures). Then:

```bash
/tmp/sortformer-venv/bin/python tools/generate_sortformer_reference.py \
  /tmp/sortformer-src/diar_streaming_sortformer_4spk-v2.1.nemo \
  tests/fixtures/hello-streaming.wav tests/fixtures/sortformer_reference.json
```

Expected: `wrote tests/fixtures/sortformer_reference.json with N frames` where N ≈ audio_seconds / 0.08. Sanity: the WAV is single-speaker, so exactly one of the four `probabilities` columns should dominate — spot-check a few frames in the JSON.

- [ ] **Step 3: Commit**

```bash
git add tools/generate_sortformer_reference.py tests/fixtures/sortformer_reference.json
git commit -m "test: add NeMo Sortformer reference activations for parity tests"
```

---

### Task 6: NEST Fast-Conformer encoder (offline)

**Files:**
- Create: `crates/sortformer-mlx/src/model/mod.rs`
- Create: `crates/sortformer-mlx/src/model/encoder.rs`
- Modify: `crates/sortformer-mlx/src/lib.rs` (add `pub mod model;`)
- Modify: `crates/nemotron-mlx/src/model/mod.rs` (re-export layer primitives if not already `pub use`)
- Test: `crates/sortformer-mlx/tests/encoder_parity.rs`

**Interfaces:**
- Consumes: `nemotron_mlx::model::layers::{QuantizedLinear, DepthwiseConv1d, LayerNorm, Tensor3}` (verify these are re-exported from `nemotron_mlx::model`; if `model/mod.rs` does not `pub use layers::*;` or declare `pub mod layers;`, add the re-export — that is the only permitted change to nemotron-mlx in this task), `sortformer_mlx::{audio::MelFrontend, config::SortformerConfig}`, `nemotron_mlx::weights::Artifact`.
- Produces: `sortformer_mlx::model::Encoder` with `pub fn from_artifact(artifact: &Artifact, config: &SortformerConfig) -> ModelResult<Self>` and `pub fn forward(&self, mel_frames: &[Vec<f32>]) -> ModelResult<Tensor3>` returning `[1, frames/8, encoder_dim]`; `sortformer_mlx::model::{ModelError, ModelResult}` mirroring nemotron-mlx's error enum shape (`InvalidShape(String)`, `Artifact(#[from] ArtifactError)`, `Mlx(#[from] mlx_rs::error::Exception)`).

This is the largest task. The NEST encoder is NeMo's Fast-Conformer — the same block structure `crates/nemotron-mlx/src/model/encoder.rs` already implements for Nemotron, with these differences:

| Aspect | Nemotron (existing encoder.rs) | Sortformer NEST (this task) |
|---|---|---|
| mel bins | 128 | 80 (from config) |
| d_model | 1024 | 512 (from config) |
| layers | 24 | 17 (from config) |
| attention | cache-aware, limited context, lookahead | full context over the whole utterance, no cache |
| conv module | causal, cached | full-context (symmetric padding), no cache |
| conv norm | layer/batch per checkpoint | per checkpoint (`conv.batch_norm.*` tensors in inventory ⇒ BatchNorm inference: `(x-mean)/sqrt(var+eps)*w+b` from `running_mean`/`running_var`/`weight`/`bias`) |
| subsampling | conv_in + 2 dw/pw layers | dw-striding stack per inventory (`encoder.pre_encode.*`), 8× total stride |
| tensor names | `encoder.layers.N.*` HF names | NeMo names from `sortformer_inventory.json` (e.g. `encoder.layers.N.self_attn.linear_q.weight`, `.self_attn.pos_bias_u`, `.conv.depthwise_conv.weight`, `.norm_feed_forward1.weight`) |

Implementation procedure (not optional): open `sortformer_inventory.json`, list every `encoder.pre_encode.*` and `encoder.layers.0.*` tensor with shapes, and write the module constructors against those exact names. Mirror the forward math of `nemotron-mlx/src/model/encoder.rs` block by block (ff1 half-step → MHSA with Transformer-XL relative position encoding using `pos_bias_u`/`pos_bias_v` → conv module → ff2 half-step → final norm), deleting all cache/lookahead plumbing and letting attention span the full sequence. The relative-position attention math (rel-shift, `linear_pos` projection of sinusoidal embeddings) must match NeMo's `RelPositionMultiHeadAttention`; the existing encoder.rs implements the same formulation — reuse its structure with full-length context.

- [ ] **Step 1: Write the failing parity test first**

`crates/sortformer-mlx/tests/encoder_parity.rs`:

```rust
use std::path::PathBuf;

use nemotron_mlx::weights::Artifact;
use sortformer_mlx::audio::MelFrontend;
use sortformer_mlx::config::SortformerConfig;
use sortformer_mlx::model::Encoder;

#[derive(serde::Deserialize)]
struct Summary {
    shape: Vec<usize>,
    mean: f64,
    rms: f64,
    first: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct Reference {
    features: Summary,
    pre_encode: Summary,
    encoder_layer_0: Summary,
    encoder_out: Summary,
}

fn artifact() -> Artifact {
    let path = std::env::var_os("SORTFORMER_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set SORTFORMER_MLX_ARTIFACT to a converted artifact directory");
    Artifact::load(path).unwrap()
}

fn fixture_audio() -> Vec<f32> {
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect()
}

fn reference() -> Reference {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_reference.json"
    )))
    .unwrap()
}

fn assert_close(name: &str, actual: &[f32], expected: &Summary, tolerance: f64) {
    let count = actual.len() as f64;
    let rms = (actual.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / count).sqrt();
    assert!(
        (rms - expected.rms).abs() <= tolerance * expected.rms.abs().max(1e-3),
        "{name} rms {rms} vs reference {}",
        expected.rms
    );
    for (index, value) in expected.first.iter().enumerate() {
        let difference = (actual[index] as f64 - value).abs();
        assert!(
            difference <= tolerance * expected.rms.abs().max(1e-3) * 10.0,
            "{name}[{index}] {} vs {value}",
            actual[index]
        );
    }
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn mel_features_match_nemo_preprocessor() {
    let reference = reference();
    let config = SortformerConfig::load(
        std::env::var_os("SORTFORMER_MLX_ARTIFACT").map(PathBuf::from).unwrap(),
    )
    .unwrap();
    let frames = MelFrontend::new(&config).extract_normalized(&fixture_audio());
    // Reference layout is [1, n_mels, frames]; ours is [frames][n_mels].
    assert_eq!(reference.features.shape[1], config.n_mels);
    assert_eq!(reference.features.shape[2], frames.len());
    // Reference `first` walks mel bin 0 across time.
    let bin0: Vec<f32> = frames.iter().map(|frame| frame[0]).take(64).collect();
    assert_close("features", &bin0, &reference.features, 0.02);
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn encoder_output_matches_nemo_within_int8_tolerance() {
    let reference = reference();
    let artifact = artifact();
    let config = SortformerConfig::load(
        std::env::var_os("SORTFORMER_MLX_ARTIFACT").map(PathBuf::from).unwrap(),
    )
    .unwrap();
    let frames = MelFrontend::new(&config).extract_normalized(&fixture_audio());
    let encoder = Encoder::from_artifact(&artifact, &config).unwrap();
    let output = encoder.forward(&frames).unwrap();
    assert_eq!(output.shape[2], config.encoder_dim);
    assert_eq!(output.shape[1], reference.encoder_out.shape[2]);
    // Reference layout is [1, d_model, frames]; compare per-position via rms only.
    assert_close("encoder_out", &output.values, &reference.encoder_out, 0.05);
}
```

Note the deliberate two-level tolerance: exact-first-values for the F32-dominated mel frontend, rms-level for the INT8 encoder stack. If layouts differ from these assumptions when you inspect the reference JSON (shapes are recorded in it), fix the test's indexing to the recorded shapes.

- [ ] **Step 2: Run to verify failure**

Run: `SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p sortformer-mlx --test encoder_parity -- --ignored`
Expected: FAIL — `model` module missing.

- [ ] **Step 3: Implement the encoder**

Write `model/mod.rs` (error enum, re-exports) and `model/encoder.rs`:

- `struct Subsampling` — built from every `encoder.pre_encode.*` tensor (conv stack via `nemotron_mlx::model::layers` conv/linear primitives where storage matches, plain mlx-rs `conv2d` with symmetric padding where the existing causal primitives don't fit; the existing `Fp16Conv2d` is causal-padded, so implement a local `SymmetricConv2d` in encoder.rs following its OIHW→OHWI transposition code).
- `struct ConformerBlock` — fields: `norm_ff1/ff1_linear1/ff1_linear2`, `norm_attn`, attention projections (`linear_q/k/v/out`, `linear_pos`, `pos_bias_u`, `pos_bias_v`), `norm_conv`, `pointwise_conv1` (`QuantizedLinear`, GLU), `depthwise_conv` (full-context: pad `(kernel-1)/2` both sides), `batch_norm` (from running stats), `pointwise_conv2`, `norm_ff2/ff2_*`, `norm_out` only on the final block if the inventory has `encoder.norm_out.*` at top level (check the inventory; NeMo ConformerEncoder usually has no global final norm — follow the inventory).
- Full-context relative-position MHSA: precompute sinusoidal relative position embeddings for length `2T-1`, project with `linear_pos`, score `(q + pos_bias_u)·kᵀ + rel_shift((q + pos_bias_v)·posᵀ)`, softmax over all frames, no masking.
- `Encoder::forward` drives: mel `[T,80]` → subsampling `[T/8, 512]` → 17 blocks → `Tensor3 [1, T/8, 512]`.

Match the numerics style of the existing encoder.rs (FP16 arrays for weights, F32 in/out per layer, `eval()` before reading).

- [ ] **Step 4: Run parity tests until they pass**

Run: `SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p sortformer-mlx --test encoder_parity -- --ignored`
Expected: 2 passed. Debug order when they fail: `mel_features_match_nemo_preprocessor` first (frontend), then add temporary layer-by-layer prints against `pre_encode`/`encoder_layer_0` summaries from the reference JSON to localize the first diverging block; the reference file already contains every layer's summary for exactly this purpose.

- [ ] **Step 5: Run the full crate test suite and commit**

Run: `cargo test -p sortformer-mlx` (non-ignored suites must stay green).

```bash
git add crates/sortformer-mlx crates/nemotron-mlx/src/model/mod.rs
git commit -m "feat: add Sortformer NEST encoder with NeMo parity tests"
```

---

### Task 7: Transformer stack and sigmoid head

**Files:**
- Create: `crates/sortformer-mlx/src/model/transformer.rs`
- Modify: `crates/sortformer-mlx/src/model/mod.rs`
- Test: `crates/sortformer-mlx/tests/diarizer_parity.rs`

**Interfaces:**
- Consumes: `Encoder` (Task 6), reference JSON (`projected`, `transformer_layer_*`, `probabilities`, `probabilities_full`).
- Produces: `sortformer_mlx::model::Diarizer` with `pub fn from_artifact_dir(model_dir: &Path) -> ModelResult<Self>` (loads `Artifact` + `SortformerConfig` + `MelFrontend` internally) and `pub fn diarize(&self, audio: &[f32]) -> ModelResult<Vec<[f32; 4]>>` returning per-80ms-frame speaker probabilities.

- [ ] **Step 1: Write the failing end-to-end parity test**

`crates/sortformer-mlx/tests/diarizer_parity.rs`:

```rust
use std::path::PathBuf;

use sortformer_mlx::model::Diarizer;

#[derive(serde::Deserialize)]
struct Reference {
    probabilities: Shape,
    probabilities_full: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Shape {
    shape: Vec<usize>,
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn full_pipeline_probabilities_match_nemo() {
    let model_dir = std::env::var_os("SORTFORMER_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set SORTFORMER_MLX_ARTIFACT");
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect();
    let reference: Reference = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_reference.json"
    )))
    .unwrap();

    let diarizer = Diarizer::from_artifact_dir(&model_dir).unwrap();
    let probabilities = diarizer.diarize(&samples).unwrap();

    let frames = reference.probabilities.shape[1];
    assert_eq!(probabilities.len(), frames);
    let mut maximum_error = 0.0f32;
    let mut total_error = 0.0f32;
    for (frame, actual) in probabilities.iter().enumerate() {
        for speaker in 0..4 {
            let expected = reference.probabilities_full[frame * 4 + speaker];
            let difference = (actual[speaker] - expected).abs();
            maximum_error = maximum_error.max(difference);
            total_error += difference;
        }
    }
    let mean_error = total_error / (frames * 4) as f32;
    assert!(maximum_error < 0.05, "max abs error {maximum_error}");
    assert!(mean_error < 0.01, "mean abs error {mean_error}");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p sortformer-mlx --test diarizer_parity -- --ignored`
Expected: FAIL — `Diarizer` not found.

- [ ] **Step 3: Implement transformer and head**

`model/transformer.rs`, driven by the inventory's `transformer_encoder.layers.N.*` and `sortformer_modules.*` names (list them first, exactly as in Task 6's procedure):

- `struct TransformerLayer` — self-attention (`query_net`/`key_net`/`value_net`/`out_projection` or this checkpoint's actual names, all FP16 `Array` matmuls at 192 dims — small enough that quantization is skipped by the Task 2 policy), two layer norms, feed-forward `192→inner→192` with the activation named in config (check `transformer_encoder` section; NeMo default `relu`). **No positional embeddings** — Sortformer's transformer is permutation-driven by design; if the inventory contains no `*position*` tensor under `transformer_encoder`, that confirms it.
- Respect the checkpoint's pre-LN/post-LN wiring: NeMo `TransformerEncoder` defaults to post-LN (`pre_ln: false`); confirm against the config fixture and the `transformer_layer_0` reference summary (a wrong choice diverges immediately and visibly).
- `struct SigmoidHead` — `sortformer_modules` linears: projection 192→192 + activation + 192→4, then sigmoid, matching the module names captured in the inventory (`sortformer_modules.encoder_proj` is used before the transformer; the head linears follow NeMo's `forward_speaker_sigmoids`).
- `Diarizer` composes `MelFrontend → Encoder → encoder_proj (QuantizedLinear or FP16 per policy) → TransformerLayer × 18 → SigmoidHead`.

- [ ] **Step 4: Run the parity test until it passes**

Run: `SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p sortformer-mlx --test diarizer_parity -- --ignored`
Expected: PASS. Use `projected` / `transformer_layer_N` reference summaries to bisect divergence exactly as in Task 6.

- [ ] **Step 5: Commit**

```bash
git add crates/sortformer-mlx
git commit -m "feat: add Sortformer transformer stack and sigmoid diarization head"
```

---

### Task 8: Speaker segments utility

**Files:**
- Create: `crates/sortformer-mlx/src/segments.rs`
- Modify: `crates/sortformer-mlx/src/lib.rs` (add `pub mod segments;`)
- Test: unit tests inside `segments.rs`

**Interfaces:**
- Consumes: `Vec<[f32; 4]>` probabilities from `Diarizer::diarize`.
- Produces:
  - `pub struct SpeakerSegment { pub speaker: u8, pub start_ms: u64, pub end_ms: u64 }`
  - `pub fn segments_from_probs(probs: &[[f32; 4]], threshold: f32, min_frames: usize, frame_ms: u64) -> Vec<SpeakerSegment>` — per speaker independently (diarization is multi-label): binarize at `threshold`, close silent gaps shorter than `min_frames`, drop active islands shorter than `min_frames`, emit segments sorted by `start_ms` then `speaker`.

- [ ] **Step 1: Write failing unit tests**

In `segments.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn probs(rows: &[[f32; 4]]) -> Vec<[f32; 4]> {
        rows.to_vec()
    }

    #[test]
    fn continuous_activity_becomes_one_segment() {
        let input = probs(&[[0.9, 0.0, 0.0, 0.0]; 10]);
        let segments = segments_from_probs(&input, 0.5, 2, 80);
        assert_eq!(
            segments,
            vec![SpeakerSegment { speaker: 0, start_ms: 0, end_ms: 800 }]
        );
    }

    #[test]
    fn short_islands_are_dropped_and_short_gaps_closed() {
        let mut input = vec![[0.0f32; 4]; 20];
        for frame in 0..8 {
            input[frame][1] = 0.9; // speaker 1 active 0..8
        }
        input[9][1] = 0.9; // 1-frame island after a 1-frame gap: gap closes
        for frame in 15..16 {
            input[frame][2] = 0.9; // 1-frame island for speaker 2: dropped
        }
        let segments = segments_from_probs(&input, 0.5, 2, 80);
        assert_eq!(
            segments,
            vec![SpeakerSegment { speaker: 1, start_ms: 0, end_ms: 800 }]
        );
    }

    #[test]
    fn overlapping_speakers_yield_overlapping_segments() {
        let mut input = vec![[0.0f32; 4]; 10];
        for frame in 0..10 {
            input[frame][0] = 0.8;
        }
        for frame in 5..10 {
            input[frame][3] = 0.8;
        }
        let segments = segments_from_probs(&input, 0.5, 2, 80);
        assert_eq!(
            segments,
            vec![
                SpeakerSegment { speaker: 0, start_ms: 0, end_ms: 800 },
                SpeakerSegment { speaker: 3, start_ms: 400, end_ms: 800 },
            ]
        );
    }
}
```

- [ ] **Step 2: Run to verify failure, implement, re-run**

Run: `cargo test -p sortformer-mlx segments` → FAIL, then implement:

```rust
//! Threshold-based conversion of frame probabilities into speaker segments.

/// A contiguous span of one speaker's activity.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SpeakerSegment {
    /// Zero-based Sortformer output slot.
    pub speaker: u8,
    /// Segment start in milliseconds.
    pub start_ms: u64,
    /// Exclusive segment end in milliseconds.
    pub end_ms: u64,
}

/// Binarizes per-speaker activity and merges it into stable segments.
pub fn segments_from_probs(
    probs: &[[f32; 4]],
    threshold: f32,
    min_frames: usize,
    frame_ms: u64,
) -> Vec<SpeakerSegment> {
    let mut segments = Vec::new();
    for speaker in 0..4u8 {
        let mut active: Vec<bool> = probs
            .iter()
            .map(|frame| frame[speaker as usize] >= threshold)
            .collect();
        close_short_gaps(&mut active, min_frames);
        drop_short_islands(&mut active, min_frames);
        let mut start = None;
        for (frame, &on) in active.iter().enumerate() {
            match (on, start) {
                (true, None) => start = Some(frame),
                (false, Some(begin)) => {
                    segments.push(segment(speaker, begin, frame, frame_ms));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(begin) = start {
            segments.push(segment(speaker, begin, active.len(), frame_ms));
        }
    }
    segments.sort_by_key(|segment| (segment.start_ms, segment.speaker));
    segments
}

fn segment(speaker: u8, start: usize, end: usize, frame_ms: u64) -> SpeakerSegment {
    SpeakerSegment {
        speaker,
        start_ms: start as u64 * frame_ms,
        end_ms: end as u64 * frame_ms,
    }
}

fn close_short_gaps(active: &mut [bool], min_frames: usize) {
    let mut frame = 0;
    while frame < active.len() {
        if !active[frame] {
            let gap_start = frame;
            while frame < active.len() && !active[frame] {
                frame += 1;
            }
            let bounded = gap_start > 0 && frame < active.len();
            if bounded && frame - gap_start < min_frames {
                active[gap_start..frame].fill(true);
            }
        } else {
            frame += 1;
        }
    }
}

fn drop_short_islands(active: &mut [bool], min_frames: usize) {
    let mut frame = 0;
    while frame < active.len() {
        if active[frame] {
            let island_start = frame;
            while frame < active.len() && active[frame] {
                frame += 1;
            }
            if frame - island_start < min_frames {
                active[island_start..frame].fill(false);
            }
        } else {
            frame += 1;
        }
    }
}
```

Run: `cargo test -p sortformer-mlx segments` → 3 passed.

- [ ] **Step 3: Commit**

```bash
git add crates/sortformer-mlx
git commit -m "feat: add speaker segment extraction from diarization probabilities"
```

---

### Task 9: `catcher diarize` subcommand

**Files:**
- Modify: `crates/nemotron-cli/src/main.rs`
- Modify: `crates/nemotron-cli/Cargo.toml`
- Test: `crates/nemotron-cli/tests/cli.rs` (extend)

**Interfaces:**
- Consumes: `sortformer_mlx::model::Diarizer`, `sortformer_mlx::segments::{segments_from_probs, SpeakerSegment}`.
- Produces: `catcher diarize --model DIR --audio WAV [--threshold 0.5] [--min-duration-ms 400] [--json]` printing either lines `speaker N  MM:SS.mmm - MM:SS.mmm` or a JSON array of segments.

- [ ] **Step 1: Extend CLI tests (failing)**

Append to `crates/nemotron-cli/tests/cli.rs`:

```rust
#[test]
fn help_lists_the_diarize_subcommand() {
    let output = Command::new(env!("CARGO_BIN_EXE_catcher"))
        .arg("--help")
        .output()
        .expect("run catcher help");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("diarize"));
}

#[test]
fn diarize_with_missing_model_fails_cleanly() {
    let output = Command::new(env!("CARGO_BIN_EXE_catcher"))
        .args([
            "diarize",
            "--model",
            "/nonexistent",
            "--audio",
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../tests/fixtures/hello-streaming.wav"
            ),
        ])
        .output()
        .expect("run catcher diarize");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("catcher:"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p nemotron-cli --test cli`
Expected: the two new tests FAIL (`diarize` unknown).

- [ ] **Step 3: Implement the subcommand**

Add to `crates/nemotron-cli/Cargo.toml` dependencies: `sortformer-mlx = { path = "../sortformer-mlx" }`.

Add the variant and handler in `main.rs` (following the existing `Transcribe` style):

```rust
    /// Detect who speaks when in a mono 16 kHz WAV (up to 4 speakers).
    Diarize {
        /// Converted Sortformer MLX artifact directory.
        #[arg(long)]
        model: PathBuf,
        /// Mono 16 kHz PCM or float WAV file.
        #[arg(long)]
        audio: PathBuf,
        /// Speaker activity probability threshold.
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        /// Minimum segment/gap duration in milliseconds.
        #[arg(long, default_value_t = 400)]
        min_duration_ms: u64,
        /// Emit a JSON array instead of plain text.
        #[arg(long)]
        json: bool,
    },
```

```rust
        Command::Diarize {
            model,
            audio,
            threshold,
            min_duration_ms,
            json,
        } => {
            let samples = read_wav(&audio)?;
            let diarizer = sortformer_mlx::model::Diarizer::from_artifact_dir(&model)?;
            let probabilities = diarizer.diarize(&samples)?;
            const FRAME_MS: u64 = 80;
            let min_frames = (min_duration_ms / FRAME_MS).max(1) as usize;
            let segments = sortformer_mlx::segments::segments_from_probs(
                &probabilities,
                threshold,
                min_frames,
                FRAME_MS,
            );
            if json {
                println!("{}", serde_json::to_string(&segments)?);
            } else {
                for segment in &segments {
                    println!(
                        "speaker {}  {} - {}",
                        segment.speaker + 1,
                        format_timestamp(segment.start_ms),
                        format_timestamp(segment.end_ms)
                    );
                }
            }
            Ok(())
        }
```

```rust
fn format_timestamp(milliseconds: u64) -> String {
    format!(
        "{:02}:{:02}.{:03}",
        milliseconds / 60_000,
        milliseconds % 60_000 / 1_000,
        milliseconds % 1_000
    )
}
```

(The existing `read_wav` helper is reused; keep `run`'s return type unchanged.)

- [ ] **Step 4: Run tests and a real end-to-end check**

Run: `cargo test -p nemotron-cli --test cli` → all pass.

```bash
cargo run -p nemotron-cli --release -- diarize \
  --model /tmp/catcher-diar-mlx-int8 \
  --audio tests/fixtures/hello-streaming.wav
```

Expected: one or more `speaker 1  …` lines covering roughly the speech span of the fixture (single speaker).

- [ ] **Step 5: Commit**

```bash
git add crates/nemotron-cli
git commit -m "feat: add catcher diarize subcommand"
```

---

### Task 10: Artifact packaging, publication, and docs

**Files:**
- Modify: `README.md` (diarization model + CLI sections)
- Create (outside repo): published Hugging Face repo `wcamon/catcher-diar-mlx-int8`

**Interfaces:**
- Consumes: `/tmp/catcher-diar-mlx-int8` artifact (Task 3), NVIDIA license files from the source HF repo.
- Produces: public artifact download URL used by Phase 3's `ModelStore` manifest (file names, byte counts, SHA-256 hashes recorded in the commit message body for Phase 3 to consume).

- [ ] **Step 1: Assemble license and notice files into the artifact**

```bash
hf download nvidia/diar_streaming_sortformer_4spk-v2.1 README.md --local-dir /tmp/sortformer-card
cp /tmp/sortformer-card/README.md /tmp/catcher-diar-mlx-int8/NVIDIA_MODEL_CARD.md
```

Write `/tmp/catcher-diar-mlx-int8/NOTICE.md`:

```markdown
# Notice

This artifact is a quantized MLX conversion of
[`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1),
produced for the Catcher/Tippi runtime. The model weights remain governed by
the NVIDIA Open Model License (see `LICENSE`); the conversion introduces no
new training data. Original model card: `NVIDIA_MODEL_CARD.md`.
```

Copy the NVIDIA Open Model License text: it is embedded in the upstream model card; extract the license section (or download the canonical text NVIDIA links) into `/tmp/catcher-diar-mlx-int8/LICENSE`. Verify all files present: `weights.safetensors`, `manifest.json`, `config.json`, `LICENSE`, `NOTICE.md`, `NVIDIA_MODEL_CARD.md`.

- [ ] **Step 2: Write the artifact README**

`/tmp/catcher-diar-mlx-int8/README.md`:

```markdown
---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/diar_streaming_sortformer_4spk-v2.1
---

# Catcher Diarization — Streaming Sortformer v2.1, MLX INT8

MLX affine INT8 (group 128) conversion of NVIDIA Streaming Sortformer
Diarizer 4spk v2.1 for the [Catcher/Tippi](https://github.com/wcAmon/catcher-tippi)
on-device speech runtime. Detects up to 4 speakers with 80 ms frame
resolution at 16 kHz. Converted with `sortformer-convert`; inference runs on
MLX-C/Metal with no Python, PyTorch, NeMo, or ONNX Runtime.
```

- [ ] **Step 3: Compute pinned hashes and publish**

```bash
cd /tmp/catcher-diar-mlx-int8
for file in *; do shasum -a 256 "$file"; stat -f "%z %N" "$file"; done
hf repo create wcamon/catcher-diar-mlx-int8 --type model
hf upload wcamon/catcher-diar-mlx-int8 . .
```

Record every file's SHA-256 and byte count — paste the full listing into the commit message body (Phase 3's `ModelFile` constants are built from it). Verify the public download works without a token:

```bash
curl -sIL https://huggingface.co/wcamon/catcher-diar-mlx-int8/resolve/main/manifest.json | head -1
```

Expected: `HTTP/2 200`.

- [ ] **Step 4: Update the repo README**

Add to `README.md` after the existing model-download section: a "Download the public diarization model" subsection (mirroring the ASR one, `hf download wcamon/catcher-diar-mlx-int8 --local-dir catcher-diar-mlx-int8`), and a `catcher diarize` example under the CLI section:

```sh
target/release/catcher diarize \
  --model catcher-diar-mlx-int8 \
  --audio meeting.wav
```

- [ ] **Step 5: Full workspace verification and commit**

Run: `cargo test --workspace` (green) and
`SORTFORMER_MLX_ARTIFACT=/tmp/catcher-diar-mlx-int8 cargo test -p sortformer-mlx -- --ignored` (green).

```bash
git add README.md
git commit -m "docs: document the Sortformer diarization model and catcher diarize"
```

---

## Self-Review Notes

- Spec coverage (phase 1 scope): export tool ✓ (T1), conversion + INT8 policy ✓ (T2/T3), offline inference with numeric parity ✓ (T4–T7), CLI speaker timeline ✓ (T8/T9), published pinned artifact ✓ (T10). AOSC streaming, token timestamps, fusion, and s2twp are Phase 2 by design.
- Ground-truth discipline: every architectural constant asserted in tests traces to `sortformer_config.json` / `sortformer_inventory.json` / `sortformer_reference.json`, all generated from the real checkpoint before the code that depends on them.
- Known judgment calls: INT8 quantization boundary (`MIN_QUANTIZED_ROWS`, `%128`) intentionally leaves the 192-wide transformer in FP16; parity tolerances (0.05 max abs on probabilities) are the acceptance gate for INT8 quality — if Task 7 cannot meet them, rerun Tasks 3→7 with `--group-size 64` and, failing that, escalate to the user per the spec's FP16 fallback.

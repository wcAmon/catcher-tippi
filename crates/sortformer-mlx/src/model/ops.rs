//! Shared F32 tensor primitives used by both the Conformer encoder
//! (`encoder.rs`) and the Transformer stack (`transformer.rs`).
//!
//! These are plain scalar implementations (no MLX arrays) operating on
//! row-major `[rows, dims]` buffers, mirroring the small-matrix numerics
//! style used throughout this crate for tensors too small to benefit from
//! MLX dispatch.

use nemotron_mlx::weights::Artifact;

use super::{ModelError, ModelResult};

const LAYER_NORM_EPSILON: f32 = 1.0e-5;

/// F32 layer normalization over the channel dimension (NeMo `nn.LayerNorm`).
#[derive(Debug)]
pub(crate) struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    pub(crate) fn from_artifact(artifact: &Artifact, prefix: &str) -> ModelResult<Self> {
        let weight = artifact.f16_to_f32(&format!("{prefix}.weight"))?;
        let bias = artifact.f16_to_f32(&format!("{prefix}.bias"))?;
        if weight.is_empty() || weight.len() != bias.len() {
            return Err(ModelError::InvalidShape(format!(
                "layer norm {prefix} weight and bias lengths must match"
            )));
        }
        Ok(Self { weight, bias })
    }

    pub(crate) fn forward(&self, input: &[f32], rows: usize) -> ModelResult<Vec<f32>> {
        let dimensions = self.weight.len();
        if rows.checked_mul(dimensions) != Some(input.len()) {
            return Err(ModelError::InvalidShape(format!(
                "layer norm input has {} values, expected {rows}x{dimensions}",
                input.len()
            )));
        }
        let mut output = Vec::with_capacity(input.len());
        for row in input.chunks_exact(dimensions) {
            let mean = row.iter().sum::<f32>() / dimensions as f32;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / dimensions as f32;
            let scale = 1.0 / (variance + LAYER_NORM_EPSILON).sqrt();
            output.extend(
                row.iter()
                    .zip(self.weight.iter().zip(&self.bias))
                    .map(|(value, (weight, bias))| (value - mean) * scale * weight + bias),
            );
        }
        Ok(output)
    }
}

pub(crate) fn add_in_place(accumulator: &mut [f32], update: &[f32], scale: f32) {
    for (accumulated, update) in accumulator.iter_mut().zip(update) {
        *accumulated += scale * update;
    }
}

pub(crate) fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

pub(crate) fn silu_in_place(values: &mut [f32]) {
    for value in values {
        *value /= 1.0 + (-*value).exp();
    }
}

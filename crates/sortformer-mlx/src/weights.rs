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

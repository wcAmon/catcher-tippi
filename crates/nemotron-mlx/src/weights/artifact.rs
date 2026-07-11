use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use mlx_rs::{Array, Dtype as MlxDtype};

use super::{DType, ModelManifest, Storage, TensorSpec, TensorTransform};

const FORMAT_VERSION: u32 = 1;
const WEIGHTS_FILE: &str = "weights.safetensors";
const MANIFEST_FILE: &str = "manifest.json";

/// Errors produced while converting or loading an MLX artifact.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    /// Quantization configuration is incompatible with the checkpoint.
    #[error("invalid quantization configuration: {0}")]
    InvalidQuantization(String),
    /// A filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// MLX file I/O failed.
    #[error(transparent)]
    MlxIo(#[from] mlx_rs::error::IoError),
    /// MLX rejected an array operation.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
    /// An evaluated array could not be read with its expected element type.
    #[error(transparent)]
    ArraySlice(#[from] mlx_rs::error::AsSliceError),
    /// JSON metadata is invalid.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// A tensor required by the conversion plan is missing.
    #[error("missing source tensor {0}")]
    MissingSourceTensor(String),
    /// A tensor has dimensions different from its conversion plan.
    #[error("tensor {name} shape mismatch: expected {expected:?}, found {actual:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Planned dimensions.
        expected: Vec<usize>,
        /// Source dimensions.
        actual: Vec<usize>,
    },
    /// A source tensor is not F32.
    #[error("tensor {name} must be F32, found {actual:?}")]
    SourceDType {
        /// Tensor name.
        name: String,
        /// Actual MLX dtype.
        actual: MlxDtype,
    },
    /// The output path already exists.
    #[error("artifact output already exists: {0}")]
    OutputExists(PathBuf),
    /// A storage policy is not implemented by this artifact version.
    #[error("unsupported artifact storage for {name}: {storage:?}")]
    UnsupportedStorage {
        /// Tensor name.
        name: String,
        /// Requested storage policy.
        storage: Storage,
    },
    /// An array key does not exist in the artifact.
    #[error("missing artifact array {0}")]
    MissingArtifactArray(String),
    /// A requested tensor does not exist in the artifact manifest.
    #[error("missing artifact tensor {0}")]
    MissingArtifactTensor(String),
    /// A tensor method was called for an incompatible storage policy.
    #[error("tensor {name} has incompatible storage {storage:?}")]
    IncompatibleStorage {
        /// Tensor name.
        name: String,
        /// Actual storage policy.
        storage: Storage,
    },
}

/// Result type for artifact operations.
pub type ArtifactResult<T> = std::result::Result<T, ArtifactError>;

/// Public tensor information persisted in `manifest.json`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactTensorInfo {
    /// Runtime tensor dimensions after conversion transforms.
    pub shape: Vec<usize>,
    /// Runtime storage policy.
    pub storage: Storage,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ArtifactManifest {
    format_version: u32,
    model_id: String,
    tensors: HashMap<String, ArtifactTensorInfo>,
}

/// Loaded MLX release artifact.
#[derive(Debug)]
pub struct Artifact {
    manifest: ArtifactManifest,
    arrays: HashMap<String, Array>,
}

/// Converts the complete published Nemotron 3.5 checkpoint.
pub fn convert_model(source: impl AsRef<Path>, output: impl AsRef<Path>) -> ArtifactResult<()> {
    convert_model_with_group_size(source, output, 128)
}

/// Converts the complete checkpoint with the selected affine INT8 group size.
pub fn convert_model_with_group_size(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
    group_size: usize,
) -> ArtifactResult<()> {
    let source = source.as_ref();
    let output = output.as_ref();
    let manifest = ModelManifest::nemotron_3_5_with_group_size(group_size)
        .map_err(|error| ArtifactError::InvalidQuantization(error.to_string()))?;
    convert_tensors(source, output, manifest.model_id(), manifest.tensors())?;
    copy_model_companion_files(source, output)
}

/// Copies only tokenizer/configuration and license-notice files beside the source checkpoint.
pub fn copy_model_companion_files(
    source_model: impl AsRef<Path>,
    output: impl AsRef<Path>,
) -> ArtifactResult<()> {
    let source_dir = source_model
        .as_ref()
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let output = output.as_ref();
    for name in [
        "config.json",
        "processor_config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "README.md",
        "LICENSE",
        "LICENSE.md",
        "NOTICE",
        "NOTICE.md",
    ] {
        let source = source_dir.join(name);
        if source.is_file() {
            fs::copy(source, output.join(name))?;
        }
    }
    Ok(())
}

impl Artifact {
    /// Loads `manifest.json` and `weights.safetensors` from a directory.
    pub fn load(path: impl AsRef<Path>) -> ArtifactResult<Self> {
        let path = path.as_ref();
        let manifest = serde_json::from_reader(fs::File::open(path.join(MANIFEST_FILE))?)?;
        let arrays = Array::load_safetensors(path.join(WEIGHTS_FILE))?;
        Ok(Self { manifest, arrays })
    }

    /// Returns the persisted tensor information.
    pub fn tensor_info(&self, name: &str) -> Option<&ArtifactTensorInfo> {
        self.manifest.tensors.get(name)
    }

    /// Returns the element type of a stored MLX array.
    pub fn array_dtype(&self, name: &str) -> ArtifactResult<DType> {
        let array = self
            .arrays
            .get(name)
            .ok_or_else(|| ArtifactError::MissingArtifactArray(name.to_string()))?;
        mlx_dtype(array.dtype()).ok_or_else(|| ArtifactError::SourceDType {
            name: name.to_string(),
            actual: array.dtype(),
        })
    }

    /// Dequantizes a named affine INT8 tensor and returns row-major F32 values.
    pub fn dequantize_to_f32(&self, name: &str) -> ArtifactResult<Vec<f32>> {
        let info = self.require_tensor(name)?;
        let Storage::Int8Affine { group_size } = info.storage else {
            return Err(ArtifactError::IncompatibleStorage {
                name: name.to_string(),
                storage: info.storage,
            });
        };
        let qweight = self.require_array(&format!("{name}.__qweight"))?;
        let scales = self.require_array(&format!("{name}.__scales"))?;
        let biases = self.require_array(&format!("{name}.__biases"))?;
        let output = mlx_rs::ops::dequantize(qweight, scales, biases, group_size as i32, 8)?
            .as_type::<f32>()?;
        output.eval()?;
        Ok(output.try_as_slice::<f32>()?.to_vec())
    }

    /// Converts a named FP16 tensor to row-major F32 values.
    pub fn f16_to_f32(&self, name: &str) -> ArtifactResult<Vec<f32>> {
        let info = self.require_tensor(name)?;
        if info.storage != Storage::F16 {
            return Err(ArtifactError::IncompatibleStorage {
                name: name.to_string(),
                storage: info.storage,
            });
        }
        let output = self.require_array(name)?.as_type::<f32>()?;
        output.eval()?;
        Ok(output.try_as_slice::<f32>()?.to_vec())
    }

    pub(crate) fn quantized_arrays(
        &self,
        name: &str,
    ) -> ArtifactResult<(Array, Array, Array, usize, Vec<usize>)> {
        let info = self.require_tensor(name)?;
        let Storage::Int8Affine { group_size } = info.storage else {
            return Err(ArtifactError::IncompatibleStorage {
                name: name.to_string(),
                storage: info.storage,
            });
        };
        Ok((
            self.require_array(&format!("{name}.__qweight"))?.clone(),
            self.require_array(&format!("{name}.__scales"))?.clone(),
            self.require_array(&format!("{name}.__biases"))?.clone(),
            group_size,
            info.shape.clone(),
        ))
    }

    pub(crate) fn f16_array(&self, name: &str) -> ArtifactResult<Array> {
        let info = self.require_tensor(name)?;
        if info.storage != Storage::F16 {
            return Err(ArtifactError::IncompatibleStorage {
                name: name.to_string(),
                storage: info.storage,
            });
        }
        Ok(self.require_array(name)?.clone())
    }

    fn require_tensor(&self, name: &str) -> ArtifactResult<&ArtifactTensorInfo> {
        self.tensor_info(name)
            .ok_or_else(|| ArtifactError::MissingArtifactTensor(name.to_string()))
    }

    fn require_array(&self, name: &str) -> ArtifactResult<&Array> {
        self.arrays
            .get(name)
            .ok_or_else(|| ArtifactError::MissingArtifactArray(name.to_string()))
    }
}

/// Converts selected F32 safetensors into an atomic MLX release artifact directory.
pub fn convert_tensors(
    source: impl AsRef<Path>,
    output: impl AsRef<Path>,
    model_id: &str,
    specs: &[TensorSpec],
) -> ArtifactResult<()> {
    let source = source.as_ref();
    let output = output.as_ref();
    if output.exists() {
        return Err(ArtifactError::OutputExists(output.to_path_buf()));
    }

    let source_arrays = Array::load_safetensors(source)?;
    validate_specs(&source_arrays, specs)?;

    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = tempfile::Builder::new()
        .prefix(".nemotron-mlx-")
        .tempdir_in(parent)?;
    let staging_path = staging.path();

    let mut destination_arrays = HashMap::new();
    let mut tensor_info = HashMap::new();
    for spec in specs {
        let source_array = &source_arrays[&spec.name];
        let runtime_array = apply_transform(source_array, spec)?;
        match spec.storage {
            Storage::Int8Affine { group_size } => {
                let weight = runtime_array.as_type::<half::f16>()?;
                let (qweight, scales, biases) =
                    mlx_rs::ops::quantize(&weight, group_size as i32, 8)?;
                destination_arrays.insert(format!("{}.__qweight", spec.name), qweight);
                destination_arrays.insert(format!("{}.__scales", spec.name), scales);
                destination_arrays.insert(format!("{}.__biases", spec.name), biases);
            }
            Storage::F16 => {
                destination_arrays.insert(spec.name.clone(), runtime_array.as_type::<half::f16>()?);
            }
            storage => {
                return Err(ArtifactError::UnsupportedStorage {
                    name: spec.name.clone(),
                    storage,
                });
            }
        }
        tensor_info.insert(
            spec.name.clone(),
            ArtifactTensorInfo {
                shape: spec.artifact_shape.clone(),
                storage: spec.storage,
            },
        );
    }

    let weights_tmp = staging_path.join("weights.tmp.safetensors");
    Array::save_safetensors(&destination_arrays, None, &weights_tmp)?;
    fs::rename(weights_tmp, staging_path.join(WEIGHTS_FILE))?;

    let manifest = ArtifactManifest {
        format_version: FORMAT_VERSION,
        model_id: model_id.to_string(),
        tensors: tensor_info,
    };
    let manifest_tmp = staging_path.join("manifest.tmp.json");
    serde_json::to_writer_pretty(fs::File::create(&manifest_tmp)?, &manifest)?;
    fs::rename(manifest_tmp, staging_path.join(MANIFEST_FILE))?;

    let persisted = staging.keep();
    fs::rename(persisted, output)?;
    Ok(())
}

fn validate_specs(source: &HashMap<String, Array>, specs: &[TensorSpec]) -> ArtifactResult<()> {
    for spec in specs {
        let array = source
            .get(&spec.name)
            .ok_or_else(|| ArtifactError::MissingSourceTensor(spec.name.clone()))?;
        let actual_shape = array
            .shape()
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        if actual_shape != spec.source_shape {
            return Err(ArtifactError::ShapeMismatch {
                name: spec.name.clone(),
                expected: spec.source_shape.clone(),
                actual: actual_shape,
            });
        }
        if array.dtype() != MlxDtype::Float32 {
            return Err(ArtifactError::SourceDType {
                name: spec.name.clone(),
                actual: array.dtype(),
            });
        }
    }
    Ok(())
}

fn apply_transform(array: &Array, spec: &TensorSpec) -> ArtifactResult<Array> {
    match spec.transform {
        TensorTransform::Identity => Ok(array.clone()),
        TensorTransform::SqueezeTrailingUnitDimensions => {
            let shape = spec
                .artifact_shape
                .iter()
                .map(|value| *value as i32)
                .collect::<Vec<_>>();
            Ok(array.reshape(&shape)?)
        }
    }
}

fn mlx_dtype(dtype: MlxDtype) -> Option<DType> {
    match dtype {
        MlxDtype::Float16 => Some(DType::F16),
        MlxDtype::Float32 => Some(DType::F32),
        MlxDtype::Int32 => Some(DType::I32),
        MlxDtype::Uint32 => Some(DType::U32),
        _ => None,
    }
}

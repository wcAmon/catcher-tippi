use std::collections::BTreeMap;

/// Tensor element types accepted in source and release artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// IEEE half precision.
    F16,
    /// IEEE single precision.
    F32,
    /// Signed 32-bit integer.
    I32,
    /// Unsigned 32-bit integer used for packed MLX quantized weights.
    U32,
}

/// On-disk representation selected for a tensor in the MLX artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// MLX affine weight-only INT8 with per-group scale and bias.
    Int8Affine { group_size: usize },
    /// IEEE half precision.
    F16,
    /// IEEE single precision.
    F32,
    /// Signed 32-bit integer.
    I32,
}

/// Shape-only transform applied before quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorTransform {
    /// Preserve the source tensor shape.
    Identity,
    /// Remove trailing unit kernel dimensions so a pointwise convolution can use matmul.
    SqueezeTrailingUnitDimensions,
}

/// Metadata read from an input tensor index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    /// Dimensions in row-major order.
    pub shape: Vec<usize>,
    /// Source element type.
    pub dtype: DType,
}

/// Tensor metadata keyed by the Hugging Face checkpoint tensor name.
pub type TensorIndex = BTreeMap<String, TensorMetadata>;

/// Required source tensor and its destination storage policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Hugging Face safetensors name.
    pub name: String,
    /// Expected source dimensions.
    pub source_shape: Vec<usize>,
    /// Dimensions after applying `transform`.
    pub artifact_shape: Vec<usize>,
    /// Destination precision and quantization policy.
    pub storage: Storage,
    /// Shape transform applied before storage.
    pub transform: TensorTransform,
}

/// Errors found while validating a source checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ManifestError {
    /// A required tensor is absent.
    #[error("missing tensor {0}")]
    MissingTensor(String),
    /// A tensor has dimensions different from the published checkpoint.
    #[error("tensor {name} shape mismatch: expected {expected:?}, found {actual:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Published dimensions.
        expected: Vec<usize>,
        /// Input dimensions.
        actual: Vec<usize>,
    },
    /// A tensor uses a source element type other than F32.
    #[error("tensor {name} dtype mismatch: expected {expected:?}, found {actual:?}")]
    DTypeMismatch {
        /// Tensor name.
        name: String,
        /// Published source type.
        expected: DType,
        /// Input source type.
        actual: DType,
    },
}

/// Published Nemotron 3.5 checkpoint layout and conversion policy.
#[derive(Debug, Clone)]
pub struct ModelManifest {
    model_id: &'static str,
    encoder_layers: usize,
    vocab_size: usize,
    tensors: Vec<TensorSpec>,
}

impl ModelManifest {
    /// Builds the exact 655-tensor layout of the published 0.6B checkpoint.
    pub fn nemotron_3_5() -> Self {
        let mut tensors = Vec::with_capacity(655);

        for layer in 0..24 {
            add_encoder_layer(&mut tensors, layer);
        }

        add_f16(&mut tensors, "encoder.subsampling.conv_in.bias", &[256]);
        add_f16(
            &mut tensors,
            "encoder.subsampling.conv_in.weight",
            &[256, 1, 3, 3],
        );
        for layer in 0..2 {
            add_f16(
                &mut tensors,
                &format!("encoder.subsampling.layers.{layer}.depthwise_conv.bias"),
                &[256],
            );
            add_f16(
                &mut tensors,
                &format!("encoder.subsampling.layers.{layer}.depthwise_conv.weight"),
                &[256, 1, 3, 3],
            );
            add_f16(
                &mut tensors,
                &format!("encoder.subsampling.layers.{layer}.pointwise_conv.bias"),
                &[256],
            );
            add_int8_pointwise(
                &mut tensors,
                &format!("encoder.subsampling.layers.{layer}.pointwise_conv.weight"),
                &[256, 256, 1, 1],
            );
        }
        add_f16(&mut tensors, "encoder.subsampling.linear.bias", &[1024]);
        add_int8(
            &mut tensors,
            "encoder.subsampling.linear.weight",
            &[1024, 4352],
        );

        add_int8(&mut tensors, "encoder_projector.weight", &[640, 1024]);
        add_f16(&mut tensors, "encoder_projector.bias", &[640]);
        add_int8(
            &mut tensors,
            "prompt_projector.linear_1.weight",
            &[2048, 1152],
        );
        add_f16(&mut tensors, "prompt_projector.linear_1.bias", &[2048]);
        add_int8(
            &mut tensors,
            "prompt_projector.linear_2.weight",
            &[1024, 2048],
        );
        add_f16(&mut tensors, "prompt_projector.linear_2.bias", &[1024]);

        add_int8(&mut tensors, "decoder.embedding.weight", &[13_088, 640]);
        for layer in 0..2 {
            add_int8(
                &mut tensors,
                &format!("decoder.lstm.weight_ih_l{layer}"),
                &[2560, 640],
            );
            add_int8(
                &mut tensors,
                &format!("decoder.lstm.weight_hh_l{layer}"),
                &[2560, 640],
            );
            add_f16(
                &mut tensors,
                &format!("decoder.lstm.bias_ih_l{layer}"),
                &[2560],
            );
            add_f16(
                &mut tensors,
                &format!("decoder.lstm.bias_hh_l{layer}"),
                &[2560],
            );
        }
        add_int8(
            &mut tensors,
            "decoder.decoder_projector.weight",
            &[640, 640],
        );
        add_f16(&mut tensors, "decoder.decoder_projector.bias", &[640]);
        add_int8(&mut tensors, "joint.head.weight", &[13_088, 640]);
        add_f16(&mut tensors, "joint.head.bias", &[13_088]);

        debug_assert_eq!(tensors.len(), 655);
        Self {
            model_id: "nvidia/nemotron-3.5-asr-streaming-0.6b",
            encoder_layers: 24,
            vocab_size: 13_088,
            tensors,
        }
    }

    /// Hugging Face repository identifier.
    pub fn model_id(&self) -> &str {
        self.model_id
    }

    /// Number of FastConformer blocks.
    pub fn encoder_layers(&self) -> usize {
        self.encoder_layers
    }

    /// Token count including the RNNT blank token.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Ordered tensor specifications.
    pub fn tensors(&self) -> &[TensorSpec] {
        &self.tensors
    }

    /// Looks up a tensor by its published checkpoint name.
    pub fn tensor(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Returns the total number of scalar parameters in source tensors.
    pub fn parameter_count(&self) -> usize {
        self.tensors
            .iter()
            .map(|tensor| tensor.source_shape.iter().product::<usize>())
            .sum()
    }

    /// Verifies all required tensors, shapes, and source dtypes.
    pub fn validate(&self, index: &TensorIndex) -> Result<(), ManifestError> {
        for spec in &self.tensors {
            let actual = index
                .get(&spec.name)
                .ok_or_else(|| ManifestError::MissingTensor(spec.name.clone()))?;
            if actual.shape != spec.source_shape {
                return Err(ManifestError::ShapeMismatch {
                    name: spec.name.clone(),
                    expected: spec.source_shape.clone(),
                    actual: actual.shape.clone(),
                });
            }
            if actual.dtype != DType::F32 {
                return Err(ManifestError::DTypeMismatch {
                    name: spec.name.clone(),
                    expected: DType::F32,
                    actual: actual.dtype,
                });
            }
        }
        Ok(())
    }
}

fn add_encoder_layer(tensors: &mut Vec<TensorSpec>, layer: usize) {
    let prefix = format!("encoder.layers.{layer}");
    add_f16(
        tensors,
        &format!("{prefix}.conv.depthwise_conv.weight"),
        &[1024, 1, 9],
    );
    for parameter in ["weight", "bias"] {
        add_f16(tensors, &format!("{prefix}.conv.norm.{parameter}"), &[1024]);
    }
    add_int8_pointwise(
        tensors,
        &format!("{prefix}.conv.pointwise_conv1.weight"),
        &[2048, 1024, 1],
    );
    add_int8_pointwise(
        tensors,
        &format!("{prefix}.conv.pointwise_conv2.weight"),
        &[1024, 1024, 1],
    );

    for feed_forward in ["feed_forward1", "feed_forward2"] {
        add_int8(
            tensors,
            &format!("{prefix}.{feed_forward}.linear1.weight"),
            &[4096, 1024],
        );
        add_int8(
            tensors,
            &format!("{prefix}.{feed_forward}.linear2.weight"),
            &[1024, 4096],
        );
    }

    for norm in [
        "norm_conv",
        "norm_feed_forward1",
        "norm_feed_forward2",
        "norm_out",
        "norm_self_att",
    ] {
        for parameter in ["weight", "bias"] {
            add_f16(tensors, &format!("{prefix}.{norm}.{parameter}"), &[1024]);
        }
    }

    add_f16(tensors, &format!("{prefix}.self_attn.bias_u"), &[8, 128]);
    add_f16(tensors, &format!("{prefix}.self_attn.bias_v"), &[8, 128]);
    for projection in ["k_proj", "o_proj", "q_proj", "relative_k_proj", "v_proj"] {
        add_int8(
            tensors,
            &format!("{prefix}.self_attn.{projection}.weight"),
            &[1024, 1024],
        );
    }
}

fn add_int8(tensors: &mut Vec<TensorSpec>, name: &str, shape: &[usize]) {
    debug_assert_eq!(shape.len(), 2);
    debug_assert_eq!(shape[1] % 128, 0);
    tensors.push(TensorSpec {
        name: name.to_string(),
        source_shape: shape.to_vec(),
        artifact_shape: shape.to_vec(),
        storage: Storage::Int8Affine { group_size: 128 },
        transform: TensorTransform::Identity,
    });
}

fn add_int8_pointwise(tensors: &mut Vec<TensorSpec>, name: &str, shape: &[usize]) {
    debug_assert!(shape.len() >= 3);
    debug_assert!(shape[2..].iter().all(|dimension| *dimension == 1));
    debug_assert_eq!(shape[1] % 128, 0);
    tensors.push(TensorSpec {
        name: name.to_string(),
        source_shape: shape.to_vec(),
        artifact_shape: shape[..2].to_vec(),
        storage: Storage::Int8Affine { group_size: 128 },
        transform: TensorTransform::SqueezeTrailingUnitDimensions,
    });
}

fn add_f16(tensors: &mut Vec<TensorSpec>, name: &str, shape: &[usize]) {
    tensors.push(TensorSpec {
        name: name.to_string(),
        source_shape: shape.to_vec(),
        artifact_shape: shape.to_vec(),
        storage: Storage::F16,
        transform: TensorTransform::Identity,
    });
}

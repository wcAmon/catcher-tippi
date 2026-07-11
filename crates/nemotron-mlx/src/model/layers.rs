use half::f16;
use mlx_rs::Array;

use super::{CausalConv1dCache, CausalConv2dCache, ModelError, ModelResult};
use crate::weights::Artifact;

/// Owned row-major rank-three tensor returned to Rust control flow.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor3 {
    /// `[batch, time, channels]` dimensions.
    pub shape: [usize; 3],
    /// Row-major values.
    pub values: Vec<f32>,
}

/// Owned NHWC tensor whose second dimension is streaming time.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor4 {
    /// `[batch, time, frequency, channels]` dimensions.
    pub shape: [usize; 4],
    /// Row-major values.
    pub values: Vec<f32>,
}

/// FP16 MLX Conv2D with NeMo's asymmetric causal time/frequency padding.
#[derive(Debug)]
pub struct Fp16Conv2d {
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    stride: usize,
    groups: usize,
    weight: Array,
    bias: Array,
}

impl Fp16Conv2d {
    pub(crate) fn streaming_output_length(&self, input_frames: usize) -> usize {
        let left_pad = self.kernel_size - self.stride;
        if input_frames + left_pad < self.kernel_size {
            0
        } else {
            (input_frames + left_pad - self.kernel_size) / self.stride + 1
        }
    }

    pub fn from_artifact(
        artifact: &Artifact,
        weight_name: &str,
        bias_name: &str,
        stride: usize,
        groups: usize,
    ) -> ModelResult<Self> {
        let shape = artifact
            .tensor_info(weight_name)
            .ok_or_else(|| {
                crate::weights::ArtifactError::MissingArtifactTensor(weight_name.to_string())
            })?
            .shape
            .clone();
        if shape.len() != 4 || shape[2] != shape[3] {
            return Err(ModelError::InvalidShape(format!(
                "Conv2D artifact {weight_name} must have OIHW shape"
            )));
        }
        Self::from_f32(
            &artifact.f16_to_f32(weight_name)?,
            &artifact.f16_to_f32(bias_name)?,
            shape[0],
            shape[1] * groups,
            shape[2],
            stride,
            groups,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_f32(
        pytorch_weight: &[f32],
        bias: &[f32],
        output_channels: usize,
        input_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> ModelResult<Self> {
        if groups == 0
            || input_channels % groups != 0
            || output_channels % groups != 0
            || stride == 0
            || bias.len() != output_channels
            || pytorch_weight.len()
                != output_channels * (input_channels / groups) * kernel_size * kernel_size
        {
            return Err(ModelError::InvalidShape(
                "Conv2D weight, bias, stride, or groups are invalid".to_string(),
            ));
        }

        // PyTorch OIHW -> MLX OHWI.
        let channels_per_group = input_channels / groups;
        let mut mlx_weight = vec![0.0; pytorch_weight.len()];
        for output in 0..output_channels {
            for input in 0..channels_per_group {
                for kernel_t in 0..kernel_size {
                    for kernel_f in 0..kernel_size {
                        let source = (((output * channels_per_group + input) * kernel_size
                            + kernel_t)
                            * kernel_size)
                            + kernel_f;
                        let destination = (((output * kernel_size + kernel_t) * kernel_size
                            + kernel_f)
                            * channels_per_group)
                            + input;
                        mlx_weight[destination] = pytorch_weight[source];
                    }
                }
            }
        }
        let weight = Array::from_slice(
            &mlx_weight,
            &[
                output_channels as i32,
                kernel_size as i32,
                kernel_size as i32,
                channels_per_group as i32,
            ],
        )
        .as_type::<f16>()?;
        let bias = Array::from_slice(bias, &[output_channels as i32]).as_type::<f16>()?;
        Ok(Self {
            input_channels,
            output_channels,
            kernel_size,
            stride,
            groups,
            weight,
            bias,
        })
    }

    pub fn forward_causal(
        &self,
        input: &Tensor4,
        cache: &mut CausalConv2dCache,
    ) -> ModelResult<Tensor4> {
        let [batch, time, frequency, channels] = input.shape;
        if batch != 1
            || channels != self.input_channels
            || input.values.len() != batch * time * frequency * channels
        {
            return Err(ModelError::InvalidShape(format!(
                "Conv2D input must be [1,time,freq,{}]",
                self.input_channels
            )));
        }
        let (combined, combined_time) =
            cache.prepend_and_update(&input.values, time, frequency, channels)?;
        let left_frequency_pad = self.kernel_size - 1;
        let right_frequency_pad = self.stride - 1;
        let padded_frequency = frequency + left_frequency_pad + right_frequency_pad;
        let mut padded = vec![0.0; combined_time * padded_frequency * channels];
        for frame in 0..combined_time {
            for bin in 0..frequency {
                let source = (frame * frequency + bin) * channels;
                let destination = (frame * padded_frequency + bin + left_frequency_pad) * channels;
                padded[destination..destination + channels]
                    .copy_from_slice(&combined[source..source + channels]);
            }
        }
        let input = Array::from_slice(
            &padded,
            &[
                1,
                combined_time as i32,
                padded_frequency as i32,
                channels as i32,
            ],
        )
        .as_type::<f16>()?;
        let output = mlx_rs::ops::conv2d(
            &input,
            &self.weight,
            (self.stride as i32, self.stride as i32),
            (0, 0),
            (1, 1),
            self.groups as i32,
        )?
        .add(&self.bias)?
        .as_type::<f32>()?;
        let shape = output.shape();
        let output_shape = [
            shape[0] as usize,
            shape[1] as usize,
            shape[2] as usize,
            shape[3] as usize,
        ];
        output.eval()?;
        let values = output.try_as_slice::<f32>()?.to_vec();
        debug_assert_eq!(output_shape[3], self.output_channels);
        Ok(Tensor4 {
            shape: output_shape,
            values,
        })
    }
}

/// Affine INT8 MLX linear layer with an FP16 output bias.
#[derive(Debug)]
pub struct QuantizedLinear {
    input_dims: usize,
    output_dims: usize,
    group_size: usize,
    qweight: Array,
    scales: Array,
    quant_biases: Array,
    output_bias: Array,
}

impl QuantizedLinear {
    /// Binds directly to packed affine INT8 arrays from a converted artifact.
    pub fn from_artifact(
        artifact: &Artifact,
        weight_name: &str,
        bias_name: Option<&str>,
    ) -> ModelResult<Self> {
        let (qweight, scales, quant_biases, group_size, shape) =
            artifact.quantized_arrays(weight_name)?;
        if shape.len() != 2 {
            return Err(ModelError::InvalidShape(format!(
                "linear artifact {weight_name} must have rank 2, found {shape:?}"
            )));
        }
        let output_dims = shape[0];
        let input_dims = shape[1];
        let output_bias = if let Some(name) = bias_name {
            let bias = artifact.f16_array(name)?;
            if bias.shape() != [output_dims as i32] {
                return Err(ModelError::InvalidShape(format!(
                    "linear bias {name} must have shape [{output_dims}]"
                )));
            }
            bias
        } else {
            Array::zeros::<f16>(&[output_dims as i32])?
        };
        Ok(Self {
            input_dims,
            output_dims,
            group_size,
            qweight,
            scales,
            quant_biases,
            output_bias,
        })
    }

    /// Quantizes an F32 row-major `[output_dims, input_dims]` weight matrix.
    pub fn from_f32(
        weight: &[f32],
        output_dims: usize,
        input_dims: usize,
        bias: &[f32],
        group_size: usize,
    ) -> ModelResult<Self> {
        validate_linear_shapes(weight, output_dims, input_dims, bias, group_size)?;
        let weight =
            Array::from_slice(weight, &[output_dims as i32, input_dims as i32]).as_type::<f16>()?;
        let (qweight, scales, quant_biases) = mlx_rs::ops::quantize(&weight, group_size as i32, 8)?;
        let output_bias = Array::from_slice(bias, &[output_dims as i32]).as_type::<f16>()?;
        Ok(Self {
            input_dims,
            output_dims,
            group_size,
            qweight,
            scales,
            quant_biases,
            output_bias,
        })
    }

    /// Runs a row-major `[rows, input_dims]` F32 input and returns F32 output.
    pub fn forward_f32(&self, input: &[f32], rows: usize) -> ModelResult<Vec<f32>> {
        if rows.checked_mul(self.input_dims) != Some(input.len()) {
            return Err(ModelError::InvalidShape(format!(
                "linear input has {} values, expected {}x{}",
                input.len(),
                rows,
                self.input_dims
            )));
        }
        let input = Array::from_slice(input, &[rows as i32, self.input_dims as i32]);
        let output = self.forward_array(&input)?.as_type::<f32>()?;
        output.eval()?;
        Ok(output.try_as_slice::<f32>()?.to_vec())
    }

    pub(crate) fn forward_array(&self, input: &Array) -> ModelResult<Array> {
        let input = input.as_type::<f16>()?;
        let output = mlx_rs::ops::quantized_matmul(
            &input,
            &self.qweight,
            &self.scales,
            &self.quant_biases,
            true,
            self.group_size as i32,
            8,
        )?;
        Ok(output.add(&self.output_bias)?)
    }

    pub(crate) fn output_dims(&self) -> usize {
        self.output_dims
    }

    pub(crate) fn input_dims(&self) -> usize {
        self.input_dims
    }
}

/// Quantized 1x1 convolution over `[batch, time, channels]` input.
#[derive(Debug)]
pub struct PointwiseConv1d {
    linear: QuantizedLinear,
}

impl PointwiseConv1d {
    /// Quantizes pointwise weights represented as a squeezed matrix.
    pub fn from_f32(
        weight: &[f32],
        output_dims: usize,
        input_dims: usize,
        bias: &[f32],
        group_size: usize,
    ) -> ModelResult<Self> {
        Ok(Self {
            linear: QuantizedLinear::from_f32(weight, output_dims, input_dims, bias, group_size)?,
        })
    }

    /// Runs pointwise convolution and returns `[batch, time, output_channels]`.
    pub fn forward_f32(&self, input: &[f32], batch: usize, time: usize) -> ModelResult<Tensor3> {
        let rows = batch
            .checked_mul(time)
            .ok_or_else(|| ModelError::InvalidShape("pointwise row count overflow".to_string()))?;
        let values = self.linear.forward_f32(input, rows)?;
        Ok(Tensor3 {
            shape: [batch, time, self.linear.output_dims()],
            values,
        })
    }
}

/// FP16 MLX depthwise causal convolution.
#[derive(Debug)]
pub struct DepthwiseConv1d {
    channels: usize,
    kernel_size: usize,
    weight: Array,
    bias: Array,
}

impl DepthwiseConv1d {
    pub fn from_artifact(
        artifact: &Artifact,
        weight_name: &str,
        bias_name: Option<&str>,
    ) -> ModelResult<Self> {
        let shape = artifact
            .tensor_info(weight_name)
            .ok_or_else(|| {
                crate::weights::ArtifactError::MissingArtifactTensor(weight_name.to_string())
            })?
            .shape
            .clone();
        if shape.len() != 3 || shape[1] != 1 {
            return Err(ModelError::InvalidShape(format!(
                "depthwise artifact {weight_name} must have [channels,1,kernel] shape"
            )));
        }
        let bias = if let Some(name) = bias_name {
            artifact.f16_to_f32(name)?
        } else {
            vec![0.0; shape[0]]
        };
        Self::from_f32(
            &artifact.f16_to_f32(weight_name)?,
            shape[0],
            shape[2],
            &bias,
        )
    }

    /// Creates weights from `[channels, kernel_size]` cross-correlation kernels.
    pub fn from_f32(
        weight: &[f32],
        channels: usize,
        kernel_size: usize,
        bias: &[f32],
    ) -> ModelResult<Self> {
        if channels.checked_mul(kernel_size) != Some(weight.len()) || bias.len() != channels {
            return Err(ModelError::InvalidShape(format!(
                "depthwise weight/bias mismatch for {channels} channels and kernel {kernel_size}"
            )));
        }
        let weight = Array::from_slice(weight, &[channels as i32, kernel_size as i32, 1])
            .as_type::<f16>()?;
        let bias = Array::from_slice(bias, &[channels as i32]).as_type::<f16>()?;
        Ok(Self {
            channels,
            kernel_size,
            weight,
            bias,
        })
    }

    /// Runs one causal chunk. The first implementation supports one stream per cache.
    pub fn forward_f32(
        &self,
        input: &[f32],
        batch: usize,
        time: usize,
        cache: &mut CausalConv1dCache,
    ) -> ModelResult<Tensor3> {
        if batch != 1 || input.len() != time * self.channels {
            return Err(ModelError::InvalidShape(format!(
                "depthwise input requires batch 1 and shape [1,{time},{}]",
                self.channels
            )));
        }
        let combined = cache.prepend_and_update(input, time)?;
        if cache.values().len() != (self.kernel_size - 1) * self.channels {
            return Err(ModelError::InvalidShape(format!(
                "depthwise cache must hold {} left frames",
                self.kernel_size - 1
            )));
        }
        let input = Array::from_slice(
            &combined,
            &[
                1,
                (time + self.kernel_size - 1) as i32,
                self.channels as i32,
            ],
        )
        .as_type::<f16>()?;
        let output = mlx_rs::ops::conv1d(&input, &self.weight, 1, 0, 1, self.channels as i32)?
            .add(&self.bias)?
            .as_type::<f32>()?;
        output.eval()?;
        Ok(Tensor3 {
            shape: [1, time, self.channels],
            values: output.try_as_slice::<f32>()?.to_vec(),
        })
    }
}

/// FP16 MLX layer normalization over the last dimension.
#[derive(Debug)]
pub struct LayerNorm {
    dimensions: usize,
    weight: Array,
    bias: Array,
    epsilon: f32,
}

impl LayerNorm {
    pub fn from_artifact(artifact: &Artifact, prefix: &str, epsilon: f32) -> ModelResult<Self> {
        Self::from_f32(
            &artifact.f16_to_f32(&format!("{prefix}.weight"))?,
            &artifact.f16_to_f32(&format!("{prefix}.bias"))?,
            epsilon,
        )
    }

    /// Creates a layer normalization from F32 affine parameters.
    pub fn from_f32(weight: &[f32], bias: &[f32], epsilon: f32) -> ModelResult<Self> {
        if weight.is_empty() || weight.len() != bias.len() {
            return Err(ModelError::InvalidShape(
                "layer norm weight and bias lengths must match".to_string(),
            ));
        }
        Ok(Self {
            dimensions: weight.len(),
            weight: Array::from_slice(weight, &[weight.len() as i32]).as_type::<f16>()?,
            bias: Array::from_slice(bias, &[bias.len() as i32]).as_type::<f16>()?,
            epsilon,
        })
    }

    /// Runs row-major F32 rows and returns F32 values.
    pub fn forward_f32(&self, input: &[f32], rows: usize) -> ModelResult<Vec<f32>> {
        if rows.checked_mul(self.dimensions) != Some(input.len()) {
            return Err(ModelError::InvalidShape(format!(
                "layer norm input has {} values, expected {}x{}",
                input.len(),
                rows,
                self.dimensions
            )));
        }
        let input =
            Array::from_slice(input, &[rows as i32, self.dimensions as i32]).as_type::<f16>()?;
        let output = mlx_rs::fast::layer_norm(&input, &self.weight, &self.bias, self.epsilon)?
            .as_type::<f32>()?;
        output.eval()?;
        Ok(output.try_as_slice::<f32>()?.to_vec())
    }
}

fn validate_linear_shapes(
    weight: &[f32],
    output_dims: usize,
    input_dims: usize,
    bias: &[f32],
    group_size: usize,
) -> ModelResult<()> {
    if output_dims.checked_mul(input_dims) != Some(weight.len()) || bias.len() != output_dims {
        return Err(ModelError::InvalidShape(format!(
            "linear weight/bias mismatch for {output_dims}x{input_dims}"
        )));
    }
    if group_size == 0 || input_dims % group_size != 0 {
        return Err(ModelError::InvalidShape(format!(
            "linear input dimension {input_dims} is incompatible with group size {group_size}"
        )));
    }
    Ok(())
}

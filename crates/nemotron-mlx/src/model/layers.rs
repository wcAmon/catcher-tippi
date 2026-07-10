use half::f16;
use mlx_rs::Array;

use super::{CausalConv1dCache, ModelError, ModelResult};

/// Owned row-major rank-three tensor returned to Rust control flow.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor3 {
    /// `[batch, time, channels]` dimensions.
    pub shape: [usize; 3],
    /// Row-major values.
    pub values: Vec<f32>,
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

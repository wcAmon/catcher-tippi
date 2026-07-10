//! Narrow safe interface over the MLX-C-backed Rust bindings.

use mlx_rs::{Array, Device, DeviceType};

/// Errors produced by the MLX backend boundary.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// MLX rejected an operation.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),

    /// The evaluated MLX array did not have the requested element type.
    #[error(transparent)]
    ArraySlice(#[from] mlx_rs::error::AsSliceError),

    /// A matrix shape does not match the supplied buffers or quantization group.
    #[error("invalid matrix shape: {0}")]
    InvalidShape(String),

    /// MLX-C returned a non-zero status.
    #[error("MLX-C call failed with status {0}")]
    CApi(i32),
}

/// Result type for backend operations.
pub type Result<T> = std::result::Result<T, BackendError>;

/// Returns whether the linked MLX runtime reports an available Metal backend.
pub fn is_metal_available() -> Result<bool> {
    let mut available = false;
    let status = unsafe { mlx_sys::mlx_metal_is_available(&mut available) };
    if status != 0 {
        return Err(BackendError::CApi(status));
    }

    let device = Device::gpu();
    Ok(available && matches!(device.get_type()?, DeviceType::Gpu))
}

/// Quantizes `weight` to affine INT8 in MLX and computes `input @ weight.T`.
pub fn quantized_matmul(
    input: &[f32],
    rows: usize,
    input_dims: usize,
    weight: &[f32],
    output_dims: usize,
    group_size: usize,
) -> Result<Vec<f32>> {
    if rows.checked_mul(input_dims) != Some(input.len()) {
        return Err(BackendError::InvalidShape(format!(
            "input has {} values, expected {}x{}",
            input.len(),
            rows,
            input_dims
        )));
    }
    if output_dims.checked_mul(input_dims) != Some(weight.len()) {
        return Err(BackendError::InvalidShape(format!(
            "weight has {} values, expected {}x{}",
            weight.len(),
            output_dims,
            input_dims
        )));
    }
    if group_size == 0 {
        return Err(BackendError::InvalidShape(
            "group size must be positive".to_string(),
        ));
    }
    if input_dims % group_size != 0 {
        return Err(BackendError::InvalidShape(format!(
            "input dimension {input_dims} is not divisible by group size {group_size}"
        )));
    }

    let gpu = Device::gpu();
    Device::set_default(&gpu);

    let input = Array::from_slice(input, &[rows as i32, input_dims as i32]);
    let weight = Array::from_slice(weight, &[output_dims as i32, input_dims as i32]);
    let (packed, scales, biases) = mlx_rs::ops::quantize(&weight, group_size as i32, 8)?;
    let output = mlx_rs::ops::quantized_matmul(
        &input,
        &packed,
        &scales,
        &biases,
        true,
        group_size as i32,
        8,
    )?;

    output.eval()?;
    Ok(output.try_as_slice::<f32>()?.to_vec())
}

//! MLX-backed Nemotron model primitives and streaming state.

mod cache;
mod layers;

pub use cache::CausalConv1dCache;
pub use layers::{DepthwiseConv1d, LayerNorm, PointwiseConv1d, QuantizedLinear, Tensor3};

/// Errors produced by model layers and cache management.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// MLX rejected an operation.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
    /// An evaluated MLX array could not be read.
    #[error(transparent)]
    ArraySlice(#[from] mlx_rs::error::AsSliceError),
    /// Input, weight, or cache dimensions are inconsistent.
    #[error("invalid model shape: {0}")]
    InvalidShape(String),
}

/// Result type for model operations.
pub type ModelResult<T> = std::result::Result<T, ModelError>;

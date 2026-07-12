//! MLX-backed Sortformer model modules.

mod encoder;
pub(crate) mod ops;
mod transformer;

pub use encoder::{Encoder, EncoderTrace};
pub use transformer::Diarizer;

/// Errors produced by Sortformer model layers.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    /// MLX rejected an operation.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
    /// An evaluated MLX array could not be read.
    #[error(transparent)]
    ArraySlice(#[from] mlx_rs::error::AsSliceError),
    /// A converted model artifact is missing or has incompatible storage.
    #[error(transparent)]
    Artifact(#[from] nemotron_mlx::weights::ArtifactError),
    /// A reused nemotron-mlx layer primitive failed.
    #[error(transparent)]
    Layer(#[from] nemotron_mlx::model::ModelError),
    /// Input, weight, or configuration dimensions are inconsistent.
    #[error("invalid model shape: {0}")]
    InvalidShape(String),
}

/// Result type for model operations.
pub type ModelResult<T> = std::result::Result<T, ModelError>;

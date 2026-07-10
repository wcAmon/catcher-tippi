//! MLX-backed Nemotron model primitives and streaming state.

mod cache;
mod encoder;
mod layers;
mod prompt;

pub use cache::CausalConv1dCache;
pub use encoder::{EncoderConfig, StreamingChunkPlan};
pub use layers::{DepthwiseConv1d, LayerNorm, PointwiseConv1d, QuantizedLinear, Tensor3};
pub use prompt::{LanguagePrompt, PromptProjector};

/// Errors produced by model layers and cache management.
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
    Artifact(#[from] crate::weights::ArtifactError),
    /// Input, weight, or cache dimensions are inconsistent.
    #[error("invalid model shape: {0}")]
    InvalidShape(String),
    /// The requested language is not present in the checkpoint prompt dictionary.
    #[error("unsupported language prompt: {0}")]
    UnsupportedLanguage(String),
    /// The checkpoint was not trained for the requested right attention context.
    #[error("unsupported lookahead {requested}; supported values are {supported:?}")]
    UnsupportedLookahead {
        /// Requested lookahead in subsampled frames.
        requested: usize,
        /// Values recorded in the checkpoint configuration.
        supported: [usize; 4],
    },
}

/// Result type for model operations.
pub type ModelResult<T> = std::result::Result<T, ModelError>;

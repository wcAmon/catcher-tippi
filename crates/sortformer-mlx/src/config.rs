//! Model configuration parsed from the exported NeMo `config.json`.

use std::fs;
use std::path::Path;

/// Errors loading or interpreting the model configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The configuration file could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// The configuration JSON is invalid or missing required keys.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, serde::Deserialize)]
struct RawConfig {
    preprocessor: RawPreprocessor,
    encoder: RawEncoder,
    transformer_encoder: RawTransformer,
    sortformer_modules: RawSortformerModules,
}

#[derive(Debug, serde::Deserialize)]
struct RawPreprocessor {
    sample_rate: usize,
    features: usize,
    window_size: f64,
    window_stride: f64,
    #[serde(default)]
    n_fft: Option<usize>,
}

#[derive(Debug, serde::Deserialize)]
struct RawEncoder {
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    conv_kernel_size: usize,
    subsampling_factor: usize,
    subsampling_conv_channels: usize,
    /// NeMo's `RelPositionalEncoding` input scaling flag. Published model
    /// artifacts may omit this key entirely, so it defaults to `true` to
    /// match NeMo's own default and the checkpoints seen so far.
    #[serde(default = "default_true")]
    xscaling: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, serde::Deserialize)]
struct RawTransformer {
    num_layers: usize,
    hidden_size: usize,
    inner_size: usize,
    num_attention_heads: usize,
}

#[derive(Debug, serde::Deserialize)]
struct RawSortformerModules {
    num_spks: usize,
}

/// Validated architecture and audio-frontend parameters.
#[derive(Debug, Clone)]
pub struct SortformerConfig {
    pub sample_rate: usize,
    pub n_mels: usize,
    pub window_seconds: f64,
    pub hop_seconds: f64,
    pub n_fft: usize,
    pub preemphasis: f32,
    pub encoder_layers: usize,
    pub encoder_dim: usize,
    pub encoder_heads: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_channels: usize,
    pub transformer_layers: usize,
    pub transformer_dim: usize,
    pub transformer_inner_dim: usize,
    pub transformer_heads: usize,
    pub num_speakers: usize,
    /// Whether the encoder scales subsampled features by `sqrt(d_model)`
    /// before the first Conformer block (NeMo `RelPositionalEncoding`
    /// `xscaling`). Defaults to `true` when absent from the config.
    pub xscaling: bool,
}

impl SortformerConfig {
    /// Reads `config.json` from a converted artifact directory.
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        Self::from_json(&fs::read_to_string(model_dir.as_ref().join("config.json"))?)
    }

    /// Parses the exported NeMo configuration JSON.
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig = serde_json::from_str(json)?;
        Ok(Self {
            sample_rate: raw.preprocessor.sample_rate,
            n_mels: raw.preprocessor.features,
            window_seconds: raw.preprocessor.window_size,
            hop_seconds: raw.preprocessor.window_stride,
            n_fft: raw.preprocessor.n_fft.unwrap_or(512),
            preemphasis: 0.97,
            encoder_layers: raw.encoder.n_layers,
            encoder_dim: raw.encoder.d_model,
            encoder_heads: raw.encoder.n_heads,
            conv_kernel: raw.encoder.conv_kernel_size,
            subsampling_factor: raw.encoder.subsampling_factor,
            subsampling_channels: raw.encoder.subsampling_conv_channels,
            transformer_layers: raw.transformer_encoder.num_layers,
            transformer_dim: raw.transformer_encoder.hidden_size,
            transformer_inner_dim: raw.transformer_encoder.inner_size,
            transformer_heads: raw.transformer_encoder.num_attention_heads,
            num_speakers: raw.sortformer_modules.num_spks,
            xscaling: raw.encoder.xscaling,
        })
    }
}

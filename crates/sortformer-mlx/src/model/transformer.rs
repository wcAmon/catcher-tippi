//! Sortformer Transformer stack, sigmoid speaker head, and the end-to-end
//! `Diarizer` that composes the full diarization pipeline.
//!
//! Mirrors NeMo's `TransformerEncoder` for the Sortformer v2.1 checkpoint:
//! 18 post-LN blocks (`pre_ln: false`) of standard multi-head self-attention
//! (8 heads over 192 dims, full context, no positional embeddings) followed by
//! a position-wise feed-forward (192 -> 768 -> 192, ReLU). The speaker head is
//! NeMo's `forward_speaker_sigmoids`: `first_hidden_to_hidden` -> ReLU ->
//! `single_hidden_to_spks` -> sigmoid. Tensor names follow
//! `sortformer_inventory.json` exactly.
//!
//! The 192-wide Transformer/head matrices fall below the INT8 quantization
//! threshold (input dim not a multiple of 128, or fewer than 8 rows), so they
//! stay F16 in the artifact and are loaded here as a scalar F32 `Linear`.
//! Only `sortformer_modules.encoder_proj` ([192, 512]) is INT8.

use mlx_rs::Array;
use nemotron_mlx::model::{QuantizedLinear, Tensor3};
use nemotron_mlx::weights::{Artifact, ArtifactError, Storage};

use super::ops::{Norm, relu_in_place};
use super::{Encoder, ModelError, ModelResult};
use crate::audio::MelFrontend;
use crate::config::SortformerConfig;

/// A linear layer that adapts to the artifact's per-tensor storage policy.
///
/// The conversion quantizes a rank-2 `.weight` to INT8 only when its input
/// dim is a multiple of 128 and it has at least 8 rows. For the Transformer
/// and head that means `[*, 768]` and the `[192, 512]` projection are INT8
/// (handled by `nemotron-mlx`'s `QuantizedLinear`), while the `[*, 192]`
/// matrices and the 4-row `single_hidden_to_spks` stay F16 and are run here as
/// a scalar F32 matmul, mirroring the numerics style of `encoder.rs`.
#[derive(Debug)]
struct Linear {
    input_dims: usize,
    output_dims: usize,
    kind: LinearKind,
}

#[derive(Debug)]
enum LinearKind {
    /// INT8 affine weights via the shared `nemotron-mlx` quantized matmul.
    Quantized(QuantizedLinear),
    /// F16 weights run as an F32 MLX matmul on the GPU. The 192-wide
    /// Transformer/head matrices fall below the INT8 quantization threshold, so
    /// they stay F16 in the artifact; the matmul is dominated by these and used
    /// once per frame per layer per streaming chunk, so it runs on-device
    /// (`x · Wᵀ + b`) rather than as a scalar CPU loop. `weight_t` is the
    /// pre-transposed `[input_dims, output_dims]` weight.
    Float { weight_t: Array, bias: Array },
}

impl Linear {
    fn from_artifact(artifact: &Artifact, prefix: &str) -> ModelResult<Self> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let info = artifact
            .tensor_info(&weight_name)
            .ok_or_else(|| ArtifactError::MissingArtifactTensor(weight_name.clone()))?;
        if info.shape.len() != 2 {
            return Err(ModelError::InvalidShape(format!(
                "linear artifact {weight_name} must have rank 2, found {:?}",
                info.shape
            )));
        }
        let output_dims = info.shape[0];
        let input_dims = info.shape[1];
        let kind = match info.storage {
            Storage::Int8Affine { .. } => LinearKind::Quantized(QuantizedLinear::from_artifact(
                artifact,
                &weight_name,
                Some(&bias_name),
            )?),
            Storage::F16 => {
                let weight = artifact.f16_to_f32(&weight_name)?;
                let bias = artifact.f16_to_f32(&bias_name)?;
                if weight.len() != output_dims * input_dims || bias.len() != output_dims {
                    return Err(ModelError::InvalidShape(format!(
                        "linear {prefix} weight/bias inconsistent with [{output_dims},{input_dims}]"
                    )));
                }
                // Row-major `[output_dims, input_dims]` -> transposed
                // `[input_dims, output_dims]` so `x · Wᵀ` is a plain matmul.
                let weight_t = Array::from_slice(
                    &weight,
                    &[output_dims as i32, input_dims as i32],
                )
                .transpose_axes(&[1, 0])?;
                weight_t.eval()?;
                LinearKind::Float {
                    weight_t,
                    bias: Array::from_slice(&bias, &[output_dims as i32]),
                }
            }
            other => {
                return Err(ModelError::InvalidShape(format!(
                    "linear {prefix} has unsupported storage {other:?}"
                )));
            }
        };
        Ok(Self {
            input_dims,
            output_dims,
            kind,
        })
    }

    /// Runs a row-major `[rows, input_dims]` F32 input: `y = x Wᵀ + b`.
    fn forward(&self, input: &[f32], rows: usize) -> ModelResult<Vec<f32>> {
        if rows.checked_mul(self.input_dims) != Some(input.len()) {
            return Err(ModelError::InvalidShape(format!(
                "linear input has {} values, expected {rows}x{}",
                input.len(),
                self.input_dims
            )));
        }
        match &self.kind {
            LinearKind::Quantized(inner) => Ok(inner.forward_f32(input, rows)?),
            LinearKind::Float { weight_t, bias } => {
                let x = Array::from_slice(input, &[rows as i32, self.input_dims as i32]);
                let y = x.matmul(weight_t)?.add(bias)?;
                y.eval()?;
                Ok(y.try_as_slice::<f32>()?.to_vec())
            }
        }
    }
}

/// NeMo `MultiHeadAttention` (`first_sub_layer`): standard scaled dot-product
/// self-attention over the whole utterance, full context and no masking.
#[derive(Debug)]
struct SelfAttention {
    hidden_size: usize,
    heads: usize,
    head_dim: usize,
    query_net: Linear,
    key_net: Linear,
    value_net: Linear,
    out_projection: Linear,
}

impl SelfAttention {
    fn from_artifact(artifact: &Artifact, prefix: &str, heads: usize) -> ModelResult<Self> {
        let query_net = Linear::from_artifact(artifact, &format!("{prefix}.query_net"))?;
        let hidden_size = query_net.output_dims;
        if heads == 0 || hidden_size % heads != 0 {
            return Err(ModelError::InvalidShape(format!(
                "attention hidden size {hidden_size} must divide into {heads} heads"
            )));
        }
        Ok(Self {
            hidden_size,
            heads,
            head_dim: hidden_size / heads,
            query_net,
            key_net: Linear::from_artifact(artifact, &format!("{prefix}.key_net"))?,
            value_net: Linear::from_artifact(artifact, &format!("{prefix}.value_net"))?,
            out_projection: Linear::from_artifact(artifact, &format!("{prefix}.out_projection"))?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> ModelResult<Vec<f32>> {
        let queries = self.query_net.forward(input, frames)?;
        let keys = self.key_net.forward(input, frames)?;
        let values = self.value_net.forward(input, frames)?;
        // NeMo scales queries and keys each by 1/sqrt(sqrt(head_dim)); the
        // product scaling of the scores is therefore 1/sqrt(head_dim).
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let split = [frames as i32, self.heads as i32, self.head_dim as i32];
        // Row-major [T, H*D] -> [T, H, D] -> [H, T, D] per head.
        let q = Array::from_slice(&queries, &split).transpose_axes(&[1, 0, 2])?;
        let k = Array::from_slice(&keys, &split).transpose_axes(&[1, 0, 2])?;
        let v = Array::from_slice(&values, &split).transpose_axes(&[1, 0, 2])?;
        // Scaled dot-product scores [H, T, T], softmaxed over keys.
        let scores = q
            .matmul(k.transpose_axes(&[0, 2, 1])?)?
            .multiply(Array::from_f32(scale))?;
        let probabilities = mlx_rs::ops::softmax_axis(&scores, -1, None)?;
        // [H, T, D] -> [T, H, D] -> [T, H*D].
        let attended = probabilities
            .matmul(v)?
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[frames as i32, self.hidden_size as i32])?;
        attended.eval()?;
        self.out_projection
            .forward(attended.try_as_slice::<f32>()?, frames)
    }
}

/// NeMo `PositionWiseFF` (`second_sub_layer`): `dense_in` -> ReLU -> `dense_out`.
#[derive(Debug)]
struct FeedForward {
    dense_in: Linear,
    dense_out: Linear,
}

impl FeedForward {
    fn from_artifact(artifact: &Artifact, prefix: &str) -> ModelResult<Self> {
        Ok(Self {
            dense_in: Linear::from_artifact(artifact, &format!("{prefix}.dense_in"))?,
            dense_out: Linear::from_artifact(artifact, &format!("{prefix}.dense_out"))?,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> ModelResult<Vec<f32>> {
        let mut hidden = self.dense_in.forward(input, rows)?;
        relu_in_place(&mut hidden);
        self.dense_out.forward(&hidden, rows)
    }
}

/// One post-LN NeMo `TransformerEncoderBlock`.
///
/// `pre_ln: false`, so residuals are added *before* each layer norm:
/// `x = LN1(attn(x) + x); x = LN2(ff(x) + x)`.
#[derive(Debug)]
struct TransformerLayer {
    first_sub_layer: SelfAttention,
    layer_norm_1: Norm,
    second_sub_layer: FeedForward,
    layer_norm_2: Norm,
}

impl TransformerLayer {
    fn from_artifact(artifact: &Artifact, layer: usize, heads: usize) -> ModelResult<Self> {
        let prefix = format!("transformer_encoder.layers.{layer}");
        Ok(Self {
            first_sub_layer: SelfAttention::from_artifact(
                artifact,
                &format!("{prefix}.first_sub_layer"),
                heads,
            )?,
            layer_norm_1: Norm::from_artifact(artifact, &format!("{prefix}.layer_norm_1"))?,
            second_sub_layer: FeedForward::from_artifact(
                artifact,
                &format!("{prefix}.second_sub_layer"),
            )?,
            layer_norm_2: Norm::from_artifact(artifact, &format!("{prefix}.layer_norm_2"))?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> ModelResult<Vec<f32>> {
        let attention = self.first_sub_layer.forward(input, frames)?;
        let mut hidden: Vec<f32> = input
            .iter()
            .zip(&attention)
            .map(|(residual, update)| residual + update)
            .collect();
        hidden = self.layer_norm_1.forward(&hidden, frames)?;

        let feed_forward = self.second_sub_layer.forward(&hidden, frames)?;
        let mut output: Vec<f32> = hidden
            .iter()
            .zip(&feed_forward)
            .map(|(residual, update)| residual + update)
            .collect();
        output = self.layer_norm_2.forward(&output, frames)?;
        Ok(output)
    }
}

/// NeMo `SortformerModules.forward_speaker_sigmoids`: `first_hidden_to_hidden`
/// -> ReLU -> `single_hidden_to_spks` -> sigmoid.
#[derive(Debug)]
struct SigmoidHead {
    first_hidden_to_hidden: Linear,
    single_hidden_to_spks: Linear,
}

impl SigmoidHead {
    fn from_artifact(artifact: &Artifact) -> ModelResult<Self> {
        Ok(Self {
            first_hidden_to_hidden: Linear::from_artifact(
                artifact,
                "sortformer_modules.first_hidden_to_hidden",
            )?,
            single_hidden_to_spks: Linear::from_artifact(
                artifact,
                "sortformer_modules.single_hidden_to_spks",
            )?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> ModelResult<Vec<f32>> {
        let mut hidden = self.first_hidden_to_hidden.forward(input, frames)?;
        relu_in_place(&mut hidden);
        let mut logits = self.single_hidden_to_spks.forward(&hidden, frames)?;
        for value in logits.iter_mut() {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
        Ok(logits)
    }
}

/// End-to-end Sortformer diarizer.
///
/// Composes `MelFrontend` (raw log-mel, `normalize: "NA"`) -> `Encoder` ->
/// `encoder_proj` (INT8) -> 18 Transformer layers -> sigmoid speaker head,
/// returning per-frame (80 ms) speaker probabilities.
#[derive(Debug)]
pub struct Diarizer {
    frontend: MelFrontend,
    encoder: Encoder,
    encoder_proj: Linear,
    layers: Vec<TransformerLayer>,
    head: SigmoidHead,
    transformer_dim: usize,
    frame_ms: u64,
}

impl Diarizer {
    /// Loads the artifact, configuration, and mel frontend from a converted
    /// model directory and binds every Transformer and head tensor.
    pub fn from_artifact_dir(model_dir: impl AsRef<std::path::Path>) -> ModelResult<Self> {
        let model_dir = model_dir.as_ref();
        let artifact = Artifact::load(model_dir)?;
        let config = SortformerConfig::load(model_dir)
            .map_err(|error| ModelError::InvalidShape(error.to_string()))?;
        Self::from_parts(&artifact, &config)
    }

    fn from_parts(artifact: &Artifact, config: &SortformerConfig) -> ModelResult<Self> {
        if config.transformer_dim % config.transformer_heads != 0 {
            return Err(ModelError::InvalidShape(format!(
                "transformer dim {} must divide into {} heads",
                config.transformer_dim, config.transformer_heads
            )));
        }
        if config.num_speakers != 4 {
            return Err(ModelError::InvalidShape(format!(
                "diarizer expects 4 speakers, checkpoint has {}",
                config.num_speakers
            )));
        }
        let encoder = Encoder::from_artifact(artifact, config)?;
        let encoder_proj = Linear::from_artifact(artifact, "sortformer_modules.encoder_proj")?;
        let layers = (0..config.transformer_layers)
            .map(|layer| TransformerLayer::from_artifact(artifact, layer, config.transformer_heads))
            .collect::<ModelResult<Vec<_>>>()?;
        let head = SigmoidHead::from_artifact(artifact)?;
        let frame_ms =
            (config.hop_seconds * config.subsampling_factor as f64 * 1_000.0).round() as u64;
        Ok(Self {
            frontend: MelFrontend::new(config),
            encoder,
            encoder_proj,
            layers,
            head,
            transformer_dim: config.transformer_dim,
            frame_ms,
        })
    }

    /// Output frame duration in milliseconds: `hop_seconds * subsampling_factor * 1000`,
    /// rounded. 80 ms for the v2.1 checkpoint (10 ms hop, 8x subsampling).
    pub fn frame_ms(&self) -> u64 {
        self.frame_ms
    }

    /// Diarizes raw 16 kHz mono audio into per-frame speaker probabilities.
    pub fn diarize(&self, audio: &[f32]) -> ModelResult<Vec<[f32; 4]>> {
        let hidden = self.forward_hidden(audio)?;
        Ok(speaker_rows(&self.head.forward(&hidden.values, hidden.shape[1])?))
    }

    /// Runs the pipeline up to (and including) the final Transformer layer,
    /// returning the `[1, frames, transformer_dim]` hidden state fed to the head.
    fn forward_hidden(&self, audio: &[f32]) -> ModelResult<Tensor3> {
        // Checkpoint preprocessor uses `normalize: "NA"`: raw log-mel frames.
        let mel_frames = self.frontend.extract(audio);
        let encoded = self.encoder.forward(&mel_frames)?;
        self.transformer_tail(&encoded)
    }

    /// Shared tail: `encoder_proj` -> 18 Transformer layers, mapping the encoder
    /// output `[1, frames, encoder_dim]` to the `[1, frames, transformer_dim]`
    /// hidden state. Reused by the offline `diarize` and the streaming diarizer
    /// so the model tail is defined exactly once.
    fn transformer_tail(&self, encoded: &Tensor3) -> ModelResult<Tensor3> {
        let frames = encoded.shape[1];
        let mut hidden = self.encoder_proj.forward(&encoded.values, frames)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, frames)?;
        }
        Ok(Tensor3 {
            shape: [1, frames, self.transformer_dim],
            values: hidden,
        })
    }

    /// Log-mel frontend, shared with the streaming diarizer for incremental
    /// per-chunk mel extraction (`extract_frames`).
    pub(crate) fn frontend(&self) -> &MelFrontend {
        &self.frontend
    }

    /// NEST Fast-Conformer encoder, shared with the streaming diarizer for
    /// `pre_encode` (subsampling) of each chunk's mel window.
    pub(crate) fn encoder(&self) -> &Encoder {
        &self.encoder
    }

    /// Streaming tail: runs the Conformer blocks (`forward_embedded`, which
    /// scales the whole `[spkcache | fifo | chunk]` sequence and applies
    /// relative-position attention across it), then the shared Transformer tail
    /// and sigmoid head, returning one 4-wide speaker-probability row per input
    /// frame. `embedded` is the assembled UNSCALED pre-encode sequence.
    pub(crate) fn forward_embedded_preds(
        &self,
        embedded: &Tensor3,
    ) -> ModelResult<Vec<[f32; 4]>> {
        let encoded = self.encoder.forward_embedded(embedded)?;
        let hidden = self.transformer_tail(&encoded)?;
        Ok(speaker_rows(&self.head.forward(&hidden.values, hidden.shape[1])?))
    }
}

/// Groups a flat `[frames * 4]` sigmoid-head output into per-frame speaker rows.
///
/// `num_speakers == 4` is guaranteed by `Diarizer::from_parts`, which rejects
/// any other checkpoint at load time, so this `[f32; 4]` contract always holds.
fn speaker_rows(probabilities: &[f32]) -> Vec<[f32; 4]> {
    probabilities
        .chunks_exact(4)
        .map(|frame| [frame[0], frame[1], frame[2], frame[3]])
        .collect()
}

//! NEST Fast-Conformer encoder in offline, full-context inference mode.
//!
//! Mirrors NeMo's `ConformerEncoder` for the Sortformer v2.1 checkpoint:
//! `dw_striding` subsampling (8x), then 17 macaron Conformer blocks with
//! Transformer-XL relative-position attention over the whole utterance and a
//! symmetric (non-causal) convolution module normalized with inference-mode
//! BatchNorm. Tensor names follow `sortformer_inventory.json` exactly.

use mlx_rs::Array;
use nemotron_mlx::model::{
    QuantizedLinear, Tensor3, Tensor4, channel_frequency_flatten, relative_position_encoding,
    relative_shift,
};
use nemotron_mlx::weights::{Artifact, ArtifactError};

use super::ops::{Norm, add_in_place, relu_in_place, silu_in_place, softmax_in_place};
use super::{ModelError, ModelResult};
use crate::config::SortformerConfig;

const BATCH_NORM_EPSILON: f32 = 1.0e-5;

/// Full-context NEST Fast-Conformer encoder.
#[derive(Debug)]
pub struct Encoder {
    mel_bins: usize,
    hidden_size: usize,
    /// `sqrt(d_model)` when `config.xscaling` (NeMo's `RelPositionalEncoding`
    /// input scaling), otherwise `1.0`; applied before the first block.
    input_scale: f32,
    subsampling: Subsampling,
    layers: Vec<ConformerBlock>,
}

/// Intermediate tensors used for parity validation and diagnostics.
///
/// Phase-2 surface: not wired into `Diarizer::diarize` today, but retained
/// for NeMo parity debugging and as the hook a future streaming/incremental
/// encoder would use to inspect per-block state.
#[derive(Debug, Clone)]
pub struct EncoderTrace {
    /// `encoder.pre_encode` output before the `sqrt(d_model)` input scaling.
    pub subsampling: Tensor3,
    /// Every Conformer block output; the last entry equals the encoder output.
    pub layers: Vec<Tensor3>,
}

impl Encoder {
    /// Binds every `encoder.*` tensor from a converted artifact.
    pub fn from_artifact(artifact: &Artifact, config: &SortformerConfig) -> ModelResult<Self> {
        if config.encoder_heads == 0 || config.encoder_dim % config.encoder_heads != 0 {
            return Err(ModelError::InvalidShape(format!(
                "encoder dim {} must divide into {} heads",
                config.encoder_dim, config.encoder_heads
            )));
        }
        let layers = (0..config.encoder_layers)
            .map(|layer| ConformerBlock::from_artifact(artifact, layer, config))
            .collect::<ModelResult<Vec<_>>>()?;
        Ok(Self {
            mel_bins: config.n_mels,
            hidden_size: config.encoder_dim,
            input_scale: if config.xscaling {
                (config.encoder_dim as f32).sqrt()
            } else {
                1.0
            },
            subsampling: Subsampling::from_artifact(artifact, config)?,
            layers,
        })
    }

    /// Encodes unnormalized log-mel frames into `[1, frames/8, encoder_dim]`.
    pub fn forward(&self, mel_frames: &[Vec<f32>]) -> ModelResult<Tensor3> {
        let (hidden, frames) = self.run(mel_frames, None)?;
        Ok(Tensor3 {
            shape: [1, frames, self.hidden_size],
            values: hidden,
        })
    }

    /// Encodes and records every intermediate block output.
    ///
    /// Phase-2 surface: exists for parity diagnostics against NeMo block
    /// outputs, not for production diarization inference.
    pub fn forward_trace(&self, mel_frames: &[Vec<f32>]) -> ModelResult<EncoderTrace> {
        let mut trace = EncoderTrace {
            subsampling: Tensor3 {
                shape: [1, 0, self.hidden_size],
                values: Vec::new(),
            },
            layers: Vec::with_capacity(self.layers.len()),
        };
        self.run(mel_frames, Some(&mut trace))?;
        Ok(trace)
    }

    fn run(
        &self,
        mel_frames: &[Vec<f32>],
        mut trace: Option<&mut EncoderTrace>,
    ) -> ModelResult<(Vec<f32>, usize)> {
        if mel_frames.is_empty() || mel_frames.iter().any(|frame| frame.len() != self.mel_bins) {
            return Err(ModelError::InvalidShape(format!(
                "encoder input must be non-empty [time][{}] mel frames",
                self.mel_bins
            )));
        }
        let subsampled = self.subsampling.forward(mel_frames)?;
        let frames = subsampled.shape[1];
        if let Some(trace) = trace.as_deref_mut() {
            trace.subsampling = subsampled.clone();
        }
        let mut hidden: Vec<f32> = subsampled
            .values
            .iter()
            .map(|value| value * self.input_scale)
            .collect();
        let positions = relative_position_encoding(self.hidden_size, frames)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, frames, &positions)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.layers.push(Tensor3 {
                    shape: [1, frames, self.hidden_size],
                    values: hidden.clone(),
                });
            }
        }
        Ok((hidden, frames))
    }
}

/// FP32 MLX Conv2D with symmetric (SAME-style) padding on time and frequency.
///
/// NeMo's `dw_striding` subsampling uses non-causal `padding = (kernel-1)/2`
/// on both axes, unlike the causal `Fp16Conv2d` in nemotron-mlx. Reuses the
/// same PyTorch OIHW -> MLX OHWI weight transposition.
#[derive(Debug)]
struct SymmetricConv2d {
    input_channels: usize,
    output_channels: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    weight: Array,
    bias: Array,
}

impl SymmetricConv2d {
    fn from_artifact(
        artifact: &Artifact,
        weight_name: &str,
        bias_name: &str,
        stride: usize,
        groups: usize,
    ) -> ModelResult<Self> {
        let shape = artifact
            .tensor_info(weight_name)
            .ok_or_else(|| ArtifactError::MissingArtifactTensor(weight_name.to_string()))?
            .shape
            .clone();
        if shape.len() != 4 || shape[2] != shape[3] || groups == 0 || stride == 0 {
            return Err(ModelError::InvalidShape(format!(
                "Conv2D artifact {weight_name} must have square OIHW shape"
            )));
        }
        let output_channels = shape[0];
        let channels_per_group = shape[1];
        let kernel_size = shape[2];
        let pytorch_weight = artifact.f16_to_f32(weight_name)?;
        let bias = artifact.f16_to_f32(bias_name)?;
        if bias.len() != output_channels {
            return Err(ModelError::InvalidShape(format!(
                "Conv2D bias {bias_name} must have shape [{output_channels}]"
            )));
        }
        // PyTorch OIHW -> MLX OHWI.
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
        Ok(Self {
            input_channels: channels_per_group * groups,
            output_channels,
            stride,
            padding: (kernel_size - 1) / 2,
            groups,
            weight: Array::from_slice(
                &mlx_weight,
                &[
                    output_channels as i32,
                    kernel_size as i32,
                    kernel_size as i32,
                    channels_per_group as i32,
                ],
            ),
            bias: Array::from_slice(&bias, &[output_channels as i32]),
        })
    }

    fn forward(&self, input: &Tensor4) -> ModelResult<Tensor4> {
        let [batch, time, frequency, channels] = input.shape;
        if batch != 1
            || channels != self.input_channels
            || input.values.len() != time * frequency * channels
        {
            return Err(ModelError::InvalidShape(format!(
                "Conv2D input must be [1,time,freq,{}]",
                self.input_channels
            )));
        }
        let input = Array::from_slice(
            &input.values,
            &[1, time as i32, frequency as i32, channels as i32],
        );
        let output = mlx_rs::ops::conv2d(
            &input,
            &self.weight,
            (self.stride as i32, self.stride as i32),
            (self.padding as i32, self.padding as i32),
            (1, 1),
            self.groups as i32,
        )?
        .add(&self.bias)?;
        output.eval()?;
        let shape = output.shape();
        debug_assert_eq!(shape[3] as usize, self.output_channels);
        Ok(Tensor4 {
            shape: [
                shape[0] as usize,
                shape[1] as usize,
                shape[2] as usize,
                shape[3] as usize,
            ],
            values: output.try_as_slice::<f32>()?.to_vec(),
        })
    }
}

/// NeMo `dw_striding` subsampling: a Conv2d stem plus depthwise/pointwise
/// pairs (stride 2 each, ReLU after every stage) and a final projection.
#[derive(Debug)]
struct Subsampling {
    stem: SymmetricConv2d,
    stages: Vec<(SymmetricConv2d, QuantizedLinear, usize)>,
    output: QuantizedLinear,
    output_dims: usize,
}

impl Subsampling {
    fn from_artifact(artifact: &Artifact, config: &SortformerConfig) -> ModelResult<Self> {
        let channels = config.subsampling_channels;
        let stem = SymmetricConv2d::from_artifact(
            artifact,
            "encoder.pre_encode.conv.0.weight",
            "encoder.pre_encode.conv.0.bias",
            2,
            1,
        )?;
        // Sequential indices in NeMo's module list: stem 0, ReLU 1, then
        // (depthwise, pointwise, ReLU) triples at 2/3/4 and 5/6/7.
        let mut stages = Vec::with_capacity(2);
        for depthwise_index in [2usize, 5] {
            let depthwise = SymmetricConv2d::from_artifact(
                artifact,
                &format!("encoder.pre_encode.conv.{depthwise_index}.weight"),
                &format!("encoder.pre_encode.conv.{depthwise_index}.bias"),
                2,
                channels,
            )?;
            let pointwise_index = depthwise_index + 1;
            let pointwise_name = format!("encoder.pre_encode.conv.{pointwise_index}.weight");
            let pointwise_dims = artifact
                .tensor_info(&pointwise_name)
                .ok_or_else(|| ArtifactError::MissingArtifactTensor(pointwise_name.clone()))?
                .shape[0];
            let pointwise = QuantizedLinear::from_artifact(
                artifact,
                &pointwise_name,
                Some(&format!("encoder.pre_encode.conv.{pointwise_index}.bias")),
            )?;
            stages.push((depthwise, pointwise, pointwise_dims));
        }
        let output_name = "encoder.pre_encode.out.weight";
        let output_dims = artifact
            .tensor_info(output_name)
            .ok_or_else(|| ArtifactError::MissingArtifactTensor(output_name.to_string()))?
            .shape[0];
        Ok(Self {
            stem,
            stages,
            output: QuantizedLinear::from_artifact(
                artifact,
                output_name,
                Some("encoder.pre_encode.out.bias"),
            )?,
            output_dims,
        })
    }

    fn forward(&self, mel_frames: &[Vec<f32>]) -> ModelResult<Tensor3> {
        let time = mel_frames.len();
        let mel_bins = mel_frames[0].len();
        let values: Vec<f32> = mel_frames.iter().flatten().copied().collect();
        let mut hidden = self.stem.forward(&Tensor4 {
            shape: [1, time, mel_bins, 1],
            values,
        })?;
        relu_in_place(&mut hidden.values);
        for (depthwise, pointwise, pointwise_dims) in &self.stages {
            hidden = depthwise.forward(&hidden)?;
            let rows = hidden.shape[1] * hidden.shape[2];
            hidden = Tensor4 {
                shape: [1, hidden.shape[1], hidden.shape[2], *pointwise_dims],
                values: pointwise.forward_f32(&hidden.values, rows)?,
            };
            relu_in_place(&mut hidden.values);
        }
        let rows = hidden.shape[1];
        let flattened = channel_frequency_flatten(&hidden)?;
        Ok(Tensor3 {
            shape: [1, rows, self.output_dims],
            values: self.output.forward_f32(&flattened, rows)?,
        })
    }
}

/// Inference-mode BatchNorm1d folded into a per-channel affine transform.
#[derive(Debug)]
struct BatchNorm {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl BatchNorm {
    fn from_artifact(artifact: &Artifact, prefix: &str) -> ModelResult<Self> {
        let mean = artifact.f16_to_f32(&format!("{prefix}.running_mean"))?;
        let variance = artifact.f16_to_f32(&format!("{prefix}.running_var"))?;
        let weight = artifact.f16_to_f32(&format!("{prefix}.weight"))?;
        let bias = artifact.f16_to_f32(&format!("{prefix}.bias"))?;
        if mean.len() != variance.len() || mean.len() != weight.len() || mean.len() != bias.len() {
            return Err(ModelError::InvalidShape(format!(
                "batch norm {prefix} parameter lengths must match"
            )));
        }
        // (x - mean) / sqrt(var + eps) * w + b == x * scale + shift.
        let scale: Vec<f32> = weight
            .iter()
            .zip(&variance)
            .map(|(weight, variance)| weight / (variance + BATCH_NORM_EPSILON).sqrt())
            .collect();
        let shift = bias
            .iter()
            .zip(&mean)
            .zip(&scale)
            .map(|((bias, mean), scale)| bias - mean * scale)
            .collect();
        Ok(Self { scale, shift })
    }

    fn forward_in_place(&self, values: &mut [f32]) {
        let channels = self.scale.len();
        for frame in values.chunks_exact_mut(channels) {
            for (value, (scale, shift)) in frame.iter_mut().zip(self.scale.iter().zip(&self.shift))
            {
                *value = *value * scale + shift;
            }
        }
    }
}

/// Conformer feed-forward: linear -> Swish -> linear with half residual.
#[derive(Debug)]
struct FeedForward {
    linear1: QuantizedLinear,
    linear2: QuantizedLinear,
}

impl FeedForward {
    fn from_artifact(artifact: &Artifact, prefix: &str) -> ModelResult<Self> {
        Ok(Self {
            linear1: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.linear1.weight"),
                Some(&format!("{prefix}.linear1.bias")),
            )?,
            linear2: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.linear2.weight"),
                Some(&format!("{prefix}.linear2.bias")),
            )?,
        })
    }

    fn forward(&self, input: &[f32], rows: usize) -> ModelResult<Vec<f32>> {
        let mut hidden = self.linear1.forward_f32(input, rows)?;
        silu_in_place(&mut hidden);
        Ok(self.linear2.forward_f32(&hidden, rows)?)
    }
}

/// NeMo `RelPositionMultiHeadAttention` over the full sequence, no masking.
#[derive(Debug)]
struct SelfAttention {
    hidden_size: usize,
    heads: usize,
    head_dim: usize,
    linear_q: QuantizedLinear,
    linear_k: QuantizedLinear,
    linear_v: QuantizedLinear,
    linear_out: QuantizedLinear,
    linear_pos: QuantizedLinear,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
}

impl SelfAttention {
    fn from_artifact(artifact: &Artifact, prefix: &str, heads: usize) -> ModelResult<Self> {
        let query_name = format!("{prefix}.linear_q.weight");
        let hidden_size = artifact
            .tensor_info(&query_name)
            .ok_or_else(|| ArtifactError::MissingArtifactTensor(query_name.clone()))?
            .shape[0];
        if heads == 0 || hidden_size % heads != 0 {
            return Err(ModelError::InvalidShape(format!(
                "attention hidden size {hidden_size} must divide into {heads} heads"
            )));
        }
        let bias_u = artifact.f16_to_f32(&format!("{prefix}.pos_bias_u"))?;
        let bias_v = artifact.f16_to_f32(&format!("{prefix}.pos_bias_v"))?;
        if bias_u.len() != hidden_size || bias_v.len() != hidden_size {
            return Err(ModelError::InvalidShape(format!(
                "attention position biases for {prefix} must have {hidden_size} values"
            )));
        }
        let projection = |name: &str, bias: bool| -> ModelResult<QuantizedLinear> {
            Ok(QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.{name}.weight"),
                bias.then_some(format!("{prefix}.{name}.bias")).as_deref(),
            )?)
        };
        Ok(Self {
            hidden_size,
            heads,
            head_dim: hidden_size / heads,
            linear_q: projection("linear_q", true)?,
            linear_k: projection("linear_k", true)?,
            linear_v: projection("linear_v", true)?,
            linear_out: projection("linear_out", true)?,
            linear_pos: projection("linear_pos", false)?,
            bias_u,
            bias_v,
        })
    }

    /// Scores `(q+u)·kᵀ + rel_shift((q+v)·posᵀ)` scaled by `1/sqrt(d_head)`
    /// and softmaxed over every frame of the utterance.
    fn forward(&self, input: &[f32], frames: usize, positions: &Tensor3) -> ModelResult<Vec<f32>> {
        let hidden_size = self.hidden_size;
        let queries = self.linear_q.forward_f32(input, frames)?;
        let keys = self.linear_k.forward_f32(input, frames)?;
        let values = self.linear_v.forward_f32(input, frames)?;
        let position_frames = positions.shape[1];
        let relative_keys = self
            .linear_pos
            .forward_f32(&positions.values, position_frames)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attended = vec![0.0; frames * hidden_size];
        for head in 0..self.heads {
            let offset = head * self.head_dim;
            let mut raw_relative = vec![0.0; frames * position_frames];
            for query in 0..frames {
                for position in 0..position_frames {
                    let mut score = 0.0;
                    for dimension in 0..self.head_dim {
                        score += (queries[query * hidden_size + offset + dimension]
                            + self.bias_v[offset + dimension])
                            * relative_keys[position * hidden_size + offset + dimension];
                    }
                    raw_relative[query * position_frames + position] = score;
                }
            }
            let shifted = relative_shift(&raw_relative, frames, position_frames)?;
            for query in 0..frames {
                let mut scores = vec![0.0; frames];
                for (key, score_slot) in scores.iter_mut().enumerate() {
                    let mut content = 0.0;
                    for dimension in 0..self.head_dim {
                        content += (queries[query * hidden_size + offset + dimension]
                            + self.bias_u[offset + dimension])
                            * keys[key * hidden_size + offset + dimension];
                    }
                    *score_slot = (content + shifted[query * position_frames + key]) * scale;
                }
                softmax_in_place(&mut scores);
                for dimension in 0..self.head_dim {
                    let mut value = 0.0;
                    for (key, probability) in scores.iter().enumerate() {
                        value += probability * values[key * hidden_size + offset + dimension];
                    }
                    attended[query * hidden_size + offset + dimension] = value;
                }
            }
        }
        Ok(self.linear_out.forward_f32(&attended, frames)?)
    }
}

/// Conformer convolution module with full-context symmetric padding.
#[derive(Debug)]
struct ConvModule {
    hidden_size: usize,
    kernel_size: usize,
    pointwise1: QuantizedLinear,
    depthwise_weight: Vec<f32>,
    depthwise_bias: Vec<f32>,
    batch_norm: BatchNorm,
    pointwise2: QuantizedLinear,
}

impl ConvModule {
    fn from_artifact(artifact: &Artifact, prefix: &str) -> ModelResult<Self> {
        let depthwise_name = format!("{prefix}.depthwise_conv.weight");
        let depthwise_shape = artifact
            .tensor_info(&depthwise_name)
            .ok_or_else(|| ArtifactError::MissingArtifactTensor(depthwise_name.clone()))?
            .shape
            .clone();
        if depthwise_shape.len() != 3 || depthwise_shape[1] != 1 {
            return Err(ModelError::InvalidShape(format!(
                "depthwise artifact {depthwise_name} must have [channels,1,kernel] shape"
            )));
        }
        let (hidden_size, kernel_size) = (depthwise_shape[0], depthwise_shape[2]);
        if kernel_size % 2 == 0 {
            return Err(ModelError::InvalidShape(format!(
                "symmetric depthwise convolution requires an odd kernel, found {kernel_size}"
            )));
        }
        Ok(Self {
            hidden_size,
            kernel_size,
            pointwise1: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.pointwise_conv1.weight"),
                Some(&format!("{prefix}.pointwise_conv1.bias")),
            )?,
            depthwise_weight: artifact.f16_to_f32(&depthwise_name)?,
            depthwise_bias: artifact.f16_to_f32(&format!("{prefix}.depthwise_conv.bias"))?,
            batch_norm: BatchNorm::from_artifact(artifact, &format!("{prefix}.batch_norm"))?,
            pointwise2: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.pointwise_conv2.weight"),
                Some(&format!("{prefix}.pointwise_conv2.bias")),
            )?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> ModelResult<Vec<f32>> {
        let channels = self.hidden_size;
        let projected = self.pointwise1.forward_f32(input, frames)?;
        // GLU over the doubled channel dimension.
        let mut gated = vec![0.0; frames * channels];
        for frame in 0..frames {
            for channel in 0..channels {
                let first = projected[frame * 2 * channels + channel];
                let gate = projected[frame * 2 * channels + channels + channel];
                gated[frame * channels + channel] = first / (1.0 + (-gate).exp());
            }
        }
        // Depthwise convolution, symmetric zero padding of (kernel-1)/2.
        let half_kernel = (self.kernel_size - 1) / 2;
        let mut convolved = vec![0.0; frames * channels];
        for frame in 0..frames {
            for channel in 0..channels {
                let mut value = self.depthwise_bias[channel];
                for tap in 0..self.kernel_size {
                    let Some(source) = (frame + tap).checked_sub(half_kernel) else {
                        continue;
                    };
                    if source >= frames {
                        continue;
                    }
                    value += self.depthwise_weight[channel * self.kernel_size + tap]
                        * gated[source * channels + channel];
                }
                convolved[frame * channels + channel] = value;
            }
        }
        self.batch_norm.forward_in_place(&mut convolved);
        silu_in_place(&mut convolved);
        Ok(self.pointwise2.forward_f32(&convolved, frames)?)
    }
}

/// One macaron Fast-Conformer block in full-context inference mode.
#[derive(Debug)]
struct ConformerBlock {
    norm_feed_forward1: Norm,
    feed_forward1: FeedForward,
    norm_self_att: Norm,
    self_attn: SelfAttention,
    norm_conv: Norm,
    conv: ConvModule,
    norm_feed_forward2: Norm,
    feed_forward2: FeedForward,
    norm_out: Norm,
}

impl ConformerBlock {
    fn from_artifact(
        artifact: &Artifact,
        layer: usize,
        config: &SortformerConfig,
    ) -> ModelResult<Self> {
        let prefix = format!("encoder.layers.{layer}");
        Ok(Self {
            norm_feed_forward1: Norm::from_artifact(
                artifact,
                &format!("{prefix}.norm_feed_forward1"),
            )?,
            feed_forward1: FeedForward::from_artifact(
                artifact,
                &format!("{prefix}.feed_forward1"),
            )?,
            norm_self_att: Norm::from_artifact(artifact, &format!("{prefix}.norm_self_att"))?,
            self_attn: SelfAttention::from_artifact(
                artifact,
                &format!("{prefix}.self_attn"),
                config.encoder_heads,
            )?,
            norm_conv: Norm::from_artifact(artifact, &format!("{prefix}.norm_conv"))?,
            conv: ConvModule::from_artifact(artifact, &format!("{prefix}.conv"))?,
            norm_feed_forward2: Norm::from_artifact(
                artifact,
                &format!("{prefix}.norm_feed_forward2"),
            )?,
            feed_forward2: FeedForward::from_artifact(
                artifact,
                &format!("{prefix}.feed_forward2"),
            )?,
            norm_out: Norm::from_artifact(artifact, &format!("{prefix}.norm_out"))?,
        })
    }

    fn forward(&self, input: &[f32], frames: usize, positions: &Tensor3) -> ModelResult<Vec<f32>> {
        let normalized = self.norm_feed_forward1.forward(input, frames)?;
        let feed_forward1 = self.feed_forward1.forward(&normalized, frames)?;
        let mut hidden: Vec<f32> = input
            .iter()
            .zip(&feed_forward1)
            .map(|(input, update)| input + 0.5 * update)
            .collect();

        let normalized = self.norm_self_att.forward(&hidden, frames)?;
        let attention = self.self_attn.forward(&normalized, frames, positions)?;
        add_in_place(&mut hidden, &attention, 1.0);

        let normalized = self.norm_conv.forward(&hidden, frames)?;
        let convolution = self.conv.forward(&normalized, frames)?;
        add_in_place(&mut hidden, &convolution, 1.0);

        let normalized = self.norm_feed_forward2.forward(&hidden, frames)?;
        let feed_forward2 = self.feed_forward2.forward(&normalized, frames)?;
        add_in_place(&mut hidden, &feed_forward2, 0.5);

        self.norm_out.forward(&hidden, frames)
    }
}

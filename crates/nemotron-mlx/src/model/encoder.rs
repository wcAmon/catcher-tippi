use super::{
    AttentionKvCache, CausalConv1dCache, CausalConv2dCache, DepthwiseConv1d, Fp16Conv2d, LayerNorm,
    ModelError, ModelResult, QuantizedLinear, Tensor3, Tensor4,
};

/// Per-layer time caches for factor-N causal Conv2D subsampling.
#[derive(Debug, Clone)]
pub struct SubsamplingCache {
    layers: Vec<CausalConv2dCache>,
}

impl SubsamplingCache {
    pub fn new(
        input_frequency_bins: usize,
        channels: usize,
        kernel_size: usize,
        stride: usize,
        stages: usize,
    ) -> Self {
        let left_frames = kernel_size - stride;
        let initial_extra_frames = kernel_size - 1 - left_frames;
        let mut frequency = input_frequency_bins;
        let mut layers = Vec::with_capacity(stages);
        for stage in 0..stages {
            layers.push(CausalConv2dCache::new(
                frequency,
                if stage == 0 { 1 } else { channels },
                left_frames,
                initial_extra_frames,
            ));
            frequency = (frequency + (kernel_size - 1) + (stride - 1) - kernel_size) / stride + 1;
        }
        Self { layers }
    }
}

/// Three-stage causal Conv2D subsampling followed by the encoder input projection.
#[derive(Debug)]
pub struct Conv2dSubsampling {
    stem: Fp16Conv2d,
    stages: Vec<(Fp16Conv2d, QuantizedLinear)>,
    output: QuantizedLinear,
}

/// Multi-head content/relative-position attention with a sliding K/V cache.
#[derive(Debug)]
pub struct RelativePositionAttention {
    hidden_size: usize,
    heads: usize,
    head_dim: usize,
    q_projection: QuantizedLinear,
    k_projection: QuantizedLinear,
    v_projection: QuantizedLinear,
    output_projection: QuantizedLinear,
    relative_projection: QuantizedLinear,
    bias_u: Vec<f32>,
    bias_v: Vec<f32>,
}

impl RelativePositionAttention {
    pub fn from_artifact(
        artifact: &crate::weights::Artifact,
        layer: usize,
        heads: usize,
    ) -> ModelResult<Self> {
        let prefix = format!("encoder.layers.{layer}.self_attn");
        let q_projection =
            QuantizedLinear::from_artifact(artifact, &format!("{prefix}.q_proj.weight"), None)?;
        let hidden_size = q_projection.output_dims();
        if heads == 0 || hidden_size % heads != 0 {
            return Err(ModelError::InvalidShape(
                "artifact attention hidden size must divide into heads".to_string(),
            ));
        }
        let bias_u = artifact.f16_to_f32(&format!("{prefix}.bias_u"))?;
        let bias_v = artifact.f16_to_f32(&format!("{prefix}.bias_v"))?;
        if bias_u.len() != hidden_size || bias_v.len() != hidden_size {
            return Err(ModelError::InvalidShape(
                "artifact attention bias size is incorrect".to_string(),
            ));
        }
        Ok(Self {
            hidden_size,
            heads,
            head_dim: hidden_size / heads,
            q_projection,
            k_projection: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.k_proj.weight"),
                None,
            )?,
            v_projection: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.v_proj.weight"),
                None,
            )?,
            output_projection: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.o_proj.weight"),
                None,
            )?,
            relative_projection: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.relative_k_proj.weight"),
                None,
            )?,
            bias_u,
            bias_v,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_f32(
        q_weight: &[f32],
        k_weight: &[f32],
        v_weight: &[f32],
        output_weight: &[f32],
        relative_weight: &[f32],
        bias_u: &[f32],
        bias_v: &[f32],
        hidden_size: usize,
        heads: usize,
        group_size: usize,
    ) -> ModelResult<Self> {
        if heads == 0
            || hidden_size % heads != 0
            || bias_u.len() != hidden_size
            || bias_v.len() != hidden_size
        {
            return Err(ModelError::InvalidShape(
                "relative attention dimensions are inconsistent".to_string(),
            ));
        }
        let zero_bias = vec![0.0; hidden_size];
        Ok(Self {
            hidden_size,
            heads,
            head_dim: hidden_size / heads,
            q_projection: QuantizedLinear::from_f32(
                q_weight,
                hidden_size,
                hidden_size,
                &zero_bias,
                group_size,
            )?,
            k_projection: QuantizedLinear::from_f32(
                k_weight,
                hidden_size,
                hidden_size,
                &zero_bias,
                group_size,
            )?,
            v_projection: QuantizedLinear::from_f32(
                v_weight,
                hidden_size,
                hidden_size,
                &zero_bias,
                group_size,
            )?,
            output_projection: QuantizedLinear::from_f32(
                output_weight,
                hidden_size,
                hidden_size,
                &zero_bias,
                group_size,
            )?,
            relative_projection: QuantizedLinear::from_f32(
                relative_weight,
                hidden_size,
                hidden_size,
                &zero_bias,
                group_size,
            )?,
            bias_u: bias_u.to_vec(),
            bias_v: bias_v.to_vec(),
        })
    }

    pub fn forward_streaming(
        &self,
        input: &Tensor3,
        cache: &mut crate::model::AttentionKvCache,
    ) -> ModelResult<Tensor3> {
        let [batch, query_frames, channels] = input.shape;
        if batch != 1
            || channels != self.hidden_size
            || input.values.len() != query_frames * channels
        {
            return Err(ModelError::InvalidShape(format!(
                "attention input must have shape [1,time,{}]",
                self.hidden_size
            )));
        }
        let queries = self.q_projection.forward_f32(&input.values, query_frames)?;
        let current_keys = token_major_to_head_major(
            &self.k_projection.forward_f32(&input.values, query_frames)?,
            query_frames,
            self.heads,
            self.head_dim,
        );
        let current_values = token_major_to_head_major(
            &self.v_projection.forward_f32(&input.values, query_frames)?,
            query_frames,
            self.heads,
            self.head_dim,
        );
        let visible = cache.update(&current_keys, &current_values, query_frames)?;
        let positions = relative_position_encoding(self.hidden_size, visible.frames)?;
        let position_frames = positions.shape[1];
        let relative_keys = self
            .relative_projection
            .forward_f32(&positions.values, position_frames)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attended = vec![0.0; query_frames * self.hidden_size];

        for head in 0..self.heads {
            let mut raw_relative = vec![0.0; query_frames * position_frames];
            for query in 0..query_frames {
                for position in 0..position_frames {
                    let mut score = 0.0;
                    for dimension in 0..self.head_dim {
                        let query_index =
                            query * self.hidden_size + head * self.head_dim + dimension;
                        let relative_index =
                            position * self.hidden_size + head * self.head_dim + dimension;
                        score += (queries[query_index]
                            + self.bias_v[head * self.head_dim + dimension])
                            * relative_keys[relative_index];
                    }
                    raw_relative[query * position_frames + position] = score;
                }
            }
            let shifted = relative_shift(&raw_relative, query_frames, position_frames)?;

            for query in 0..query_frames {
                let mut scores = vec![0.0; visible.frames];
                for (key, score_slot) in scores.iter_mut().enumerate() {
                    let mut content = 0.0;
                    for dimension in 0..self.head_dim {
                        let query_index =
                            query * self.hidden_size + head * self.head_dim + dimension;
                        let key_index = (head * visible.frames + key) * self.head_dim + dimension;
                        content += (queries[query_index]
                            + self.bias_u[head * self.head_dim + dimension])
                            * visible.keys[key_index];
                    }
                    *score_slot = (content + shifted[query * position_frames + key]) * scale;
                }
                softmax_in_place(&mut scores);
                for dimension in 0..self.head_dim {
                    let mut value = 0.0;
                    for (key, probability) in scores.iter().enumerate() {
                        let value_index = (head * visible.frames + key) * self.head_dim + dimension;
                        value += probability * visible.values[value_index];
                    }
                    attended[query * self.hidden_size + head * self.head_dim + dimension] = value;
                }
            }
        }
        Ok(Tensor3 {
            shape: input.shape,
            values: self
                .output_projection
                .forward_f32(&attended, query_frames)?,
        })
    }
}

fn token_major_to_head_major(
    values: &[f32],
    frames: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let hidden_size = heads * head_dim;
    let mut output = vec![0.0; values.len()];
    for head in 0..heads {
        for frame in 0..frames {
            let source = frame * hidden_size + head * head_dim;
            let destination = (head * frames + frame) * head_dim;
            output[destination..destination + head_dim]
                .copy_from_slice(&values[source..source + head_dim]);
        }
    }
    output
}

fn softmax_in_place(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        total += *value;
    }
    for value in values {
        *value /= total;
    }
}

#[derive(Debug)]
struct QuantizedFeedForward {
    hidden_size: usize,
    linear1: QuantizedLinear,
    linear2: QuantizedLinear,
}

impl QuantizedFeedForward {
    fn from_artifact(artifact: &crate::weights::Artifact, prefix: &str) -> ModelResult<Self> {
        let linear1 =
            QuantizedLinear::from_artifact(artifact, &format!("{prefix}.linear1.weight"), None)?;
        let hidden_size = linear1.input_dims();
        Ok(Self {
            hidden_size,
            linear1,
            linear2: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.linear2.weight"),
                None,
            )?,
        })
    }

    fn forward(&self, input: &Tensor3) -> ModelResult<Tensor3> {
        let rows = input.shape[0] * input.shape[1];
        if input.shape[2] != self.hidden_size {
            return Err(ModelError::InvalidShape(
                "feed-forward input hidden size is incorrect".to_string(),
            ));
        }
        let mut hidden = self.linear1.forward_f32(&input.values, rows)?;
        silu_in_place(&mut hidden);
        Ok(Tensor3 {
            shape: input.shape,
            values: self.linear2.forward_f32(&hidden, rows)?,
        })
    }
}

#[derive(Debug)]
struct ConformerConvolution {
    hidden_size: usize,
    pointwise1: QuantizedLinear,
    depthwise: DepthwiseConv1d,
    norm: LayerNorm,
    pointwise2: QuantizedLinear,
}

impl ConformerConvolution {
    fn from_artifact(artifact: &crate::weights::Artifact, layer: usize) -> ModelResult<Self> {
        let prefix = format!("encoder.layers.{layer}.conv");
        let pointwise1 = QuantizedLinear::from_artifact(
            artifact,
            &format!("{prefix}.pointwise_conv1.weight"),
            None,
        )?;
        let hidden_size = pointwise1.input_dims();
        Ok(Self {
            hidden_size,
            pointwise1,
            depthwise: DepthwiseConv1d::from_artifact(
                artifact,
                &format!("{prefix}.depthwise_conv.weight"),
                None,
            )?,
            norm: LayerNorm::from_artifact(artifact, &format!("{prefix}.norm"), 1.0e-5)?,
            pointwise2: QuantizedLinear::from_artifact(
                artifact,
                &format!("{prefix}.pointwise_conv2.weight"),
                None,
            )?,
        })
    }

    fn forward(&self, input: &Tensor3, cache: &mut CausalConv1dCache) -> ModelResult<Tensor3> {
        let [batch, time, channels] = input.shape;
        if batch != 1 || channels != self.hidden_size {
            return Err(ModelError::InvalidShape(
                "conformer convolution requires [1,time,hidden]".to_string(),
            ));
        }
        let projected = self.pointwise1.forward_f32(&input.values, time)?;
        let mut gated = vec![0.0; time * channels];
        for frame in 0..time {
            for channel in 0..channels {
                let first = projected[frame * 2 * channels + channel];
                let gate = projected[frame * 2 * channels + channels + channel];
                gated[frame * channels + channel] = first / (1.0 + (-gate).exp());
            }
        }
        let depthwise = self.depthwise.forward_f32(&gated, 1, time, cache)?;
        let mut normalized = self.norm.forward_f32(&depthwise.values, time)?;
        silu_in_place(&mut normalized);
        Ok(Tensor3 {
            shape: input.shape,
            values: self.pointwise2.forward_f32(&normalized, time)?,
        })
    }
}

/// Streaming state for one FastConformer block.
#[derive(Debug, Clone)]
pub struct EncoderLayerCache {
    attention: AttentionKvCache,
    convolution: CausalConv1dCache,
}

impl EncoderLayerCache {
    pub const fn attention_frames(&self) -> usize {
        self.attention.frames()
    }
}

/// One macaron FastConformer block in inference mode.
#[derive(Debug)]
pub struct FastConformerLayer {
    hidden_size: usize,
    heads: usize,
    max_cache_frames: usize,
    convolution_left_frames: usize,
    feed_forward1: QuantizedFeedForward,
    attention: RelativePositionAttention,
    convolution: ConformerConvolution,
    feed_forward2: QuantizedFeedForward,
    norm_feed_forward1: LayerNorm,
    norm_attention: LayerNorm,
    norm_convolution: LayerNorm,
    norm_feed_forward2: LayerNorm,
    norm_out: LayerNorm,
}

impl FastConformerLayer {
    pub fn from_artifact(
        artifact: &crate::weights::Artifact,
        layer: usize,
        heads: usize,
        max_cache_frames: usize,
    ) -> ModelResult<Self> {
        let prefix = format!("encoder.layers.{layer}");
        let feed_forward1 =
            QuantizedFeedForward::from_artifact(artifact, &format!("{prefix}.feed_forward1"))?;
        let hidden_size = feed_forward1.hidden_size;
        let convolution_weight = format!("{prefix}.conv.depthwise_conv.weight");
        let convolution_kernel = artifact
            .tensor_info(&convolution_weight)
            .ok_or_else(|| {
                crate::weights::ArtifactError::MissingArtifactTensor(convolution_weight.clone())
            })?
            .shape[2];
        Ok(Self {
            hidden_size,
            heads,
            max_cache_frames,
            convolution_left_frames: convolution_kernel - 1,
            feed_forward1,
            attention: RelativePositionAttention::from_artifact(artifact, layer, heads)?,
            convolution: ConformerConvolution::from_artifact(artifact, layer)?,
            feed_forward2: QuantizedFeedForward::from_artifact(
                artifact,
                &format!("{prefix}.feed_forward2"),
            )?,
            norm_feed_forward1: LayerNorm::from_artifact(
                artifact,
                &format!("{prefix}.norm_feed_forward1"),
                1.0e-5,
            )?,
            norm_attention: LayerNorm::from_artifact(
                artifact,
                &format!("{prefix}.norm_self_att"),
                1.0e-5,
            )?,
            norm_convolution: LayerNorm::from_artifact(
                artifact,
                &format!("{prefix}.norm_conv"),
                1.0e-5,
            )?,
            norm_feed_forward2: LayerNorm::from_artifact(
                artifact,
                &format!("{prefix}.norm_feed_forward2"),
                1.0e-5,
            )?,
            norm_out: LayerNorm::from_artifact(artifact, &format!("{prefix}.norm_out"), 1.0e-5)?,
        })
    }

    pub fn new_cache(&self) -> EncoderLayerCache {
        EncoderLayerCache {
            attention: AttentionKvCache::new(
                self.heads,
                self.hidden_size / self.heads,
                self.max_cache_frames,
            ),
            convolution: CausalConv1dCache::new(self.hidden_size, self.convolution_left_frames),
        }
    }

    pub fn forward(&self, input: &Tensor3, cache: &mut EncoderLayerCache) -> ModelResult<Tensor3> {
        let rows = input.shape[0] * input.shape[1];
        let normalized = Tensor3 {
            shape: input.shape,
            values: self.norm_feed_forward1.forward_f32(&input.values, rows)?,
        };
        let feed_forward1 = self.feed_forward1.forward(&normalized)?;
        let mut hidden = residual_add(input, &feed_forward1, 0.5)?;

        let normalized = Tensor3 {
            shape: hidden.shape,
            values: self.norm_attention.forward_f32(&hidden.values, rows)?,
        };
        let attention = self
            .attention
            .forward_streaming(&normalized, &mut cache.attention)?;
        hidden = residual_add(&hidden, &attention, 1.0)?;

        let normalized = Tensor3 {
            shape: hidden.shape,
            values: self.norm_convolution.forward_f32(&hidden.values, rows)?,
        };
        let convolution = self
            .convolution
            .forward(&normalized, &mut cache.convolution)?;
        hidden = residual_add(&hidden, &convolution, 1.0)?;

        let normalized = Tensor3 {
            shape: hidden.shape,
            values: self.norm_feed_forward2.forward_f32(&hidden.values, rows)?,
        };
        let feed_forward2 = self.feed_forward2.forward(&normalized)?;
        hidden = residual_add(&hidden, &feed_forward2, 0.5)?;
        Ok(Tensor3 {
            shape: hidden.shape,
            values: self.norm_out.forward_f32(&hidden.values, rows)?,
        })
    }
}

fn residual_add(left: &Tensor3, right: &Tensor3, scale: f32) -> ModelResult<Tensor3> {
    if left.shape != right.shape || left.values.len() != right.values.len() {
        return Err(ModelError::InvalidShape(
            "residual tensors must have identical shapes".to_string(),
        ));
    }
    Ok(Tensor3 {
        shape: left.shape,
        values: left
            .values
            .iter()
            .zip(&right.values)
            .map(|(left, right)| left + scale * right)
            .collect(),
    })
}

fn silu_in_place(values: &mut [f32]) {
    for value in values {
        *value /= 1.0 + (-*value).exp();
    }
}

/// All persistent state for one streaming encoder utterance.
#[derive(Debug, Clone)]
pub struct StreamingEncoderCache {
    subsampling: SubsamplingCache,
    layers: Vec<EncoderLayerCache>,
}

/// Factor-8 subsampling, FastConformer stack, and language-prompt projection.
#[derive(Debug)]
pub struct StreamingEncoder {
    config: EncoderConfig,
    input_mel_bins: usize,
    subsampling_channels: usize,
    subsampling: Conv2dSubsampling,
    layers: Vec<FastConformerLayer>,
    prompt: crate::model::PromptProjector,
}

/// Intermediate tensors used for reference validation and numerical diagnostics.
#[derive(Debug, Clone)]
pub struct EncoderTrace {
    pub subsampling: Tensor3,
    pub layers: Vec<Tensor3>,
    pub prompted: Tensor3,
}

impl StreamingEncoder {
    pub fn from_artifact(artifact: &crate::weights::Artifact) -> ModelResult<Self> {
        Self::from_artifact_with_config(artifact, EncoderConfig::nemotron_3_5(), 128, 256)
    }

    pub fn from_artifact_with_config(
        artifact: &crate::weights::Artifact,
        config: EncoderConfig,
        input_mel_bins: usize,
        subsampling_channels: usize,
    ) -> ModelResult<Self> {
        if config.num_layers == 0 || config.sliding_window < 2 {
            return Err(ModelError::InvalidShape(
                "streaming encoder needs layers and at least one cached frame".to_string(),
            ));
        }
        let layers = (0..config.num_layers)
            .map(|layer| {
                FastConformerLayer::from_artifact(
                    artifact,
                    layer,
                    config.num_heads,
                    config.sliding_window - 1,
                )
            })
            .collect::<ModelResult<Vec<_>>>()?;
        Ok(Self {
            config,
            input_mel_bins,
            subsampling_channels,
            subsampling: Conv2dSubsampling::from_artifact(artifact)?,
            layers,
            prompt: crate::model::PromptProjector::from_artifact(artifact)?,
        })
    }

    pub fn new_cache(&self) -> StreamingEncoderCache {
        StreamingEncoderCache {
            subsampling: SubsamplingCache::new(
                self.input_mel_bins,
                self.subsampling_channels,
                3,
                2,
                3,
            ),
            layers: self
                .layers
                .iter()
                .map(FastConformerLayer::new_cache)
                .collect(),
        }
    }

    pub fn encode_chunk(
        &self,
        features: &Tensor3,
        prompt: crate::model::LanguagePrompt,
        cache: &mut StreamingEncoderCache,
    ) -> ModelResult<Tensor3> {
        self.validate_cache(cache)?;
        let mut hidden = self.subsampling.forward(features, &mut cache.subsampling)?;
        self.validate_hidden_size(&hidden)?;
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.forward(&hidden, layer_cache)?;
        }
        self.prompt
            .forward_f32(&hidden.values, hidden.shape[0], hidden.shape[1], prompt)
    }

    pub fn encode_chunk_trace(
        &self,
        features: &Tensor3,
        prompt: crate::model::LanguagePrompt,
        cache: &mut StreamingEncoderCache,
    ) -> ModelResult<EncoderTrace> {
        self.validate_cache(cache)?;
        let mut hidden = self.subsampling.forward(features, &mut cache.subsampling)?;
        let subsampling = hidden.clone();
        self.validate_hidden_size(&hidden)?;
        let mut layer_outputs = Vec::with_capacity(self.layers.len());
        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            hidden = layer.forward(&hidden, layer_cache)?;
            layer_outputs.push(hidden.clone());
        }
        let prompted =
            self.prompt
                .forward_f32(&hidden.values, hidden.shape[0], hidden.shape[1], prompt)?;
        Ok(EncoderTrace {
            subsampling,
            layers: layer_outputs,
            prompted,
        })
    }

    fn validate_cache(&self, cache: &StreamingEncoderCache) -> ModelResult<()> {
        if cache.layers.len() != self.layers.len() {
            return Err(ModelError::InvalidShape(
                "encoder cache layer count does not match model".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_hidden_size(&self, hidden: &Tensor3) -> ModelResult<()> {
        if hidden.shape[2] != self.config.hidden_size {
            return Err(ModelError::InvalidShape(format!(
                "subsampling emitted {}, expected {} hidden channels",
                hidden.shape[2], self.config.hidden_size
            )));
        }
        Ok(())
    }
}

impl Conv2dSubsampling {
    pub fn from_artifact(artifact: &crate::weights::Artifact) -> ModelResult<Self> {
        let stem = Fp16Conv2d::from_artifact(
            artifact,
            "encoder.subsampling.conv_in.weight",
            "encoder.subsampling.conv_in.bias",
            2,
            1,
        )?;
        let mut stages = Vec::with_capacity(2);
        for layer in 0..2 {
            let prefix = format!("encoder.subsampling.layers.{layer}");
            let weight_name = format!("{prefix}.depthwise_conv.weight");
            let channels = artifact
                .tensor_info(&weight_name)
                .ok_or_else(|| {
                    crate::weights::ArtifactError::MissingArtifactTensor(weight_name.clone())
                })?
                .shape[0];
            stages.push((
                Fp16Conv2d::from_artifact(
                    artifact,
                    &weight_name,
                    &format!("{prefix}.depthwise_conv.bias"),
                    2,
                    channels,
                )?,
                QuantizedLinear::from_artifact(
                    artifact,
                    &format!("{prefix}.pointwise_conv.weight"),
                    Some(&format!("{prefix}.pointwise_conv.bias")),
                )?,
            ));
        }
        Self::new(
            stem,
            stages,
            QuantizedLinear::from_artifact(
                artifact,
                "encoder.subsampling.linear.weight",
                Some("encoder.subsampling.linear.bias"),
            )?,
        )
    }

    pub fn new(
        stem: Fp16Conv2d,
        stages: Vec<(Fp16Conv2d, QuantizedLinear)>,
        output: QuantizedLinear,
    ) -> ModelResult<Self> {
        if stages.is_empty() {
            return Err(ModelError::InvalidShape(
                "subsampling requires at least one depthwise-separable stage".to_string(),
            ));
        }
        Ok(Self {
            stem,
            stages,
            output,
        })
    }

    pub fn forward(&self, input: &Tensor3, cache: &mut SubsamplingCache) -> ModelResult<Tensor3> {
        let [batch, time, mel_bins] = input.shape;
        if batch != 1 || input.values.len() != time * mel_bins {
            return Err(ModelError::InvalidShape(
                "subsampling input must have shape [1,time,mel_bins]".to_string(),
            ));
        }
        if cache.layers.len() != self.stages.len() + 1 {
            return Err(ModelError::InvalidShape(
                "subsampling cache layer count does not match model".to_string(),
            ));
        }
        let mut valid_frames = time;
        let mut hidden = self.stem.forward_causal(
            &Tensor4 {
                shape: [1, time, mel_bins, 1],
                values: input.values.clone(),
            },
            &mut cache.layers[0],
        )?;
        valid_frames = self.stem.streaming_output_length(valid_frames);
        mask_invalid_time(&mut hidden, valid_frames);
        relu_in_place(&mut hidden.values);

        for (stage_index, (depthwise, pointwise)) in self.stages.iter().enumerate() {
            hidden = depthwise.forward_causal(&hidden, &mut cache.layers[stage_index + 1])?;
            valid_frames = depthwise.streaming_output_length(valid_frames);
            let rows = hidden.shape[1] * hidden.shape[2];
            hidden.values = pointwise.forward_f32(&hidden.values, rows)?;
            hidden.shape[3] = pointwise.output_dims();
            mask_invalid_time(&mut hidden, valid_frames);
            relu_in_place(&mut hidden.values);
        }

        let rows = hidden.shape[0] * hidden.shape[1];
        let flattened = channel_frequency_flatten(&hidden)?;
        let values = self.output.forward_f32(&flattened, rows)?;
        Ok(Tensor3 {
            shape: [1, rows, self.output.output_dims()],
            values,
        })
    }
}

fn mask_invalid_time(tensor: &mut Tensor4, valid_frames: usize) {
    let frame_width = tensor.shape[2] * tensor.shape[3];
    let start = valid_frames.min(tensor.shape[1]) * frame_width;
    tensor.values[start..].fill(0.0);
}

/// Reorders MLX NHWC pixels into PyTorch's channel-major `(C,F)` flattening.
pub fn channel_frequency_flatten(input: &Tensor4) -> ModelResult<Vec<f32>> {
    let [batch, time, frequency, channels] = input.shape;
    if input.values.len() != batch * time * frequency * channels {
        return Err(ModelError::InvalidShape(
            "NHWC tensor length does not match its shape".to_string(),
        ));
    }
    let mut output = vec![0.0; input.values.len()];
    for batch_index in 0..batch {
        for frame in 0..time {
            for channel in 0..channels {
                for bin in 0..frequency {
                    let source =
                        ((batch_index * time + frame) * frequency + bin) * channels + channel;
                    let destination =
                        ((batch_index * time + frame) * channels + channel) * frequency + bin;
                    output[destination] = input.values[source];
                }
            }
        }
    }
    Ok(output)
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

/// Offline form of the checkpoint's chunk-limited bidirectional attention mask.
pub fn chunked_attention_mask(
    sequence_length: usize,
    left_context: usize,
    right_context: usize,
) -> Vec<bool> {
    let chunk_size = right_context + 1;
    let left_context_chunks = left_context / chunk_size;
    let mut mask = vec![false; sequence_length * sequence_length];
    for query in 0..sequence_length {
        let query_chunk = query / chunk_size;
        for key in 0..sequence_length {
            let key_chunk = key / chunk_size;
            if query_chunk >= key_chunk && query_chunk - key_chunk <= left_context_chunks {
                mask[query * sequence_length + key] = true;
            }
        }
    }
    mask
}

/// Transformer-XL relative positions ordered from `L-1` through `-(L-1)`.
pub fn relative_position_encoding(hidden_size: usize, total_frames: usize) -> ModelResult<Tensor3> {
    if hidden_size == 0 || hidden_size % 2 != 0 || total_frames == 0 {
        return Err(ModelError::InvalidShape(
            "relative positions require a positive even hidden size and nonzero frames".to_string(),
        ));
    }
    let position_count = 2 * total_frames - 1;
    let mut values = Vec::with_capacity(position_count * hidden_size);
    for position in (-(total_frames as isize - 1)..=total_frames as isize - 1).rev() {
        for index in 0..hidden_size / 2 {
            let exponent = (2 * index) as f32 / hidden_size as f32;
            let angle = position as f32 / 10_000_f32.powf(exponent);
            values.push(angle.sin());
            values.push(angle.cos());
        }
    }
    Ok(Tensor3 {
        shape: [1, position_count, hidden_size],
        values,
    })
}

/// Shaw/Transformer-XL relative shift used by the official Transformers implementation.
pub fn relative_shift(
    attention_scores: &[f32],
    query_length: usize,
    position_length: usize,
) -> ModelResult<Vec<f32>> {
    if query_length == 0
        || position_length == 0
        || attention_scores.len() != query_length * position_length
    {
        return Err(ModelError::InvalidShape(
            "relative shift score dimensions are inconsistent".to_string(),
        ));
    }
    let mut padded = Vec::with_capacity(query_length * (position_length + 1));
    for row in attention_scores.chunks_exact(position_length) {
        padded.push(0.0);
        padded.extend_from_slice(row);
    }
    Ok(padded[query_length..].to_vec())
}

/// Encoder dimensions and streaming constants from the published checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub conv_kernel_size: usize,
    pub subsampling_factor: usize,
    pub sliding_window: usize,
    pub supported_lookahead: [usize; 4],
    pub default_lookahead: usize,
}

impl EncoderConfig {
    /// Exact architecture settings from `config.json`.
    pub const fn nemotron_3_5() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_layers: 24,
            num_heads: 8,
            conv_kernel_size: 9,
            subsampling_factor: 8,
            sliding_window: 57,
            supported_lookahead: [3, 0, 6, 13],
            default_lookahead: 3,
        }
    }
}

/// Exact audio/feature framing for one cache-aware streaming session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingChunkPlan {
    lookahead: usize,
    subsampling_factor: usize,
}

impl StreamingChunkPlan {
    pub fn new(config: &EncoderConfig, lookahead: usize) -> ModelResult<Self> {
        if !config.supported_lookahead.contains(&lookahead) {
            return Err(ModelError::UnsupportedLookahead {
                requested: lookahead,
                supported: config.supported_lookahead,
            });
        }
        Ok(Self {
            lookahead,
            subsampling_factor: config.subsampling_factor,
        })
    }

    /// Mel frames required for the centered first chunk.
    pub const fn first_mel_frames(&self) -> usize {
        1 + self.subsampling_factor * self.lookahead
    }

    /// Mel frames required for each subsequent uncentered chunk.
    pub const fn subsequent_mel_frames(&self) -> usize {
        self.subsampling_factor * (self.lookahead + 1)
    }

    /// PCM samples at 16 kHz required for the centered first chunk.
    pub const fn first_audio_samples(&self) -> usize {
        (self.first_mel_frames() - 1) * 160 + 200
    }

    /// PCM samples at 16 kHz required for each uncentered chunk.
    pub const fn subsequent_audio_samples(&self) -> usize {
        self.subsequent_mel_frames() * 160 + 400
    }

    /// Frames passed through every FastConformer block per call.
    pub const fn encoder_frames_per_chunk(&self) -> usize {
        self.lookahead + 1
    }

    /// New frames that become safe to emit after the lookahead is supplied.
    pub const fn emitted_frames_per_chunk(&self) -> usize {
        1
    }

    /// Algorithmic streaming latency at 80 ms per subsampled frame.
    pub const fn latency_ms(&self) -> usize {
        (self.lookahead + 1) * 80
    }
}

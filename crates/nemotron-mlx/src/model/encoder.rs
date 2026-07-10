use super::{
    CausalConv2dCache, Fp16Conv2d, ModelError, ModelResult, QuantizedLinear, Tensor3, Tensor4,
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

impl Conv2dSubsampling {
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
        let mut hidden = self.stem.forward_causal(
            &Tensor4 {
                shape: [1, time, mel_bins, 1],
                values: input.values.clone(),
            },
            &mut cache.layers[0],
        )?;
        relu_in_place(&mut hidden.values);

        for (stage_index, (depthwise, pointwise)) in self.stages.iter().enumerate() {
            hidden = depthwise.forward_causal(&hidden, &mut cache.layers[stage_index + 1])?;
            let rows = hidden.shape[1] * hidden.shape[2];
            hidden.values = pointwise.forward_f32(&hidden.values, rows)?;
            hidden.shape[3] = pointwise.output_dims();
            relu_in_place(&mut hidden.values);
        }

        let rows = hidden.shape[1];
        let values = self.output.forward_f32(&hidden.values, rows)?;
        Ok(Tensor3 {
            shape: [1, rows, self.output.output_dims()],
            values,
        })
    }
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

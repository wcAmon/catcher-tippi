use super::{ModelError, ModelResult};

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

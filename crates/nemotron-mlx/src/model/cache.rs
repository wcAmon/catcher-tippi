use super::{ModelError, ModelResult};

/// Left-context state for one causal depthwise-convolution stream.
#[derive(Debug, Clone)]
pub struct CausalConv1dCache {
    channels: usize,
    left_frames: usize,
    values: Vec<f32>,
}

impl CausalConv1dCache {
    /// Creates a zero-initialized cache with `left_frames * channels` values.
    pub fn new(channels: usize, left_frames: usize) -> Self {
        Self {
            channels,
            left_frames,
            values: vec![0.0; channels * left_frames],
        }
    }

    /// Current cached frames in `[time, channels]` order.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns old cache plus chunk and updates the cache to the newest frames.
    pub fn prepend_and_update(&mut self, chunk: &[f32], frames: usize) -> ModelResult<Vec<f32>> {
        if frames.checked_mul(self.channels) != Some(chunk.len()) {
            return Err(ModelError::InvalidShape(format!(
                "cache chunk has {} values, expected {}x{}",
                chunk.len(),
                frames,
                self.channels
            )));
        }

        let mut combined = Vec::with_capacity(self.values.len() + chunk.len());
        combined.extend_from_slice(&self.values);
        combined.extend_from_slice(chunk);
        let keep = self.left_frames * self.channels;
        self.values
            .copy_from_slice(&combined[combined.len() - keep..]);
        Ok(combined)
    }
}

/// Time-axis state used by one causal subsampling Conv2D layer.
#[derive(Debug, Clone)]
pub struct CausalConv2dCache {
    frequency_bins: usize,
    channels: usize,
    left_frames: usize,
    initial_extra_frames: usize,
    first_chunk: bool,
    values: Vec<f32>,
}

impl CausalConv2dCache {
    pub fn new(
        frequency_bins: usize,
        channels: usize,
        left_frames: usize,
        initial_extra_frames: usize,
    ) -> Self {
        Self {
            frequency_bins,
            channels,
            left_frames,
            initial_extra_frames,
            first_chunk: true,
            values: vec![0.0; frequency_bins * channels * left_frames],
        }
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub(crate) fn prepend_and_update(
        &mut self,
        chunk: &[f32],
        frames: usize,
        frequency_bins: usize,
        channels: usize,
    ) -> ModelResult<(Vec<f32>, usize)> {
        let frame_width = self.frequency_bins * self.channels;
        if frequency_bins != self.frequency_bins
            || channels != self.channels
            || chunk.len() != frames * frame_width
        {
            return Err(ModelError::InvalidShape(format!(
                "conv2d cache expected [1,{frames},{},{}]",
                self.frequency_bins, self.channels
            )));
        }

        let leading_frames = self.left_frames
            + if self.first_chunk {
                self.initial_extra_frames
            } else {
                0
            };
        let mut combined = Vec::with_capacity((leading_frames + frames) * frame_width);
        if self.first_chunk {
            combined.resize(self.initial_extra_frames * frame_width, 0.0);
        }
        combined.extend_from_slice(&self.values);
        combined.extend_from_slice(chunk);

        let mut history = Vec::with_capacity(self.values.len() + chunk.len());
        history.extend_from_slice(&self.values);
        history.extend_from_slice(chunk);
        let keep = self.left_frames * frame_width;
        self.values
            .copy_from_slice(&history[history.len() - keep..]);
        self.first_chunk = false;
        Ok((combined, leading_frames + frames))
    }
}

/// Keys and values visible to the current attention call in `[heads,time,head_dim]` order.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionKv {
    pub frames: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

/// Sliding-window attention state for one FastConformer layer and one utterance.
#[derive(Debug, Clone)]
pub struct AttentionKvCache {
    heads: usize,
    head_dim: usize,
    max_frames: usize,
    frames: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl AttentionKvCache {
    pub fn new(heads: usize, head_dim: usize, max_frames: usize) -> Self {
        Self {
            heads,
            head_dim,
            max_frames,
            frames: 0,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub const fn frames(&self) -> usize {
        self.frames
    }

    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn update(
        &mut self,
        current_keys: &[f32],
        current_values: &[f32],
        current_frames: usize,
    ) -> ModelResult<AttentionKv> {
        let current_len = self.heads * current_frames * self.head_dim;
        if current_keys.len() != current_len || current_values.len() != current_len {
            return Err(ModelError::InvalidShape(format!(
                "attention cache update expected {current_len} key/value elements"
            )));
        }
        let total_frames = self.frames + current_frames;
        let mut keys = Vec::with_capacity(self.heads * total_frames * self.head_dim);
        let mut values = Vec::with_capacity(keys.capacity());
        for head in 0..self.heads {
            let old_start = head * self.frames * self.head_dim;
            let old_end = old_start + self.frames * self.head_dim;
            let current_start = head * current_frames * self.head_dim;
            let current_end = current_start + current_frames * self.head_dim;
            keys.extend_from_slice(&self.keys[old_start..old_end]);
            keys.extend_from_slice(&current_keys[current_start..current_end]);
            values.extend_from_slice(&self.values[old_start..old_end]);
            values.extend_from_slice(&current_values[current_start..current_end]);
        }

        let retained_frames = total_frames.min(self.max_frames);
        let drop_frames = total_frames - retained_frames;
        let mut retained_keys = Vec::with_capacity(self.heads * retained_frames * self.head_dim);
        let mut retained_values = Vec::with_capacity(retained_keys.capacity());
        for head in 0..self.heads {
            let start = (head * total_frames + drop_frames) * self.head_dim;
            let end = start + retained_frames * self.head_dim;
            retained_keys.extend_from_slice(&keys[start..end]);
            retained_values.extend_from_slice(&values[start..end]);
        }
        self.frames = retained_frames;
        self.keys = retained_keys;
        self.values = retained_values;
        Ok(AttentionKv {
            frames: total_frames,
            keys,
            values,
        })
    }
}

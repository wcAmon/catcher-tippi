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

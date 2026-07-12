//! Push-based streaming Sortformer diarizer.
//!
//! Wires the offline model parts (mel frontend, Fast-Conformer `pre_encode`,
//! and the shared Conformer + Transformer + head tail) to the AOSC speaker
//! cache (`streaming_update`). Audio is pushed in arbitrary-sized pieces; each
//! completed chunk is diarized against the current `[spkcache | fifo | chunk]`
//! sequence and its per-frame (80 ms) speaker probabilities are returned.
//!
//! Chunk geometry mirrors NeMo's `streaming_feat_loader` (synchronous,
//! low-latency preset). For chunk `k`, with `sub = subsampling_factor` and mel
//! chunk width `chunk_mel = chunk_len * sub`:
//! - the chunk covers mel `[k * chunk_mel, k * chunk_mel + chunk_mel)`, clamped
//!   to the available mel frames on the last chunk;
//! - `left_off = min(left_context * sub, k * chunk_mel)` mel frames of history;
//! - `right_off = min(right_context * sub, total_mel - chunk_end)` look-ahead;
//! - the mel window `[chunk_start - left_off, chunk_end + right_off)` is
//!   `pre_encode`d into `lc + chunk_len + rc` embedding rows, where
//!   `lc = round(left_off / sub)` and `rc = ceil(right_off / sub)`.

use std::path::Path;

use nemotron_mlx::model::Tensor3;

use super::aosc::{StreamingConfig, StreamingState, streaming_update};
use crate::config::SortformerConfig;
use crate::model::{Diarizer, ModelError, ModelResult};

/// Streaming Sortformer diarizer driven by pushed audio.
#[derive(Debug)]
pub struct StreamingDiarizer {
    diarizer: Diarizer,
    config: StreamingConfig,
    state: StreamingState,
    /// Whole 16 kHz mono recording pushed so far. A recording is minutes long,
    /// so this stays a plain buffer; the mel cursor (`next_chunk`) tracks
    /// progress rather than dropping consumed samples.
    audio: Vec<f32>,
    /// Index of the next chunk to emit.
    next_chunk: usize,
    /// Conv subsampling factor (mel frames per output frame), 8 for v2.1.
    subsampling_factor: usize,
    finished: bool,
}

impl StreamingDiarizer {
    /// Loads the model from a converted artifact directory and initializes the
    /// low-latency-v2 AOSC state.
    pub fn from_artifact_dir(model_dir: impl AsRef<Path>) -> ModelResult<Self> {
        let model_dir = model_dir.as_ref();
        let diarizer = Diarizer::from_artifact_dir(model_dir)?;
        let sortformer_config = SortformerConfig::load(model_dir)
            .map_err(|error| ModelError::InvalidShape(error.to_string()))?;
        let config = StreamingConfig::low_latency_v2();
        let state = StreamingState::new(&config);
        Ok(Self {
            diarizer,
            config,
            state,
            audio: Vec::new(),
            next_chunk: 0,
            subsampling_factor: sortformer_config.subsampling_factor,
            finished: false,
        })
    }

    /// Output frame duration in milliseconds (80 ms for the v2.1 checkpoint).
    pub fn frame_ms(&self) -> u64 {
        self.diarizer.frame_ms()
    }

    /// Current `(fifo_len, spkcache_len)` — the AOSC queue depths. Mirrors
    /// NeMo's `state.fifo`/`state.spkcache` lengths and is used to assert
    /// against the reference `length_trajectory`.
    pub fn state_lengths(&self) -> (usize, usize) {
        (self.state.fifo.len(), self.state.spkcache.len())
    }

    /// Appends audio and emits speaker probabilities for every chunk that now
    /// has a full right context available. Returns per-frame `[f32; 4]` rows in
    /// chunk order.
    pub fn push_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<[f32; 4]>> {
        self.audio.extend_from_slice(audio);
        let total_mel = self.diarizer.frontend().frame_count(self.audio.len());
        let chunk_mel = self.chunk_mel();
        let right_max = self.config.right_context * self.subsampling_factor;
        let mut output = Vec::new();
        // Process a chunk only once its full right context is available, so a
        // push mid-recording sees the same look-ahead NeMo's offline loader did.
        while (self.next_chunk + 1) * chunk_mel + right_max <= total_mel {
            output.extend(self.process_chunk(self.next_chunk, total_mel)?);
            self.next_chunk += 1;
        }
        Ok(output)
    }

    /// Flushes the trailing chunks whose right context is shorter than the
    /// full look-ahead (down to `rc = 0` for the final chunk) and marks the
    /// stream finished.
    pub fn finish(&mut self) -> ModelResult<Vec<[f32; 4]>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let total_mel = self.diarizer.frontend().frame_count(self.audio.len());
        let chunk_mel = self.chunk_mel();
        let mut output = Vec::new();
        while self.next_chunk * chunk_mel < total_mel {
            output.extend(self.process_chunk(self.next_chunk, total_mel)?);
            self.next_chunk += 1;
        }
        self.finished = true;
        Ok(output)
    }

    /// Restores a fresh streaming state and clears the audio buffer.
    pub fn reset(&mut self) {
        self.state = StreamingState::new(&self.config);
        self.audio.clear();
        self.next_chunk = 0;
        self.finished = false;
    }

    /// Mel frames per chunk (`chunk_len * subsampling_factor`, 48 for v2.1).
    fn chunk_mel(&self) -> usize {
        self.config.chunk_len * self.subsampling_factor
    }

    /// Diarizes chunk `k` against the current cache and advances the AOSC state.
    fn process_chunk(&mut self, k: usize, total_mel: usize) -> ModelResult<Vec<[f32; 4]>> {
        let sub = self.subsampling_factor;
        let chunk_mel = self.chunk_mel();
        let emb_dim = self.config.emb_dim;

        let chunk_start = k * chunk_mel;
        let chunk_end = (chunk_start + chunk_mel).min(total_mel);
        let left_off = (self.config.left_context * sub).min(chunk_start);
        let right_off = (self.config.right_context * sub).min(total_mel - chunk_end);

        // Pre-encode the [left | chunk | right] mel window into embedding rows.
        let window_start = chunk_start - left_off;
        let window_len = left_off + (chunk_end - chunk_start) + right_off;
        let mel_window = self
            .diarizer
            .frontend()
            .extract_frames(&self.audio, window_start, window_len);
        let chunk_embs = self.diarizer.encoder().pre_encode(&mel_window)?;
        let chunk_rows: Vec<Vec<f32>> = chunk_embs
            .values
            .chunks_exact(emb_dim)
            .map(|row| row.to_vec())
            .collect();

        // Context sizes in embedding frames (NeMo: round left, ceil right).
        let lc = ((left_off as f32) / (sub as f32)).round() as usize;
        let rc = right_off.div_ceil(sub);

        // Assemble [spkcache | fifo | chunk] as one UNSCALED sequence and run
        // the streaming tail; forward_embedded scales the whole thing together.
        let spkcache_len = self.state.spkcache.len();
        let fifo_len = self.state.fifo.len();
        let frames = spkcache_len + fifo_len + chunk_rows.len();
        let mut assembled = Vec::with_capacity(frames * emb_dim);
        for row in &self.state.spkcache {
            assembled.extend_from_slice(row);
        }
        for row in &self.state.fifo {
            assembled.extend_from_slice(row);
        }
        assembled.extend_from_slice(&chunk_embs.values);
        let embedded = Tensor3 {
            shape: [1, frames, emb_dim],
            values: assembled,
        };
        let preds: Vec<Vec<f32>> = self
            .diarizer
            .forward_embedded_preds(&embedded)?
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        // The chunk region sits after [spkcache | fifo | lc] in the preds; the
        // same slice streaming_update consumes as this step's chunk preds.
        let chunk_len = chunk_rows.len() - lc - rc;
        let base = spkcache_len + fifo_len + lc;
        let chunk_preds: Vec<[f32; 4]> = preds[base..base + chunk_len]
            .iter()
            .map(|row| [row[0], row[1], row[2], row[3]])
            .collect();

        streaming_update(&mut self.state, &self.config, &chunk_rows, &preds, lc, rc);
        Ok(chunk_preds)
    }
}

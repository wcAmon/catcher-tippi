use crate::{audio::LogMelFrontend, weights::Artifact};

use super::{
    EncoderConfig, LanguagePrompt, ModelError, ModelResult, PredictionState, StreamingChunkPlan,
    StreamingEncoder, StreamingEncoderCache, StreamingRnntDecoder, Tensor3, TimedToken,
};

/// One exact audio window ready for the Nemotron log-mel frontend.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub center: bool,
    pub mel_frames: usize,
}

/// Buffers arbitrary microphone blocks into exact cache-aware model windows.
#[derive(Debug, Clone)]
pub struct AudioChunkScheduler {
    plan: StreamingChunkPlan,
    audio: Vec<f32>,
    first_processed: bool,
    mel_frame_index: usize,
    finished: bool,
}

impl AudioChunkScheduler {
    pub const fn new(plan: StreamingChunkPlan) -> Self {
        Self {
            plan,
            audio: Vec::new(),
            first_processed: false,
            mel_frame_index: 0,
            finished: false,
        }
    }

    /// Appends arbitrary 16 kHz samples and returns every newly complete window.
    pub fn push(&mut self, samples: &[f32]) -> ModelResult<Vec<AudioChunk>> {
        self.ensure_open()?;
        self.audio.extend_from_slice(samples);
        let mut chunks = Vec::new();
        loop {
            if !self.first_processed {
                let length = self.plan.first_audio_samples();
                if self.audio.len() < length {
                    break;
                }
                chunks.push(AudioChunk {
                    samples: self.audio[..length].to_vec(),
                    center: true,
                    mel_frames: self.plan.first_mel_frames(),
                });
                self.first_processed = true;
                self.mel_frame_index = self.plan.first_mel_frames();
                continue;
            }

            let start = self.next_start()?;
            let length = self.plan.subsequent_audio_samples();
            if self.audio.len() < start + length {
                break;
            }
            chunks.push(AudioChunk {
                samples: self.audio[start..start + length].to_vec(),
                center: false,
                mel_frames: self.plan.subsequent_mel_frames(),
            });
            self.mel_frame_index += self.plan.subsequent_mel_frames();
        }
        Ok(chunks)
    }

    /// Pads and returns the final incomplete window, if the utterance is non-empty.
    pub fn finish(&mut self) -> ModelResult<Option<AudioChunk>> {
        self.ensure_open()?;
        self.finished = true;
        if self.audio.is_empty() {
            return Ok(None);
        }

        if !self.first_processed {
            self.first_processed = true;
            self.mel_frame_index = self.plan.first_mel_frames();
            return Ok(Some(AudioChunk {
                samples: padded_slice(&self.audio, 0, self.plan.first_audio_samples()),
                center: true,
                mel_frames: self.plan.first_mel_frames(),
            }));
        }

        let start = self.next_start()?;
        if start >= self.audio.len() {
            return Ok(None);
        }
        self.mel_frame_index += self.plan.subsequent_mel_frames();
        Ok(Some(AudioChunk {
            samples: padded_slice(&self.audio, start, self.plan.subsequent_audio_samples()),
            center: false,
            mel_frames: self.plan.subsequent_mel_frames(),
        }))
    }

    /// Clears buffered audio and begins a new utterance.
    pub fn reset(&mut self) {
        self.audio.clear();
        self.first_processed = false;
        self.mel_frame_index = 0;
        self.finished = false;
    }

    fn ensure_open(&self) -> ModelResult<()> {
        if self.finished {
            return Err(ModelError::InvalidShape(
                "streaming audio session is already finished".to_string(),
            ));
        }
        Ok(())
    }

    fn next_start(&self) -> ModelResult<usize> {
        (self.mel_frame_index * 160)
            .checked_sub(512 / 2)
            .ok_or_else(|| ModelError::InvalidShape("streaming audio offset underflow".to_string()))
    }
}

/// One mono 16 kHz cache-aware encoder/RNNT session.
pub struct StreamingTranscriber {
    frontend: LogMelFrontend,
    encoder: StreamingEncoder,
    encoder_cache: StreamingEncoderCache,
    decoder: StreamingRnntDecoder,
    decoder_state: PredictionState,
    prompt: LanguagePrompt,
    scheduler: AudioChunkScheduler,
    frames_seen: u64,
}

impl StreamingTranscriber {
    pub fn new(artifact: &Artifact, language: &str, lookahead: usize) -> ModelResult<Self> {
        let config = EncoderConfig::nemotron_3_5();
        let plan = StreamingChunkPlan::new(&config, lookahead)?;
        let encoder = StreamingEncoder::from_artifact(artifact)?;
        let encoder_cache = encoder.new_cache();
        let decoder = StreamingRnntDecoder::from_artifact(artifact)?;
        let decoder_state = decoder.new_state();
        Ok(Self {
            frontend: LogMelFrontend::nemotron(),
            encoder,
            encoder_cache,
            decoder,
            decoder_state,
            prompt: LanguagePrompt::from_code(language)?,
            scheduler: AudioChunkScheduler::new(plan),
            frames_seen: 0,
        })
    }

    /// Appends arbitrary mono 16 kHz samples and returns newly emitted tokens
    /// carrying their GLOBAL utterance frame index.
    pub fn push_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<TimedToken>> {
        let mut output = Vec::new();
        for chunk in self.scheduler.push(audio)? {
            output.extend(self.decode_chunk(chunk)?);
        }
        Ok(output)
    }

    /// Pads the last incomplete window and returns its newly emitted tokens.
    pub fn finish(&mut self) -> ModelResult<Vec<TimedToken>> {
        let Some(chunk) = self.scheduler.finish()? else {
            return Ok(Vec::new());
        };
        self.decode_chunk(chunk)
    }

    /// Starts a new utterance while retaining the already loaded model weights.
    pub fn reset(&mut self) -> ModelResult<()> {
        self.encoder_cache = self.encoder.new_cache();
        self.decoder_state = self.decoder.new_state();
        self.scheduler.reset();
        self.frames_seen = 0;
        Ok(())
    }

    /// Transcribes one complete mono 16 kHz utterance and returns non-blank tokens.
    pub fn transcribe_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<TimedToken>> {
        let mut output = self.push_samples(audio)?;
        output.extend(self.finish()?);
        Ok(output)
    }

    fn decode_chunk(&mut self, chunk: AudioChunk) -> ModelResult<Vec<TimedToken>> {
        let features = self.frontend.extract(&chunk.samples, chunk.center);
        self.validate_feature_count(&features, chunk.mel_frames)?;
        self.decode_features(features)
    }

    fn validate_feature_count(&self, features: &[Vec<f32>], expected: usize) -> ModelResult<()> {
        if features.len() != expected || features.iter().any(|frame| frame.len() != 128) {
            return Err(ModelError::InvalidShape(format!(
                "audio frontend produced {} frames, expected {expected}x128",
                features.len()
            )));
        }
        Ok(())
    }

    fn decode_features(&mut self, features: Vec<Vec<f32>>) -> ModelResult<Vec<TimedToken>> {
        let time = features.len();
        let encoded = self.encoder.encode_chunk(
            &Tensor3 {
                shape: [1, time, 128],
                values: features.into_iter().flatten().collect(),
            },
            self.prompt,
            &mut self.encoder_cache,
        )?;
        // `encode_chunk` has already mutated the encoder cache in place, so
        // this window counts as consumed no matter how decoding goes below.
        // Advance `frames_seen` here — decoupled from `decode_frames`'s
        // outcome — so a surfaced decode error cannot desync the global frame
        // clock from the encoder cache when the caller keeps pushing audio.
        let offset = self.frames_seen;
        self.frames_seen += encoded.shape[1] as u64;
        let tokens = self
            .decoder
            .decode_frames(&encoded, &mut self.decoder_state)?
            .into_iter()
            .map(|token| TimedToken {
                id: token.id,
                frame: token.frame + offset,
            })
            .collect();
        Ok(tokens)
    }
}

fn padded_slice(audio: &[f32], start: usize, length: usize) -> Vec<f32> {
    let mut output = vec![0.0; length];
    if start < audio.len() {
        let available = length.min(audio.len() - start);
        output[..available].copy_from_slice(&audio[start..start + available]);
    }
    output
}

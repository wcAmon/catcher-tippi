use crate::{audio::LogMelFrontend, weights::Artifact};

use super::{
    EncoderConfig, LanguagePrompt, ModelError, ModelResult, PredictionState, StreamingChunkPlan,
    StreamingEncoder, StreamingEncoderCache, StreamingRnntDecoder, Tensor3,
};

/// One mono 16 kHz cache-aware encoder/RNNT session.
pub struct StreamingTranscriber {
    frontend: LogMelFrontend,
    encoder: StreamingEncoder,
    encoder_cache: StreamingEncoderCache,
    decoder: StreamingRnntDecoder,
    decoder_state: PredictionState,
    prompt: LanguagePrompt,
    plan: StreamingChunkPlan,
    consumed: bool,
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
            plan,
            consumed: false,
        })
    }

    /// Transcribes one complete mono 16 kHz utterance and returns non-blank token IDs.
    pub fn transcribe_samples(&mut self, audio: &[f32]) -> ModelResult<Vec<u32>> {
        if self.consumed {
            return Err(ModelError::InvalidShape(
                "a streaming transcriber session can consume only one utterance".to_string(),
            ));
        }
        self.consumed = true;
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let mut output = Vec::new();
        let first_audio = padded_slice(audio, 0, self.plan.first_audio_samples());
        let first_features = self.frontend.extract(&first_audio, true);
        self.validate_feature_count(&first_features, self.plan.first_mel_frames())?;
        output.extend(self.decode_features(first_features)?);

        let mut mel_frame_index = self.plan.first_mel_frames();
        loop {
            let start = mel_frame_index * 160;
            let Some(start) = start.checked_sub(512 / 2) else {
                return Err(ModelError::InvalidShape(
                    "streaming audio offset underflow".to_string(),
                ));
            };
            if start >= audio.len() {
                break;
            }
            let chunk = padded_slice(audio, start, self.plan.subsequent_audio_samples());
            let features = self.frontend.extract(&chunk, false);
            self.validate_feature_count(&features, self.plan.subsequent_mel_frames())?;
            output.extend(self.decode_features(features)?);
            mel_frame_index += self.plan.subsequent_mel_frames();
        }
        Ok(output)
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

    fn decode_features(&mut self, features: Vec<Vec<f32>>) -> ModelResult<Vec<u32>> {
        let time = features.len();
        let encoded = self.encoder.encode_chunk(
            &Tensor3 {
                shape: [1, time, 128],
                values: features.into_iter().flatten().collect(),
            },
            self.prompt,
            &mut self.encoder_cache,
        )?;
        self.decoder
            .decode_frames(&encoded, &mut self.decoder_state)
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

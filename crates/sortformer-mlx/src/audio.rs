//! NeMo-compatible log-mel frontend with per-feature normalization.

use rustfft::{FftPlanner, num_complex::Complex};

use crate::config::SortformerConfig;

/// Offline mel-spectrogram extractor matching NeMo preprocessing.
#[derive(Debug)]
pub struct MelFrontend {
    sample_rate: usize,
    window_length: usize,
    hop_length: usize,
    n_fft: usize,
    preemphasis: f32,
    filterbank: Vec<Vec<f32>>, // [n_mels][n_fft / 2 + 1]
}

const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8; // 2^-24, NeMo's log zero guard.

impl MelFrontend {
    /// Input sampling rate in hertz.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Builds the frontend from the model configuration.
    pub fn new(config: &SortformerConfig) -> Self {
        let window_length = (config.window_seconds * config.sample_rate as f64).round() as usize;
        let hop_length = (config.hop_seconds * config.sample_rate as f64).round() as usize;
        Self {
            sample_rate: config.sample_rate,
            window_length,
            hop_length,
            n_fft: config.n_fft,
            preemphasis: config.preemphasis,
            filterbank: slaney_mel_filterbank(
                config.n_mels,
                config.n_fft,
                config.sample_rate,
                0.0,
                config.sample_rate as f32 / 2.0,
            ),
        }
    }

    /// Extracts log-mel frames and applies per-feature normalization.
    ///
    /// Phase-2 surface: the checkpoint's preprocessor uses `normalize: "NA"`,
    /// so `Diarizer` calls `extract` directly today; this is kept as the
    /// entry point a future per-feature-normalized streaming frontend would
    /// use, and is exercised by the audio parity tests.
    pub fn extract_normalized(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let mut frames = self.extract(audio);
        normalize_per_feature(&mut frames);
        frames
    }

    /// Extracts raw log-mel frames without normalization.
    ///
    /// The Sortformer checkpoint's preprocessor is configured with
    /// `normalize: "NA"`, so the encoder consumes these unnormalized frames.
    pub fn extract(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        self.extract_frames(audio, 0, self.frame_count(audio.len()))
    }

    /// Number of log-mel frames `extract` emits for `num_samples` input samples.
    ///
    /// NeMo keeps `get_seq_len = floor((T + 2*(n_fft/2) - n_fft) / hop)` frames
    /// (features.py:403-407). `torch.stft(center=True)` produces `floor(T/hop) +
    /// 1` columns, but NeMo's reported sequence length drops the trailing
    /// column, so we emit exactly `floor(T/hop)` frames. Streaming (Task 7) uses
    /// this to know how many mel frames the current audio buffer can supply.
    pub fn frame_count(&self, num_samples: usize) -> usize {
        if num_samples == 0 {
            return 0;
        }
        let pad_amount = (self.n_fft / 2) * 2;
        (num_samples + pad_amount).saturating_sub(self.n_fft) / self.hop_length
    }

    /// Extracts log-mel frames `[start, start + count)` directly from `audio`.
    ///
    /// Bit-for-bit identical to slicing `extract(audio)` at the same indices:
    /// both share `frame_mel`, which reads only the samples a centered window
    /// touches (`frame * hop ± n_fft / 2`) and computes preemphasis locally
    /// (`y[s] = x[s] - k * x[s-1]`, `x[-1] = 0`). Streaming pre-encodes each
    /// chunk's `[lc | chunk | rc]` mel window with this, without re-running the
    /// whole recording through the frontend every push.
    pub fn extract_frames(&self, audio: &[f32], start: usize, count: usize) -> Vec<Vec<f32>> {
        // NeMo builds the window with torch.hann_window(win_length,
        // periodic=False), i.e. the symmetric convention dividing by N - 1.
        let window: Vec<f32> = (0..self.window_length)
            .map(|index| {
                let phase =
                    2.0 * std::f32::consts::PI * index as f32 / (self.window_length - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let bins = self.n_fft / 2 + 1;
        (start..start + count)
            .map(|frame| self.frame_mel(audio, frame, &window, fft.as_ref(), bins))
            .collect()
    }

    /// One frame's log-mel vector, reading `audio` through the same centered,
    /// zero-padded, preemphasized window `extract` uses.
    fn frame_mel(
        &self,
        audio: &[f32],
        frame: usize,
        window: &[f32],
        fft: &dyn rustfft::Fft<f32>,
        bins: usize,
    ) -> Vec<f32> {
        // torch.stft's center=True convention zero-pads the signal by `n_fft / 2`
        // on each side (features.py:377-386); with `win_length < n_fft` it also
        // centers the analysis window inside the `n_fft` FFT frame.
        let pad = self.n_fft / 2;
        let window_offset = (self.n_fft - self.window_length) / 2;
        let start = frame * self.hop_length;
        let mut buffer = vec![Complex::new(0.0f32, 0.0f32); self.n_fft];
        for (index, weight) in window.iter().enumerate() {
            // Padded index into `[n_fft/2 zeros | preemphasized signal | zeros]`.
            let padded_index = start + window_offset + index;
            let sample = if padded_index < pad {
                0.0
            } else {
                let signal_index = padded_index - pad;
                if signal_index >= audio.len() {
                    0.0
                } else {
                    let previous = if signal_index == 0 {
                        0.0
                    } else {
                        audio[signal_index - 1]
                    };
                    audio[signal_index] - self.preemphasis * previous
                }
            };
            buffer[window_offset + index] = Complex::new(sample * weight, 0.0);
        }
        fft.process(&mut buffer);
        self.filterbank
            .iter()
            .map(|filter| {
                let energy: f32 = filter
                    .iter()
                    .zip(&buffer[..bins])
                    .map(|(weight, value)| weight * value.norm_sqr())
                    .sum();
                (energy + LOG_ZERO_GUARD).ln()
            })
            .collect()
    }
}

fn normalize_per_feature(frames: &mut [Vec<f32>]) {
    if frames.len() < 2 {
        return;
    }
    let bins = frames[0].len();
    let count = frames.len() as f32;
    for bin in 0..bins {
        let mean: f32 = frames.iter().map(|frame| frame[bin]).sum::<f32>() / count;
        let variance: f32 = frames
            .iter()
            .map(|frame| (frame[bin] - mean).powi(2))
            .sum::<f32>()
            / (count - 1.0);
        let std = variance.sqrt() + 1e-5;
        for frame in frames.iter_mut() {
            frame[bin] = (frame[bin] - mean) / std;
        }
    }
}

fn slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<Vec<f32>> {
    fn hz_to_mel(hz: f32) -> f32 {
        // Slaney scale: linear below 1 kHz, logarithmic above.
        if hz < 1_000.0 {
            hz * 3.0 / 200.0
        } else {
            15.0 + (hz / 1_000.0).ln() / (6.4f32.ln() / 27.0)
        }
    }
    fn mel_to_hz(mel: f32) -> f32 {
        if mel < 15.0 {
            mel * 200.0 / 3.0
        } else {
            1_000.0 * ((mel - 15.0) * (6.4f32.ln() / 27.0)).exp()
        }
    }
    let bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let edges: Vec<f32> = (0..n_mels + 2)
        .map(|index| mel_to_hz(mel_min + (mel_max - mel_min) * index as f32 / (n_mels + 1) as f32))
        .collect();
    let bin_hz: Vec<f32> = (0..bins)
        .map(|index| index as f32 * sample_rate as f32 / n_fft as f32)
        .collect();
    (0..n_mels)
        .map(|mel| {
            let (lower, center, upper) = (edges[mel], edges[mel + 1], edges[mel + 2]);
            let norm = 2.0 / (upper - lower); // Slaney area normalization.
            bin_hz
                .iter()
                .map(|&hz| {
                    let weight = if hz <= lower || hz >= upper {
                        0.0
                    } else if hz <= center {
                        (hz - lower) / (center - lower)
                    } else {
                        (upper - hz) / (upper - center)
                    };
                    weight * norm
                })
                .collect()
        })
        .collect()
}

use std::sync::Arc;

use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

/// Stateless Nemotron 3.5 preemphasis, STFT, and Slaney log-mel frontend.
pub struct LogMelFrontend {
    sample_rate: usize,
    hop_length: usize,
    n_fft: usize,
    preemphasis: f32,
    window: Vec<f32>,
    mel_filters: Vec<Vec<f32>>,
    fft: Arc<dyn Fft<f32>>,
}

impl LogMelFrontend {
    /// Creates the exact feature configuration published with Nemotron 3.5 ASR.
    pub fn nemotron() -> Self {
        Self::new(16_000, 160, 512, 400, 128, 0.97)
    }

    /// Creates a log-mel frontend with an explicitly specified configuration.
    pub fn new(
        sample_rate: usize,
        hop_length: usize,
        n_fft: usize,
        win_length: usize,
        mel_bins: usize,
        preemphasis: f32,
    ) -> Self {
        assert!(win_length <= n_fft);
        assert!(sample_rate > 0 && hop_length > 0 && n_fft > 0 && mel_bins > 0);

        let mut window = vec![0.0; n_fft];
        let left_padding = (n_fft - win_length) / 2;
        for index in 0..win_length {
            let phase = 2.0 * std::f32::consts::PI * index as f32 / (win_length - 1) as f32;
            window[left_padding + index] = 0.5 - 0.5 * phase.cos();
        }

        let mel_filters = slaney_mel_filters(sample_rate, n_fft, mel_bins);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(n_fft);
        Self {
            sample_rate,
            hop_length,
            n_fft,
            preemphasis,
            window,
            mel_filters,
            fft,
        }
    }

    /// Input sampling rate in hertz.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Extracts valid log-mel frames. Centered mode matches offline and first-chunk extraction.
    pub fn extract(&self, audio: &[f32], center: bool) -> Vec<Vec<f32>> {
        let frame_count = if center {
            audio.len() / self.hop_length
        } else if audio.len() < self.n_fft {
            0
        } else {
            (audio.len() - self.n_fft) / self.hop_length + 1
        };
        if frame_count == 0 {
            return Vec::new();
        }

        let emphasized = preemphasize(audio, self.preemphasis);
        let mut result = Vec::with_capacity(frame_count);
        let mut fft_buffer = vec![Complex32::new(0.0, 0.0); self.n_fft];
        let mut power = vec![0.0_f32; self.n_fft / 2 + 1];

        for frame in 0..frame_count {
            fft_buffer.fill(Complex32::new(0.0, 0.0));
            let start = frame as isize * self.hop_length as isize
                - if center { self.n_fft as isize / 2 } else { 0 };
            for (fft_index, coefficient) in self.window.iter().enumerate() {
                let sample_index = start + fft_index as isize;
                if sample_index >= 0 && (sample_index as usize) < emphasized.len() {
                    fft_buffer[fft_index].re = emphasized[sample_index as usize] * coefficient;
                }
            }

            self.fft.process(&mut fft_buffer);
            for (bin, value) in fft_buffer.iter().take(power.len()).enumerate() {
                power[bin] = value.re.mul_add(value.re, value.im * value.im);
            }

            let mut mel_frame = Vec::with_capacity(self.mel_filters.len());
            for filter in &self.mel_filters {
                let energy = filter
                    .iter()
                    .zip(power.iter())
                    .fold(0.0_f32, |sum, (weight, value)| weight.mul_add(*value, sum));
                mel_frame.push((energy + LOG_ZERO_GUARD).ln());
            }
            result.push(mel_frame);
        }
        result
    }
}

fn preemphasize(audio: &[f32], coefficient: f32) -> Vec<f32> {
    let mut output = Vec::with_capacity(audio.len());
    if let Some(first) = audio.first() {
        output.push(*first);
        output.extend(
            audio[1..]
                .iter()
                .zip(audio.iter())
                .map(|(current, previous)| current - coefficient * previous),
        );
    }
    output
}

fn slaney_mel_filters(sample_rate: usize, n_fft: usize, mel_bins: usize) -> Vec<Vec<f32>> {
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(sample_rate as f32 / 2.0);
    let mel_step = (max_mel - min_mel) / (mel_bins + 1) as f32;
    let mel_frequencies = (0..mel_bins + 2)
        .map(|index| mel_to_hz(min_mel + mel_step * index as f32))
        .collect::<Vec<_>>();
    let fft_step = sample_rate as f32 / n_fft as f32;

    (0..mel_bins)
        .map(|mel| {
            let lower = mel_frequencies[mel];
            let center = mel_frequencies[mel + 1];
            let upper = mel_frequencies[mel + 2];
            let normalization = 2.0 / (upper - lower);
            (0..n_fft / 2 + 1)
                .map(|bin| {
                    let frequency = fft_step * bin as f32;
                    let rising = (frequency - lower) / (center - lower);
                    let falling = (upper - frequency) / (upper - center);
                    rising.min(falling).max(0.0) * normalization
                })
                .collect()
        })
        .collect()
}

fn hz_to_mel(frequency: f32) -> f32 {
    const FREQ_SPACING: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / FREQ_SPACING;
    const LOG_STEP: f32 = 0.068_751_775;
    if frequency >= MIN_LOG_HZ {
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / LOG_STEP
    } else {
        frequency / FREQ_SPACING
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    const FREQ_SPACING: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / FREQ_SPACING;
    const LOG_STEP: f32 = 0.068_751_775;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        mel * FREQ_SPACING
    }
}

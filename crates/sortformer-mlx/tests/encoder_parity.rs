//! Numerical parity against captured NeMo Sortformer activations.
//!
//! `tests/fixtures/sortformer_reference.json` records full-context module
//! outputs for `hello-streaming.wav`. The checkpoint preprocessor uses
//! `normalize: "NA"`, so the encoder consumes *unnormalized* log-mel frames
//! (reference `features` mean is -10.18, clearly not per-feature normalized).

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use nemotron_mlx::weights::Artifact;
use sortformer_mlx::audio::MelFrontend;
use sortformer_mlx::config::SortformerConfig;
use sortformer_mlx::model::Encoder;

/// MLX evaluates onto a process-global Metal command buffer that is not safe
/// for concurrent submission; two MLX-driving tests running on separate
/// threads abort with "encodeWaitForEvent:value: with uncommitted encoder".
/// Each MLX-driving test holds this lock so they run serially (same pattern as
/// tests/diarizer_parity.rs).
static MLX_PIPELINE: Mutex<()> = Mutex::new(());

fn serialize_mlx() -> MutexGuard<'static, ()> {
    MLX_PIPELINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(serde::Deserialize)]
struct Summary {
    shape: Vec<usize>,
    rms: f64,
    first: Vec<f64>,
}

#[derive(serde::Deserialize)]
struct Reference {
    features: Summary,
    encoder_out: Summary,
}

fn artifact_dir() -> PathBuf {
    std::env::var_os("SORTFORMER_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set SORTFORMER_MLX_ARTIFACT to a converted artifact directory")
}

fn fixture_audio() -> Vec<f32> {
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect()
}

fn reference() -> Reference {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_reference.json"
    )))
    .unwrap()
}

#[derive(serde::Deserialize)]
struct PreEncodeReference {
    frames: usize,
    dim: usize,
    values: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct StreamingReference {
    chunk0_pre_encode: PreEncodeReference,
}

fn streaming_reference() -> StreamingReference {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_streaming_reference.json"
    )))
    .unwrap()
}

fn fixture_config() -> SortformerConfig {
    SortformerConfig::load(artifact_dir()).unwrap()
}

fn load_encoder() -> Encoder {
    let artifact = Artifact::load(artifact_dir()).unwrap();
    let config = fixture_config();
    Encoder::from_artifact(&artifact, &config).unwrap()
}

fn conversation_audio() -> Vec<f32> {
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/conversation.wav"
    ))
    .unwrap();
    reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect()
}

/// Returns `rms(actual - expected) / rms(expected)`.
fn relative_rms(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    let diff: Vec<f32> = actual.iter().zip(expected).map(|(a, e)| a - e).collect();
    rms(&diff) / rms(expected).max(1e-9)
}

/// Asserts `rms(actual - expected) / rms(expected)` is below `tolerance`.
fn assert_relative_rms_below(actual: &[f32], expected: &[f32], tolerance: f64) {
    let relative = relative_rms(actual, expected);
    assert!(
        relative < tolerance,
        "relative rms {relative} exceeds tolerance {tolerance}"
    );
}

fn rms(values: &[f32]) -> f64 {
    let count = values.len() as f64;
    (values.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / count).sqrt()
}

/// Checks whole-tensor rms and the reference's first-64 flattened values.
///
/// `actual_first` must be re-indexed by the caller to walk the same layout the
/// reference tensor was flattened in (recorded in `Summary::shape`).
fn assert_close(
    name: &str,
    full: &[f32],
    actual_first: &[f32],
    expected: &Summary,
    tolerance: f64,
) {
    let actual_rms = rms(full);
    assert!(
        (actual_rms - expected.rms).abs() <= tolerance * expected.rms.abs().max(1e-3),
        "{name} rms {actual_rms} vs reference {}",
        expected.rms
    );
    for (index, value) in expected.first.iter().enumerate() {
        let difference = (actual_first[index] as f64 - value).abs();
        assert!(
            difference <= tolerance * expected.rms.abs().max(1e-3) * 10.0,
            "{name}[{index}] {} vs {value}",
            actual_first[index]
        );
    }
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn mel_features_match_nemo_preprocessor() {
    let _guard = serialize_mlx();
    let reference = reference();
    let config = SortformerConfig::load(artifact_dir()).unwrap();
    // The checkpoint preprocessor does not normalize (`normalize: "NA"`).
    let frames = MelFrontend::new(&config).extract(&fixture_audio());
    // Reference layout is [1, n_mels, frames]; ours is [frames][n_mels].
    assert_eq!(reference.features.shape[1], config.n_mels);
    // NeMo's `torch.stft(center=True)` yields `floor(T/hop) + 1` columns, so
    // the captured raw-preprocessor tensor has one extra trailing column. NeMo
    // reports `get_seq_len = floor(T/hop)` and the model consumes only that
    // many frames, which is exactly what `extract` now emits, so our frame
    // count is the reference's minus that dropped trailing column.
    assert_eq!(reference.features.shape[2], frames.len() + 1);
    // Reference `first` walks mel bin 0 across time.
    let bin0: Vec<f32> = frames.iter().map(|frame| frame[0]).take(64).collect();
    let flat: Vec<f32> = frames.iter().flatten().copied().collect();
    assert_close("features", &flat, &bin0, &reference.features, 0.02);
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn pre_encode_matches_nemo_streaming_chunk0() {
    let _guard = serialize_mlx();
    let reference = streaming_reference();
    let config = fixture_config();
    let encoder = load_encoder();
    let audio = conversation_audio();
    let mel = MelFrontend::new(&config).extract(&audio);
    // chunk 0: no left context, 48 chunk + 56 right-context mel frames.
    let chunk0 = &mel[..(48 + 56).min(mel.len())];
    let ours = encoder.pre_encode(chunk0).unwrap();
    assert_eq!(ours.shape[1], reference.chunk0_pre_encode.frames);
    assert_eq!(ours.shape[2], reference.chunk0_pre_encode.dim);
    let dim = reference.chunk0_pre_encode.dim;

    // Task 5b aligned the MelFrontend's boundary behavior with NeMo: it now
    // zero-pads (`torch.stft(center=True, pad_mode="constant")`) instead of
    // reflecting, and emits `floor(T/hop)` frames. That makes mel frame 0 match
    // NeMo bit-for-bit, so pre-encode output frame 0 (the utterance boundary,
    // where the subsampling conv's left zero-padding meets the signal-start mel
    // frames) now agrees within int8 tolerance. Restore the plan's original
    // whole-tensor gate (<= 5% relative rms) and additionally keep the tighter
    // interior gate (frames 1.., <= 3%) as a regression check on the split.
    let full_rel = relative_rms(&ours.values, &reference.chunk0_pre_encode.values);
    let interior_rel = relative_rms(
        &ours.values[dim..],
        &reference.chunk0_pre_encode.values[dim..],
    );
    let frame0_rel = relative_rms(
        &ours.values[..dim],
        &reference.chunk0_pre_encode.values[..dim],
    );
    eprintln!(
        "pre_encode chunk0: full_rel_rms={full_rel:.5} interior_rel_rms={interior_rel:.5} frame0_rel_rms={frame0_rel:.5}"
    );
    assert_relative_rms_below(
        &ours.values[dim..],
        &reference.chunk0_pre_encode.values[dim..],
        0.03,
    );
    assert_relative_rms_below(&ours.values, &reference.chunk0_pre_encode.values, 0.05);
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn encoder_output_matches_nemo_within_int8_tolerance() {
    let _guard = serialize_mlx();
    let reference = reference();
    let artifact = Artifact::load(artifact_dir()).unwrap();
    let config = SortformerConfig::load(artifact_dir()).unwrap();
    let frames = MelFrontend::new(&config).extract(&fixture_audio());
    let encoder = Encoder::from_artifact(&artifact, &config).unwrap();
    let output = encoder.forward(&frames).unwrap();
    // Ours is [1, frames/8, d_model]; the reference is [1, d_model, frames/8].
    assert_eq!(output.shape[2], config.encoder_dim);
    assert_eq!(output.shape[1], reference.encoder_out.shape[2]);
    // Reference `first` walks channel-major [1, d_model, frames/8] order.
    let subsampled = output.shape[1];
    let first: Vec<f32> = (0..64)
        .map(|index| {
            let channel = index / subsampled;
            let frame = index % subsampled;
            output.values[frame * config.encoder_dim + channel]
        })
        .collect();
    assert_close(
        "encoder_out",
        &output.values,
        &first,
        &reference.encoder_out,
        0.05,
    );
}

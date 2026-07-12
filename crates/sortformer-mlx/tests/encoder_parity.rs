//! Numerical parity against captured NeMo Sortformer activations.
//!
//! `tests/fixtures/sortformer_reference.json` records full-context module
//! outputs for `hello-streaming.wav`. The checkpoint preprocessor uses
//! `normalize: "NA"`, so the encoder consumes *unnormalized* log-mel frames
//! (reference `features` mean is -10.18, clearly not per-feature normalized).

use std::path::PathBuf;

use nemotron_mlx::weights::Artifact;
use sortformer_mlx::audio::MelFrontend;
use sortformer_mlx::config::SortformerConfig;
use sortformer_mlx::model::Encoder;

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
    let reference = reference();
    let config = SortformerConfig::load(artifact_dir()).unwrap();
    // The checkpoint preprocessor does not normalize (`normalize: "NA"`).
    let frames = MelFrontend::new(&config).extract(&fixture_audio());
    // Reference layout is [1, n_mels, frames]; ours is [frames][n_mels].
    assert_eq!(reference.features.shape[1], config.n_mels);
    assert_eq!(reference.features.shape[2], frames.len());
    // Reference `first` walks mel bin 0 across time.
    let bin0: Vec<f32> = frames.iter().map(|frame| frame[0]).take(64).collect();
    let flat: Vec<f32> = frames.iter().flatten().copied().collect();
    assert_close("features", &flat, &bin0, &reference.features, 0.02);
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn encoder_output_matches_nemo_within_int8_tolerance() {
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

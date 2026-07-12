//! End-to-end parity of the full Sortformer diarization pipeline.
//!
//! `Diarizer::diarize` runs mel frontend -> encoder -> encoder projection ->
//! 18 post-LN Transformer layers -> sigmoid speaker head and must match the
//! per-frame speaker probabilities captured from NeMo under eval()/no_grad in
//! `tests/fixtures/sortformer_reference.json` (`probabilities_full`, 52x4).

use std::path::PathBuf;

use sortformer_mlx::model::Diarizer;

#[derive(serde::Deserialize)]
struct Reference {
    probabilities: Shape,
    probabilities_full: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Shape {
    shape: Vec<usize>,
}

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn full_pipeline_probabilities_match_nemo() {
    let model_dir = std::env::var_os("SORTFORMER_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set SORTFORMER_MLX_ARTIFACT");
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect();
    let reference: Reference = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_reference.json"
    )))
    .unwrap();

    let diarizer = Diarizer::from_artifact_dir(&model_dir).unwrap();
    let probabilities = diarizer.diarize(&samples).unwrap();

    let frames = reference.probabilities.shape[1];
    assert_eq!(probabilities.len(), frames);
    let mut maximum_error = 0.0f32;
    let mut total_error = 0.0f32;
    for (frame, actual) in probabilities.iter().enumerate() {
        for speaker in 0..4 {
            let expected = reference.probabilities_full[frame * 4 + speaker];
            let difference = (actual[speaker] - expected).abs();
            maximum_error = maximum_error.max(difference);
            total_error += difference;
        }
    }
    let mean_error = total_error / (frames * 4) as f32;
    assert!(maximum_error < 0.05, "max abs error {maximum_error}");
    assert!(mean_error < 0.01, "mean abs error {mean_error}");
}

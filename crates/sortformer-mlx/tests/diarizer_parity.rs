//! End-to-end parity of the full Sortformer diarization pipeline.
//!
//! `Diarizer::diarize` runs mel frontend -> encoder -> encoder projection ->
//! 18 post-LN Transformer layers -> sigmoid speaker head and must match the
//! per-frame speaker probabilities captured from NeMo under eval()/no_grad in
//! `tests/fixtures/sortformer_reference.json` (`probabilities_full`, 52x4).

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};

use sortformer_mlx::model::Diarizer;

/// MLX evaluates onto a process-global Metal command buffer that is not safe
/// for concurrent submission; two full-pipeline tests running on separate
/// threads abort with "A command encoder is already encoding to this command
/// buffer". Each MLX-driving test holds this lock so they run serially.
static MLX_PIPELINE: Mutex<()> = Mutex::new(());

fn serialize_mlx() -> MutexGuard<'static, ()> {
    MLX_PIPELINE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(serde::Deserialize)]
struct Reference {
    probabilities: Shape,
    probabilities_full: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct Shape {
    shape: Vec<usize>,
}

fn load_diarizer() -> Diarizer {
    let model_dir = std::env::var_os("SORTFORMER_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set SORTFORMER_MLX_ARTIFACT");
    Diarizer::from_artifact_dir(&model_dir).unwrap()
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

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn full_pipeline_probabilities_match_nemo() {
    let _guard = serialize_mlx();
    let samples = fixture_audio();
    let reference: Reference = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_reference.json"
    )))
    .unwrap();

    let diarizer = load_diarizer();
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

#[test]
#[ignore = "requires SORTFORMER_MLX_ARTIFACT"]
fn diarize_reports_wall_time() {
    let _guard = serialize_mlx();
    let diarizer = load_diarizer();
    let audio = fixture_audio();
    let started = std::time::Instant::now();
    diarizer.diarize(&audio).unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let rtf = seconds / (audio.len() as f64 / 16_000.0);
    eprintln!("offline diarize RTF = {rtf:.3}");
    assert!(rtf < 1.0, "offline diarize slower than real time: {rtf:.3}");
}

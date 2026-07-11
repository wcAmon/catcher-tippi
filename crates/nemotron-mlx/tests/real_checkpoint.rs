use std::collections::HashMap;
use std::path::PathBuf;

use nemotron_mlx::{
    model::{LanguagePrompt, StreamingEncoder, StreamingTranscriber, Tensor3},
    weights::Artifact,
};

#[derive(serde::Deserialize)]
struct EncoderReference {
    input_shape: [usize; 3],
    input: Vec<f32>,
    output_shape: [usize; 3],
    output: Vec<f32>,
    checkpoints: HashMap<String, CheckpointSummary>,
    second_input_shape: [usize; 3],
    second_input: Vec<f32>,
    second_output_shape: [usize; 3],
    second_output: Vec<f32>,
    second_checkpoints: HashMap<String, CheckpointSummary>,
}

#[derive(serde::Deserialize)]
struct CheckpointSummary {
    shape: Vec<usize>,
    mean: f32,
    rms: f32,
    first: Vec<f32>,
}

#[derive(serde::Deserialize)]
struct TranscriptionReference {
    token_ids: Vec<u32>,
    text: String,
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and the downloaded checkpoint"]
fn real_wav_matches_official_streaming_token_ids() {
    let path = std::env::var_os("NEMOTRON_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set NEMOTRON_MLX_ARTIFACT to a converted artifact directory");
    let artifact = Artifact::load(path).unwrap();
    let mut reader = hound::WavReader::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming.wav"
    ))
    .unwrap();
    assert_eq!(reader.spec().sample_rate, 16_000);
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.unwrap() as f32 / 32768.0)
        .collect::<Vec<_>>();
    let reference: TranscriptionReference = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hello-streaming-reference.json"
    )))
    .unwrap();
    let expected = reference
        .token_ids
        .into_iter()
        .filter(|token| *token != 13_087)
        .collect::<Vec<_>>();
    let mut transcriber = StreamingTranscriber::new(&artifact, "en-US", 3).unwrap();

    let one_shot = transcriber.transcribe_samples(&samples).unwrap();
    assert_eq!(one_shot, expected, "official text: {}", reference.text);

    transcriber.reset().unwrap();
    let block_sizes = [127, 1_024, 333, 4_096, 511, 2_048];
    let mut actual = Vec::new();
    let mut offset = 0;
    let mut block = 0;
    while offset < samples.len() {
        let end = (offset + block_sizes[block % block_sizes.len()]).min(samples.len());
        actual.extend(transcriber.push_samples(&samples[offset..end]).unwrap());
        offset = end;
        block += 1;
    }
    actual.extend(transcriber.finish().unwrap());

    assert_eq!(actual, expected, "official text: {}", reference.text);
}

#[test]
#[ignore = "requires NEMOTRON_MLX_ARTIFACT and the downloaded checkpoint"]
fn real_checkpoint_loads_and_encodes_the_first_streaming_chunk() {
    let path = std::env::var_os("NEMOTRON_MLX_ARTIFACT")
        .map(PathBuf::from)
        .expect("set NEMOTRON_MLX_ARTIFACT to a converted artifact directory");
    let artifact = Artifact::load(path).unwrap();
    let encoder = StreamingEncoder::from_artifact(&artifact).unwrap();
    let mut cache = encoder.new_cache();
    let reference: EncoderReference = serde_json::from_str(include_str!(
        "../../../tests/fixtures/encoder_reference.json"
    ))
    .unwrap();
    let trace = encoder
        .encode_chunk_trace(
            &Tensor3 {
                shape: reference.input_shape,
                values: reference.input.clone(),
            },
            LanguagePrompt::from_code("auto").unwrap(),
            &mut cache,
        )
        .unwrap();
    let output = trace.prompted.clone();

    report_checkpoint("subsampling", &trace.subsampling, &reference);
    for (index, tensor) in trace.layers.iter().enumerate() {
        report_checkpoint(&format!("layer_{index}"), tensor, &reference);
    }
    report_checkpoint("prompted", &trace.prompted, &reference);

    assert_eq!(output.shape, reference.output_shape);
    assert!(output.values.iter().all(|value| value.is_finite()));
    let mut maximum_error = 0.0_f32;
    let mut absolute_error = 0.0_f32;
    for (actual, expected) in output.values.iter().zip(&reference.output) {
        let error = (actual - expected).abs();
        maximum_error = maximum_error.max(error);
        absolute_error += error;
    }
    let mean_absolute_error = absolute_error / output.values.len() as f32;
    for (frame_index, (actual, expected)) in output
        .values
        .chunks_exact(1024)
        .zip(reference.output.chunks_exact(1024))
        .enumerate()
    {
        let frame_mae = actual
            .iter()
            .zip(expected)
            .map(|(left, right)| (left - right).abs())
            .sum::<f32>()
            / 1024.0;
        let frame_max = actual
            .iter()
            .zip(expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let actual_rms = (actual.iter().map(|value| value * value).sum::<f32>() / 1024.0).sqrt();
        let expected_rms =
            (expected.iter().map(|value| value * value).sum::<f32>() / 1024.0).sqrt();
        eprintln!(
            "prompted frame {frame_index}: MAE={frame_mae:.6} max={frame_max:.6} rms={actual_rms:.6}/{expected_rms:.6}"
        );
    }
    assert!(
        mean_absolute_error < 0.03,
        "encoder MAE {mean_absolute_error}, maximum error {maximum_error}"
    );
    assert!(
        maximum_error < 0.2,
        "encoder MAE {mean_absolute_error}, maximum error {maximum_error}"
    );

    let second_trace = encoder
        .encode_chunk_trace(
            &Tensor3 {
                shape: reference.second_input_shape,
                values: reference.second_input.clone(),
            },
            LanguagePrompt::from_code("auto").unwrap(),
            &mut cache,
        )
        .unwrap();
    report_checkpoint_from(
        "second subsampling",
        &second_trace.subsampling,
        &reference.second_checkpoints["subsampling"],
    );
    for (index, tensor) in second_trace.layers.iter().enumerate() {
        report_checkpoint_from(
            &format!("second layer_{index}"),
            tensor,
            &reference.second_checkpoints[&format!("layer_{index}")],
        );
    }
    let second = second_trace.prompted;
    assert_eq!(second.shape, reference.second_output_shape);
    // The persistent INT8 cache can amplify hidden-state drift on a synthetic
    // boundary frame. The real-WAV test above remains exact at the token level.
    assert_reference_error(
        "second chunk",
        &second.values,
        &reference.second_output,
        0.08,
        7.0,
    );
}

fn assert_reference_error(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    maximum_mae: f32,
    maximum_error: f32,
) {
    assert_eq!(actual.len(), expected.len());
    let errors = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .collect::<Vec<_>>();
    let mae = errors.iter().sum::<f32>() / errors.len() as f32;
    let max = errors.into_iter().fold(0.0_f32, f32::max);
    for (frame, (actual_frame, expected_frame)) in actual
        .chunks_exact(1024)
        .zip(expected.chunks_exact(1024))
        .enumerate()
    {
        let frame_mae = actual_frame
            .iter()
            .zip(expected_frame)
            .map(|(actual, expected)| (actual - expected).abs())
            .sum::<f32>()
            / 1024.0;
        eprintln!("{label} frame {frame} MAE={frame_mae:.6}");
    }
    assert!(mae < maximum_mae, "{label} MAE {mae}, maximum error {max}");
    assert!(
        max < maximum_error,
        "{label} MAE {mae}, maximum error {max}"
    );
}

fn report_checkpoint(name: &str, actual: &Tensor3, reference: &EncoderReference) {
    let expected = &reference.checkpoints[name];
    report_checkpoint_from(name, actual, expected);
}

fn report_checkpoint_from(name: &str, actual: &Tensor3, expected: &CheckpointSummary) {
    let mean = actual.values.iter().sum::<f32>() / actual.values.len() as f32;
    let rms = (actual.values.iter().map(|value| value * value).sum::<f32>()
        / actual.values.len() as f32)
        .sqrt();
    let first_mae = actual
        .values
        .iter()
        .zip(&expected.first)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / expected.first.len() as f32;
    eprintln!(
        "{name:>12} shape={:?}/{:?} mean={mean:+.5}/{:+.5} rms={rms:.5}/{:.5} first64_mae={first_mae:.5}",
        actual.shape, expected.shape, expected.mean, expected.rms,
    );
}

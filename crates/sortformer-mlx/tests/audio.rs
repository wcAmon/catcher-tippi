use sortformer_mlx::audio::MelFrontend;
use sortformer_mlx::config::SortformerConfig;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/sortformer_config.json"
));

#[test]
fn one_second_of_audio_yields_100_normalized_frames() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frontend = MelFrontend::new(&config);
    let audio: Vec<f32> = (0..16_000)
        .map(|index| (index as f32 * 0.02).sin() * 0.3)
        .collect();
    let frames = frontend.extract_normalized(&audio);
    assert!(
        (98..=101).contains(&frames.len()),
        "frames {}",
        frames.len()
    );
    assert!(frames.iter().all(|frame| frame.len() == config.n_mels));
    // Per-feature normalization: each mel bin is zero-mean unit-variance over time.
    for bin in 0..config.n_mels {
        let count = frames.len() as f32;
        let mean: f32 = frames.iter().map(|frame| frame[bin]).sum::<f32>() / count;
        let variance: f32 = frames
            .iter()
            .map(|frame| (frame[bin] - mean).powi(2))
            .sum::<f32>()
            / (count - 1.0);
        assert!(mean.abs() < 1e-3, "bin {bin} mean {mean}");
        assert!(
            (variance - 1.0).abs() < 2e-2,
            "bin {bin} variance {variance}"
        );
    }
}

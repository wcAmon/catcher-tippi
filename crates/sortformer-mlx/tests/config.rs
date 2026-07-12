use sortformer_mlx::config::SortformerConfig;

const FIXTURE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/sortformer_config.json"
));

#[test]
fn real_config_parses_with_expected_architecture() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    assert_eq!(config.sample_rate, 16_000);
    assert_eq!(config.n_mels, 128);
    assert_eq!(config.encoder_layers, 17);
    assert_eq!(config.encoder_dim, 512);
    assert_eq!(config.subsampling_factor, 8);
    assert_eq!(config.transformer_layers, 18);
    assert_eq!(config.transformer_dim, 192);
    assert_eq!(config.num_speakers, 4);
    assert!((config.hop_seconds - 0.01).abs() < 1e-9);
}

#[test]
fn output_frame_duration_is_80_ms() {
    let config = SortformerConfig::from_json(FIXTURE).unwrap();
    let frame_ms = config.hop_seconds * config.subsampling_factor as f64 * 1_000.0;
    assert!((frame_ms - 80.0).abs() < 1e-6);
}

use approx::assert_abs_diff_eq;
use nemotron_mlx::model::{EncoderConfig, LanguagePrompt, PromptProjector, StreamingChunkPlan};

static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn checkpoint_prompt_dictionary_maps_locales_exactly() {
    assert_eq!(LanguagePrompt::from_code("auto").unwrap().id(), 101);
    assert_eq!(LanguagePrompt::from_code("zh-CN").unwrap().id(), 4);
    assert_eq!(LanguagePrompt::from_code("zh-TW").unwrap().id(), 5);
    assert_eq!(LanguagePrompt::from_code("en").unwrap().id(), 0);
    assert_eq!(LanguagePrompt::from_code("en-GB").unwrap().id(), 1);
    assert_eq!(LanguagePrompt::from_code("nb-NO").unwrap().id(), 103);
    assert_eq!(LanguagePrompt::from_code("nn").unwrap().id(), 104);
    assert_eq!(LanguagePrompt::supported_codes().len(), 121);
    assert!(LanguagePrompt::from_code("zh-HK").is_err());
}

#[test]
fn checkpoint_streaming_chunk_shapes_are_exact() {
    let config = EncoderConfig::nemotron_3_5();
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.sliding_window, 57);
    assert_eq!(config.supported_lookahead, [3, 0, 6, 13]);

    let default = StreamingChunkPlan::new(&config, 3).unwrap();
    assert_eq!(default.first_mel_frames(), 25);
    assert_eq!(default.subsequent_mel_frames(), 32);
    assert_eq!(default.first_audio_samples(), 4040);
    assert_eq!(default.subsequent_audio_samples(), 5520);
    assert_eq!(default.encoder_frames_per_chunk(), 4);
    assert_eq!(default.emitted_frames_per_chunk(), 1);
    assert_eq!(default.latency_ms(), 320);

    let zero = StreamingChunkPlan::new(&config, 0).unwrap();
    assert_eq!(zero.first_mel_frames(), 1);
    assert_eq!(zero.subsequent_mel_frames(), 8);
    assert_eq!(zero.encoder_frames_per_chunk(), 1);
    assert_eq!(zero.latency_ms(), 80);
    assert!(StreamingChunkPlan::new(&config, 1).is_err());
}

#[test]
fn prompt_projector_broadcasts_one_hot_over_time() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let hidden = 128;
    let prompts = 128;
    let intermediate = 128;
    let mut weight1 = vec![0.0; intermediate * (hidden + prompts)];
    // Preserve hidden channel 0 and add the selected prompt bit into output channel 0.
    weight1[0] = 1.0;
    for prompt in 0..prompts {
        weight1[prompt * (hidden + prompts) + hidden + prompt] = 1.0;
    }
    let mut weight2 = vec![0.0; hidden * intermediate];
    for index in 0..hidden {
        weight2[index * intermediate + index] = 1.0;
    }
    let projector = PromptProjector::from_f32(
        &weight1,
        &[0.0; 128],
        &weight2,
        &[0.0; 128],
        hidden,
        prompts,
        intermediate,
        128,
    )
    .unwrap();
    let mut input = vec![0.0; 2 * hidden];
    input[0] = 2.0;
    input[hidden] = 3.0;

    let output = projector
        .forward_f32(&input, 1, 2, LanguagePrompt::from_code("zh-TW").unwrap())
        .unwrap();

    assert_eq!(output.shape, [1, 2, hidden]);
    assert_abs_diff_eq!(output.values[0], 2.0, epsilon = 0.04);
    assert_abs_diff_eq!(output.values[hidden], 3.0, epsilon = 0.04);
    assert_abs_diff_eq!(output.values[5], 1.0, epsilon = 0.04);
    assert_abs_diff_eq!(output.values[hidden + 5], 1.0, epsilon = 0.04);
}

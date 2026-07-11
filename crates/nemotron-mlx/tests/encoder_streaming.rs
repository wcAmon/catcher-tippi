use approx::assert_abs_diff_eq;
use nemotron_mlx::model::{
    AttentionKvCache, CausalConv2dCache, Conv2dSubsampling, EncoderConfig, Fp16Conv2d,
    LanguagePrompt, PromptProjector, QuantizedLinear, RelativePositionAttention,
    StreamingChunkPlan, SubsamplingCache, Tensor3, Tensor4, channel_frequency_flatten,
    chunked_attention_mask, relative_position_encoding, relative_shift,
};

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

#[test]
fn causal_conv2d_first_and_subsequent_chunks_use_exact_padding_cache() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let conv = Fp16Conv2d::from_f32(&[1.0; 9], &[0.0], 1, 1, 3, 2, 1).unwrap();
    let mut cache = CausalConv2dCache::new(4, 1, 1, 1);
    let first = Tensor4 {
        shape: [1, 3, 4, 1],
        values: (1..=12).map(|value| value as f32).collect(),
    };

    let first_output = conv.forward_causal(&first, &mut cache).unwrap();
    assert_eq!(first_output.shape, [1, 2, 3, 1]);
    assert_eq!(cache.values(), &[9.0, 10.0, 11.0, 12.0]);
    let expected_first = conv2d_reference(&first.values, 3, 4, true);
    for (actual, expected) in first_output.values.iter().zip(expected_first) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-3);
    }

    let second = Tensor4 {
        shape: [1, 2, 4, 1],
        values: (13..=20).map(|value| value as f32).collect(),
    };
    let second_output = conv.forward_causal(&second, &mut cache).unwrap();
    assert_eq!(second_output.shape, [1, 1, 3, 1]);
    assert_eq!(cache.values(), &[17.0, 18.0, 19.0, 20.0]);
    let mut cached_second = vec![9.0, 10.0, 11.0, 12.0];
    cached_second.extend_from_slice(&second.values);
    let expected_second = conv2d_reference(&cached_second, 3, 4, false);
    for (actual, expected) in second_output.values.iter().zip(expected_second) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-3);
    }
}

fn conv2d_reference(input: &[f32], time: usize, freq: usize, first: bool) -> Vec<f32> {
    assert_eq!(input.len(), time * freq);
    let mut temporal = if first {
        vec![0.0; 2 * freq]
    } else {
        Vec::new()
    };
    temporal.extend_from_slice(input);
    let padded_time = temporal.len() / freq;
    let out_time = (padded_time - 3) / 2 + 1;
    let out_freq = (freq + 3 - 3) / 2 + 1;
    let mut output = Vec::with_capacity(out_time * out_freq);
    for out_t in 0..out_time {
        for out_f in 0..out_freq {
            let mut sum = 0.0;
            for kernel_t in 0..3 {
                for kernel_f in 0..3 {
                    let source_f = out_f * 2 + kernel_f;
                    if (2..2 + freq).contains(&source_f) {
                        sum += temporal[(out_t * 2 + kernel_t) * freq + source_f - 2];
                    }
                }
            }
            output.push(sum);
        }
    }
    output
}

#[test]
fn factor_eight_subsampling_matches_one_shot_across_chunk_boundary() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let channels = 128;
    let (offline_model, mut offline_cache) = tiny_subsampling(channels);
    let (streaming_model, mut streaming_cache) = tiny_subsampling(channels);
    let input: Vec<f32> = (0..17 * 8)
        .map(|index| ((index % 23) as f32 - 11.0) / 17.0)
        .collect();
    let offline = offline_model
        .forward(
            &Tensor3 {
                shape: [1, 17, 8],
                values: input.clone(),
            },
            &mut offline_cache,
        )
        .unwrap();
    let first = streaming_model
        .forward(
            &Tensor3 {
                shape: [1, 9, 8],
                values: input[..9 * 8].to_vec(),
            },
            &mut streaming_cache,
        )
        .unwrap();
    let second = streaming_model
        .forward(
            &Tensor3 {
                shape: [1, 8, 8],
                values: input[9 * 8..].to_vec(),
            },
            &mut streaming_cache,
        )
        .unwrap();

    assert_eq!(offline.shape, [1, 3, channels]);
    assert_eq!(first.shape, [1, 2, channels]);
    assert_eq!(second.shape, [1, 1, channels]);
    assert!(
        first.values[channels..]
            .iter()
            .all(|value| value.abs() < 1.0e-6),
        "the extra ceil frame in the first chunk must be masked"
    );
    let mut streamed = first.values;
    streamed.extend_from_slice(&second.values);
    for (actual, expected) in streamed.iter().zip(offline.values) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 3.0e-3);
    }
}

fn tiny_subsampling(channels: usize) -> (Conv2dSubsampling, SubsamplingCache) {
    let stem = Fp16Conv2d::from_f32(
        &vec![0.01; channels * 3 * 3],
        &vec![0.0; channels],
        channels,
        1,
        3,
        2,
        1,
    )
    .unwrap();
    let mut stages = Vec::new();
    for _ in 0..2 {
        let depthwise = Fp16Conv2d::from_f32(
            &vec![0.01; channels * 3 * 3],
            &vec![0.0; channels],
            channels,
            channels,
            3,
            2,
            channels,
        )
        .unwrap();
        let pointwise = QuantizedLinear::from_f32(
            &identity(channels),
            channels,
            channels,
            &vec![0.0; channels],
            128,
        )
        .unwrap();
        stages.push((depthwise, pointwise));
    }
    let mut output_weight = vec![0.0; channels * channels * 2];
    for channel in 0..channels {
        output_weight[channel * channels * 2 + channel] = 0.5;
        output_weight[channel * channels * 2 + channels + channel] = 0.5;
    }
    let output = QuantizedLinear::from_f32(
        &output_weight,
        channels,
        channels * 2,
        &vec![0.0; channels],
        128,
    )
    .unwrap();
    (
        Conv2dSubsampling::new(stem, stages, output).unwrap(),
        SubsamplingCache::new(8, channels, 3, 2, 3),
    )
}

fn identity(dimensions: usize) -> Vec<f32> {
    let mut output = vec![0.0; dimensions * dimensions];
    for index in 0..dimensions {
        output[index * dimensions + index] = 1.0;
    }
    output
}

#[test]
fn attention_cache_returns_old_plus_current_and_retains_sliding_window() {
    let mut cache = AttentionKvCache::new(2, 2, 3);
    let first_keys: Vec<f32> = (0..8).map(|value| value as f32).collect();
    let first_values: Vec<f32> = (100..108).map(|value| value as f32).collect();
    let first = cache.update(&first_keys, &first_values, 2).unwrap();
    assert_eq!(first.frames, 2);
    assert_eq!(first.keys, first_keys);
    assert_eq!(cache.frames(), 2);

    let second_keys: Vec<f32> = (8..16).map(|value| value as f32).collect();
    let second_values: Vec<f32> = (108..116).map(|value| value as f32).collect();
    let second = cache.update(&second_keys, &second_values, 2).unwrap();
    assert_eq!(second.frames, 4);
    assert_eq!(second.keys.len(), 16);
    assert_eq!(cache.frames(), 3);
    // Per-head cache keeps global frames 1, 2, 3 rather than slicing flat storage.
    assert_eq!(
        cache.keys(),
        &[
            2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0
        ]
    );
}

#[test]
fn chunk_mask_allows_current_lookahead_and_bounded_previous_chunks() {
    let mask = chunked_attention_mask(12, 4, 3);
    let allowed = |query: usize, key: usize| mask[query * 12 + key];

    assert!(allowed(0, 3));
    assert!(!allowed(0, 4));
    assert!(allowed(4, 0));
    assert!(allowed(4, 7));
    assert!(!allowed(4, 8));
    assert!(!allowed(8, 3));
    assert!(allowed(8, 4));
    assert!(allowed(8, 11));
}

#[test]
fn relative_positions_interleave_sine_and_cosine_in_transformers_order() {
    let positions = relative_position_encoding(4, 2).unwrap();
    let expected = [
        1.0_f32.sin(),
        1.0_f32.cos(),
        0.01_f32.sin(),
        0.01_f32.cos(),
        0.0,
        1.0,
        0.0,
        1.0,
        -1.0_f32.sin(),
        1.0_f32.cos(),
        -0.01_f32.sin(),
        0.01_f32.cos(),
    ];
    assert_eq!(positions.shape, [1, 3, 4]);
    for (actual, expected) in positions.values.iter().zip(expected) {
        assert_abs_diff_eq!(*actual, expected, epsilon = 1.0e-6);
    }
}

#[test]
fn relative_shift_matches_transformers_pad_view_slice_sequence() {
    let scores = (0..10).map(|value| value as f32).collect::<Vec<_>>();
    let shifted = relative_shift(&scores, 2, 5).unwrap();
    assert_eq!(
        shifted,
        vec![1.0, 2.0, 3.0, 4.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn relative_attention_averages_values_when_content_and_position_scores_are_zero() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let hidden = 128;
    let constant = vec![0.01; hidden * hidden];
    let attention = RelativePositionAttention::from_f32(
        &constant,
        &constant,
        &identity(hidden),
        &identity(hidden),
        &identity(hidden),
        &[0.0; 128],
        &[0.0; 128],
        hidden,
        2,
        128,
    )
    .unwrap();
    let mut cache = AttentionKvCache::new(2, 64, 3);
    let first = attention
        .forward_streaming(
            &Tensor3 {
                shape: [1, 2, hidden],
                values: zero_sum_frames(hidden, &[1.0, 2.0]),
            },
            &mut cache,
        )
        .unwrap();
    assert_eq!(first.shape, [1, 2, hidden]);
    for frame in first.values.chunks_exact(hidden) {
        assert_abs_diff_eq!(frame[0], 1.5, epsilon = 0.03);
        assert_abs_diff_eq!(frame[hidden - 1], -1.5, epsilon = 0.03);
    }

    let second = attention
        .forward_streaming(
            &Tensor3 {
                shape: [1, 1, hidden],
                values: zero_sum_frames(hidden, &[3.0]),
            },
            &mut cache,
        )
        .unwrap();
    assert_eq!(cache.frames(), 3);
    assert_abs_diff_eq!(second.values[0], 2.0, epsilon = 0.03);
    assert_abs_diff_eq!(second.values[hidden - 1], -2.0, epsilon = 0.03);
}

fn zero_sum_frames(hidden: usize, magnitudes: &[f32]) -> Vec<f32> {
    let mut output = Vec::with_capacity(hidden * magnitudes.len());
    for magnitude in magnitudes {
        output.extend(std::iter::repeat_n(*magnitude, hidden / 2));
        output.extend(std::iter::repeat_n(-*magnitude, hidden / 2));
    }
    output
}

#[test]
fn subsampling_flattens_channels_before_frequency_for_pytorch_linear() {
    let input = Tensor4 {
        shape: [1, 1, 2, 3],
        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    };
    assert_eq!(
        channel_frequency_flatten(&input).unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

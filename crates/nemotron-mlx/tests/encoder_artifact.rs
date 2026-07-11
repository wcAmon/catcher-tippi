use std::collections::HashMap;

use mlx_rs::Array;
use nemotron_mlx::{
    model::{
        AttentionKvCache, Conv2dSubsampling, EncoderConfig, FastConformerLayer, LanguagePrompt,
        RelativePositionAttention, StreamingEncoder, SubsamplingCache, Tensor3,
    },
    weights::{Artifact, Storage, TensorSpec, TensorTransform, convert_tensors},
};

static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn subsampling_binds_to_converted_artifact_without_requantizing() {
    let _guard = MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.safetensors");
    let destination = temp.path().join("artifact");
    let channels = 128;
    let mut arrays = HashMap::new();
    let mut specs = Vec::new();

    add_f16(
        &mut arrays,
        &mut specs,
        "encoder.subsampling.conv_in.weight",
        &[channels, 1, 3, 3],
        0.01,
    );
    add_f16(
        &mut arrays,
        &mut specs,
        "encoder.subsampling.conv_in.bias",
        &[channels],
        0.0,
    );
    add_int8_matrix(
        &mut arrays,
        &mut specs,
        "prompt_projector.linear_1.weight",
        channels,
        channels + 128,
    );
    add_f16(
        &mut arrays,
        &mut specs,
        "prompt_projector.linear_1.bias",
        &[channels],
        0.0,
    );
    add_int8_matrix(
        &mut arrays,
        &mut specs,
        "prompt_projector.linear_2.weight",
        channels,
        channels,
    );
    add_f16(
        &mut arrays,
        &mut specs,
        "prompt_projector.linear_2.bias",
        &[channels],
        0.0,
    );
    let layer_prefix = "encoder.layers.0";
    for feed_forward in ["feed_forward1", "feed_forward2"] {
        add_int8_matrix(
            &mut arrays,
            &mut specs,
            &format!("{layer_prefix}.{feed_forward}.linear1.weight"),
            channels,
            channels,
        );
        add_int8_matrix(
            &mut arrays,
            &mut specs,
            &format!("{layer_prefix}.{feed_forward}.linear2.weight"),
            channels,
            channels,
        );
    }
    for norm in [
        "norm_feed_forward1",
        "norm_self_att",
        "norm_conv",
        "norm_feed_forward2",
        "norm_out",
        "conv.norm",
    ] {
        add_f16(
            &mut arrays,
            &mut specs,
            &format!("{layer_prefix}.{norm}.weight"),
            &[channels],
            1.0,
        );
        add_f16(
            &mut arrays,
            &mut specs,
            &format!("{layer_prefix}.{norm}.bias"),
            &[channels],
            0.0,
        );
    }
    add_int8_pointwise_1d(
        &mut arrays,
        &mut specs,
        &format!("{layer_prefix}.conv.pointwise_conv1.weight"),
        2 * channels,
        channels,
    );
    add_int8_pointwise_1d(
        &mut arrays,
        &mut specs,
        &format!("{layer_prefix}.conv.pointwise_conv2.weight"),
        channels,
        channels,
    );
    add_f16(
        &mut arrays,
        &mut specs,
        &format!("{layer_prefix}.conv.depthwise_conv.weight"),
        &[channels, 1, 9],
        0.01,
    );
    let attention_prefix = "encoder.layers.0.self_attn";
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"] {
        add_int8_matrix(
            &mut arrays,
            &mut specs,
            &format!("{attention_prefix}.{projection}.weight"),
            channels,
            channels,
        );
    }
    add_f16(
        &mut arrays,
        &mut specs,
        &format!("{attention_prefix}.bias_u"),
        &[2, 64],
        0.0,
    );
    add_f16(
        &mut arrays,
        &mut specs,
        &format!("{attention_prefix}.bias_v"),
        &[2, 64],
        0.0,
    );
    for layer in 0..2 {
        add_f16(
            &mut arrays,
            &mut specs,
            &format!("encoder.subsampling.layers.{layer}.depthwise_conv.weight"),
            &[channels, 1, 3, 3],
            0.01,
        );
        add_f16(
            &mut arrays,
            &mut specs,
            &format!("encoder.subsampling.layers.{layer}.depthwise_conv.bias"),
            &[channels],
            0.0,
        );
        add_int8_pointwise(
            &mut arrays,
            &mut specs,
            &format!("encoder.subsampling.layers.{layer}.pointwise_conv.weight"),
            channels,
        );
        add_f16(
            &mut arrays,
            &mut specs,
            &format!("encoder.subsampling.layers.{layer}.pointwise_conv.bias"),
            &[channels],
            0.0,
        );
    }
    let output_name = "encoder.subsampling.linear.weight";
    arrays.insert(
        output_name.to_string(),
        Array::from_slice(
            &vec![0.01_f32; channels * channels * 2],
            &[channels as i32, (channels * 2) as i32],
        ),
    );
    specs.push(TensorSpec {
        name: output_name.to_string(),
        source_shape: vec![channels, channels * 2],
        artifact_shape: vec![channels, channels * 2],
        storage: Storage::Int8Affine { group_size: 128 },
        transform: TensorTransform::Identity,
    });
    add_f16(
        &mut arrays,
        &mut specs,
        "encoder.subsampling.linear.bias",
        &[channels],
        0.0,
    );
    Array::save_safetensors(&arrays, None, &source).unwrap();
    convert_tensors(&source, &destination, "fixture/subsampling", &specs).unwrap();
    let artifact = Artifact::load(&destination).unwrap();
    let model = Conv2dSubsampling::from_artifact(&artifact).unwrap();
    let mut cache = SubsamplingCache::new(8, channels, 3, 2, 3);

    let output = model
        .forward(
            &Tensor3 {
                shape: [1, 9, 8],
                values: vec![0.25; 9 * 8],
            },
            &mut cache,
        )
        .unwrap();
    assert_eq!(output.shape, [1, 2, channels]);

    let attention = RelativePositionAttention::from_artifact(&artifact, 0, 2).unwrap();
    let mut attention_cache = AttentionKvCache::new(2, 64, 3);
    let attended = attention
        .forward_streaming(
            &Tensor3 {
                shape: [1, 2, channels],
                values: vec![0.25; 2 * channels],
            },
            &mut attention_cache,
        )
        .unwrap();
    assert_eq!(attended.shape, [1, 2, channels]);

    let layer = FastConformerLayer::from_artifact(&artifact, 0, 2, 3).unwrap();
    let mut layer_cache = layer.new_cache();
    let encoded = layer
        .forward(
            &Tensor3 {
                shape: [1, 2, channels],
                values: vec![0.25; 2 * channels],
            },
            &mut layer_cache,
        )
        .unwrap();
    assert_eq!(encoded.shape, [1, 2, channels]);
    assert_eq!(layer_cache.attention_frames(), 2);

    let config = EncoderConfig {
        hidden_size: channels,
        intermediate_size: channels,
        num_layers: 1,
        num_heads: 2,
        conv_kernel_size: 9,
        subsampling_factor: 8,
        sliding_window: 4,
        supported_lookahead: [3, 0, 6, 13],
        default_lookahead: 3,
    };
    let encoder =
        StreamingEncoder::from_artifact_with_config(&artifact, config, 8, channels).unwrap();
    let mut encoder_cache = encoder.new_cache();
    let encoded = encoder
        .encode_chunk(
            &Tensor3 {
                shape: [1, 9, 8],
                values: vec![0.25; 9 * 8],
            },
            LanguagePrompt::from_code("auto").unwrap(),
            &mut encoder_cache,
        )
        .unwrap();
    assert_eq!(encoded.shape, [1, 2, channels]);
}

fn add_f16(
    arrays: &mut HashMap<String, Array>,
    specs: &mut Vec<TensorSpec>,
    name: &str,
    shape: &[usize],
    value: f32,
) {
    let mlx_shape = shape
        .iter()
        .map(|dimension| *dimension as i32)
        .collect::<Vec<_>>();
    arrays.insert(
        name.to_string(),
        Array::from_slice(&vec![value; shape.iter().product()], &mlx_shape),
    );
    specs.push(TensorSpec {
        name: name.to_string(),
        source_shape: shape.to_vec(),
        artifact_shape: shape.to_vec(),
        storage: Storage::F16,
        transform: TensorTransform::Identity,
    });
}

fn add_int8_pointwise(
    arrays: &mut HashMap<String, Array>,
    specs: &mut Vec<TensorSpec>,
    name: &str,
    channels: usize,
) {
    let mut values = vec![0.0_f32; channels * channels];
    for index in 0..channels {
        values[index * channels + index] = 1.0;
    }
    arrays.insert(
        name.to_string(),
        Array::from_slice(&values, &[channels as i32, channels as i32, 1, 1]),
    );
    specs.push(TensorSpec {
        name: name.to_string(),
        source_shape: vec![channels, channels, 1, 1],
        artifact_shape: vec![channels, channels],
        storage: Storage::Int8Affine { group_size: 128 },
        transform: TensorTransform::SqueezeTrailingUnitDimensions,
    });
}

fn add_int8_matrix(
    arrays: &mut HashMap<String, Array>,
    specs: &mut Vec<TensorSpec>,
    name: &str,
    output: usize,
    input: usize,
) {
    let mut values = vec![0.0_f32; output * input];
    for index in 0..output.min(input) {
        values[index * input + index] = 1.0;
    }
    arrays.insert(
        name.to_string(),
        Array::from_slice(&values, &[output as i32, input as i32]),
    );
    specs.push(TensorSpec {
        name: name.to_string(),
        source_shape: vec![output, input],
        artifact_shape: vec![output, input],
        storage: Storage::Int8Affine { group_size: 128 },
        transform: TensorTransform::Identity,
    });
}

fn add_int8_pointwise_1d(
    arrays: &mut HashMap<String, Array>,
    specs: &mut Vec<TensorSpec>,
    name: &str,
    output: usize,
    input: usize,
) {
    let mut values = vec![0.0_f32; output * input];
    for index in 0..output.min(input) {
        values[index * input + index] = 1.0;
    }
    arrays.insert(
        name.to_string(),
        Array::from_slice(&values, &[output as i32, input as i32, 1]),
    );
    specs.push(TensorSpec {
        name: name.to_string(),
        source_shape: vec![output, input, 1],
        artifact_shape: vec![output, input],
        storage: Storage::Int8Affine { group_size: 128 },
        transform: TensorTransform::SqueezeTrailingUnitDimensions,
    });
}

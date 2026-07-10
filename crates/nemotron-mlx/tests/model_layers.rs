use approx::assert_abs_diff_eq;
use nemotron_mlx::model::{
    CausalConv1dCache, DepthwiseConv1d, LayerNorm, PointwiseConv1d, QuantizedLinear,
};

static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn lock_mlx() -> std::sync::MutexGuard<'static, ()> {
    MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn linear_reference(
    input: &[f32],
    rows: usize,
    input_dims: usize,
    weight: &[f32],
    output_dims: usize,
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0; rows * output_dims];
    for row in 0..rows {
        for out in 0..output_dims {
            let mut sum = bias[out];
            for col in 0..input_dims {
                sum += input[row * input_dims + col] * weight[out * input_dims + col];
            }
            output[row * output_dims + out] = sum;
        }
    }
    output
}

#[test]
fn quantized_linear_runs_on_mlx_with_bias() {
    let _guard = lock_mlx();
    let input_dims = 128;
    let output_dims = 3;
    let input: Vec<f32> = (0..256)
        .map(|index| ((index % 19) as f32 - 9.0) / 11.0)
        .collect();
    let weight: Vec<f32> = (0..output_dims * input_dims)
        .map(|index| ((index * 5 % 31) as f32 - 15.0) / 23.0)
        .collect();
    let bias = vec![0.25, -0.5, 0.125];
    let layer = QuantizedLinear::from_f32(&weight, output_dims, input_dims, &bias, 128).unwrap();

    let actual = layer.forward_f32(&input, 2).unwrap();
    let expected = linear_reference(&input, 2, input_dims, &weight, output_dims, &bias);

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 0.08);
    }
}

#[test]
fn pointwise_conv_preserves_batch_and_time_axes() {
    let _guard = lock_mlx();
    let input_dims = 128;
    let output_dims = 2;
    let input: Vec<f32> = (0..3 * input_dims)
        .map(|index| ((index % 13) as f32 - 6.0) / 8.0)
        .collect();
    let weight: Vec<f32> = (0..output_dims * input_dims)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 13.0)
        .collect();
    let bias = vec![0.1, -0.2];
    let layer = PointwiseConv1d::from_f32(&weight, output_dims, input_dims, &bias, 128).unwrap();

    let actual = layer.forward_f32(&input, 1, 3).unwrap();
    let expected = linear_reference(&input, 3, input_dims, &weight, output_dims, &bias);

    assert_eq!(actual.shape, [1, 3, 2]);
    for (actual, expected) in actual.values.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 0.08);
    }
}

#[test]
fn causal_cache_prepends_previous_frames_and_updates_itself() {
    let mut cache = CausalConv1dCache::new(2, 2);

    let first = cache.prepend_and_update(&[1.0, 10.0], 1).unwrap();
    assert_eq!(first, vec![0.0, 0.0, 0.0, 0.0, 1.0, 10.0]);
    assert_eq!(cache.values(), &[0.0, 0.0, 1.0, 10.0]);

    let second = cache
        .prepend_and_update(&[2.0, 20.0, 3.0, 30.0], 2)
        .unwrap();
    assert_eq!(second, vec![0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    assert_eq!(cache.values(), &[2.0, 20.0, 3.0, 30.0]);
}

#[test]
fn depthwise_conv_matches_causal_cross_correlation() {
    let _guard = lock_mlx();
    let input = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
    let weight = vec![1.0, 0.0, -1.0, 0.5, 0.25, 0.0];
    let bias = vec![0.5, -1.0];
    let layer = DepthwiseConv1d::from_f32(&weight, 2, 3, &bias).unwrap();
    let mut cache = CausalConv1dCache::new(2, 2);

    let actual = layer.forward_f32(&input, 1, 4, &mut cache).unwrap();
    let expected = [-0.5, -1.0, -1.5, 1.5, -1.5, 9.0, -1.5, 16.5];

    assert_eq!(actual.shape, [1, 4, 2]);
    for (actual, expected) in actual.values.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 2.0e-3);
    }
}

#[test]
fn layer_norm_matches_cpu_reference() {
    let _guard = lock_mlx();
    let input = vec![1.0, 2.0, 4.0, -1.0, 0.0, 3.0];
    let layer = LayerNorm::from_f32(&[1.0, 0.5, 2.0], &[0.0, 1.0, -1.0], 1.0e-5).unwrap();
    let actual = layer.forward_f32(&input, 2).unwrap();

    let mut expected = Vec::new();
    for row in input.chunks_exact(3) {
        let mean = row.iter().sum::<f32>() / 3.0;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / 3.0;
        for (index, value) in row.iter().enumerate() {
            expected.push(
                (value - mean) / (variance + 1.0e-5).sqrt() * [1.0, 0.5, 2.0][index]
                    + [0.0, 1.0, -1.0][index],
            );
        }
    }

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        // Runtime normalization intentionally uses FP16 inputs and affine parameters.
        assert_abs_diff_eq!(actual, expected, epsilon = 1.0e-3);
    }
}

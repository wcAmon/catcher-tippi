use approx::assert_abs_diff_eq;
use nemotron_mlx::backend::{is_metal_available, quantized_matmul};

fn fp32_reference(
    input: &[f32],
    rows: usize,
    input_dims: usize,
    weight: &[f32],
    output_dims: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; rows * output_dims];
    for row in 0..rows {
        for out in 0..output_dims {
            let mut sum = 0.0;
            for col in 0..input_dims {
                sum += input[row * input_dims + col] * weight[out * input_dims + col];
            }
            output[row * output_dims + out] = sum;
        }
    }
    output
}

#[test]
fn mlx_metal_int8_matmul_tracks_fp32_reference() {
    const ROWS: usize = 2;
    const INPUT_DIMS: usize = 128;
    const OUTPUT_DIMS: usize = 64;

    let input: Vec<f32> = (0..ROWS * INPUT_DIMS)
        .map(|index| ((index as i32 % 17) - 8) as f32 / 16.0)
        .collect();
    let weight: Vec<f32> = (0..OUTPUT_DIMS * INPUT_DIMS)
        .map(|index| ((index as i32 * 7 % 23) - 11) as f32 / 32.0)
        .collect();

    assert!(is_metal_available().expect("query MLX Metal device"));

    let actual = quantized_matmul(&input, ROWS, INPUT_DIMS, &weight, OUTPUT_DIMS, 128)
        .expect("run MLX INT8 matmul");
    let expected = fp32_reference(&input, ROWS, INPUT_DIMS, &weight, OUTPUT_DIMS);

    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 0.06);
    }
}

#[test]
fn quantized_matmul_rejects_mismatched_input_shape() {
    let result = quantized_matmul(&[0.0; 127], 1, 128, &[0.0; 128], 1, 128);
    let error = result.expect_err("mismatched input must be rejected");
    assert!(error.to_string().contains("input has 127 values"));
}

#[test]
fn quantized_matmul_rejects_mismatched_weight_shape() {
    let result = quantized_matmul(&[0.0; 128], 1, 128, &[0.0; 127], 1, 128);
    let error = result.expect_err("mismatched weight must be rejected");
    assert!(error.to_string().contains("weight has 127 values"));
}

#[test]
fn quantized_matmul_rejects_non_divisible_group_size() {
    let result = quantized_matmul(&[0.0; 128], 1, 128, &[0.0; 128], 1, 96);
    let error = result.expect_err("non-divisible group size must be rejected");
    assert!(error.to_string().contains("not divisible by group size 96"));
}

#[test]
fn quantized_matmul_rejects_zero_group_size() {
    let result = quantized_matmul(&[0.0; 128], 1, 128, &[0.0; 128], 1, 0);
    let error = result.expect_err("zero group size must be rejected");
    assert!(error.to_string().contains("group size must be positive"));
}

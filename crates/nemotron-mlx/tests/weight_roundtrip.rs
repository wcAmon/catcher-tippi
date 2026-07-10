use std::collections::BTreeMap;

use approx::assert_abs_diff_eq;
use bytemuck::cast_slice;
use nemotron_mlx::weights::{
    Artifact, DType, Storage, TensorSpec, TensorTransform, convert_model, convert_tensors,
};
use safetensors::tensor::{Dtype as SafeDType, TensorView, serialize_to_file};

fn write_source(path: &std::path::Path, matrix: &[f32], depthwise: &[f32]) {
    let matrix = TensorView::new(SafeDType::F32, vec![2, 128], cast_slice(matrix)).unwrap();
    let depthwise = TensorView::new(SafeDType::F32, vec![2, 1, 3], cast_slice(depthwise)).unwrap();
    let tensors = BTreeMap::from([
        ("matrix.weight".to_string(), matrix),
        ("depthwise.weight".to_string(), depthwise),
    ]);
    serialize_to_file(tensors, None, path).unwrap();
}

fn specs() -> Vec<TensorSpec> {
    vec![
        TensorSpec {
            name: "matrix.weight".to_string(),
            source_shape: vec![2, 128],
            artifact_shape: vec![2, 128],
            storage: Storage::Int8Affine { group_size: 128 },
            transform: TensorTransform::Identity,
        },
        TensorSpec {
            name: "depthwise.weight".to_string(),
            source_shape: vec![2, 1, 3],
            artifact_shape: vec![2, 1, 3],
            storage: Storage::F16,
            transform: TensorTransform::Identity,
        },
    ]
}

#[test]
fn converts_and_loads_an_mlx_int8_artifact() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source.safetensors");
    let artifact_path = temp.path().join("artifact");
    let matrix: Vec<f32> = (0..256)
        .map(|index| ((index % 29) - 14) as f32 / 17.0)
        .collect();
    let depthwise = vec![-0.75, 0.25, 1.0, 0.5, -0.25, 0.125];
    write_source(&source_path, &matrix, &depthwise);

    convert_tensors(&source_path, &artifact_path, "fixture/model", &specs()).unwrap();
    let artifact = Artifact::load(&artifact_path).unwrap();

    let matrix_info = artifact.tensor_info("matrix.weight").unwrap();
    assert_eq!(matrix_info.storage, Storage::Int8Affine { group_size: 128 });
    assert_eq!(matrix_info.shape, vec![2, 128]);
    assert_eq!(
        artifact.array_dtype("matrix.weight.__qweight").unwrap(),
        DType::U32
    );
    assert_eq!(
        artifact.array_dtype("matrix.weight.__scales").unwrap(),
        DType::F16
    );
    assert_eq!(
        artifact.array_dtype("matrix.weight.__biases").unwrap(),
        DType::F16
    );

    let reconstructed = artifact.dequantize_to_f32("matrix.weight").unwrap();
    assert_eq!(reconstructed.len(), matrix.len());
    for (actual, expected) in reconstructed.iter().zip(matrix.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 0.008);
    }

    let loaded_depthwise = artifact.f16_to_f32("depthwise.weight").unwrap();
    for (actual, expected) in loaded_depthwise.iter().zip(depthwise.iter()) {
        assert_abs_diff_eq!(actual, expected, epsilon = 0.001);
    }
}

#[test]
fn rejects_a_source_tensor_with_the_wrong_shape() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source.safetensors");
    let artifact_path = temp.path().join("artifact");
    write_source(&source_path, &[0.0; 256], &[0.0; 6]);
    let mut invalid_specs = specs();
    invalid_specs[0].source_shape = vec![1, 128];

    let error = convert_tensors(
        &source_path,
        &artifact_path,
        "fixture/model",
        &invalid_specs,
    )
    .unwrap_err();
    assert!(error.to_string().contains("matrix.weight shape mismatch"));
    assert!(!artifact_path.exists());
}

#[test]
fn full_model_conversion_requires_the_published_checkpoint_layout() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source.safetensors");
    let artifact_path = temp.path().join("artifact");
    write_source(&source_path, &[0.0; 256], &[0.0; 6]);

    let error = convert_model(&source_path, &artifact_path).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("missing source tensor encoder.layers.0.conv.depthwise_conv.weight")
    );
    assert!(!artifact_path.exists());
}

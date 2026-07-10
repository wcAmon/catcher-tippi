use std::collections::BTreeMap;

use nemotron_mlx::weights::{
    DType, ManifestError, ModelManifest, Storage, TensorIndex, TensorMetadata, TensorTransform,
};

fn valid_index(manifest: &ModelManifest) -> TensorIndex {
    manifest
        .tensors()
        .iter()
        .map(|spec| {
            (
                spec.name.clone(),
                TensorMetadata {
                    shape: spec.source_shape.clone(),
                    dtype: DType::F32,
                },
            )
        })
        .collect::<BTreeMap<_, _>>()
}

#[test]
fn manifest_matches_published_checkpoint_dimensions() {
    let manifest = ModelManifest::nemotron_3_5();

    assert_eq!(
        manifest.model_id(),
        "nvidia/nemotron-3.5-asr-streaming-0.6b"
    );
    assert_eq!(manifest.tensors().len(), 655);
    assert_eq!(manifest.parameter_count(), 637_997_088);
    assert_eq!(manifest.encoder_layers(), 24);
    assert_eq!(manifest.vocab_size(), 13_088);
}

#[test]
fn manifest_assigns_int8_only_to_matrix_compatible_weights() {
    let manifest = ModelManifest::nemotron_3_5();

    let feed_forward = manifest
        .tensor("encoder.layers.0.feed_forward1.linear1.weight")
        .unwrap();
    assert_eq!(
        feed_forward.storage,
        Storage::Int8Affine { group_size: 128 }
    );
    assert_eq!(feed_forward.transform, TensorTransform::Identity);
    assert_eq!(feed_forward.artifact_shape, vec![4096, 1024]);

    let pointwise = manifest
        .tensor("encoder.layers.0.conv.pointwise_conv1.weight")
        .unwrap();
    assert_eq!(pointwise.storage, Storage::Int8Affine { group_size: 128 });
    assert_eq!(
        pointwise.transform,
        TensorTransform::SqueezeTrailingUnitDimensions
    );
    assert_eq!(pointwise.artifact_shape, vec![2048, 1024]);

    let depthwise = manifest
        .tensor("encoder.layers.0.conv.depthwise_conv.weight")
        .unwrap();
    assert_eq!(depthwise.storage, Storage::F16);
    assert_eq!(depthwise.artifact_shape, vec![1024, 1, 9]);

    let norm = manifest.tensor("encoder.layers.0.norm_out.weight").unwrap();
    assert_eq!(norm.storage, Storage::F16);
}

#[test]
fn manifest_rejects_a_missing_tensor() {
    let manifest = ModelManifest::nemotron_3_5();
    let mut index = valid_index(&manifest);
    index.remove("joint.head.weight");

    let error = manifest.validate(&index).unwrap_err();
    assert_eq!(
        error,
        ManifestError::MissingTensor("joint.head.weight".to_string())
    );
}

#[test]
fn manifest_rejects_an_incorrect_shape() {
    let manifest = ModelManifest::nemotron_3_5();
    let mut index = valid_index(&manifest);
    index.get_mut("decoder.embedding.weight").unwrap().shape = vec![13_088, 639];

    let error = manifest.validate(&index).unwrap_err();
    assert_eq!(
        error,
        ManifestError::ShapeMismatch {
            name: "decoder.embedding.weight".to_string(),
            expected: vec![13_088, 640],
            actual: vec![13_088, 639],
        }
    );
}

#[test]
fn manifest_rejects_a_non_f32_source_tensor() {
    let manifest = ModelManifest::nemotron_3_5();
    let mut index = valid_index(&manifest);
    index.get_mut("joint.head.bias").unwrap().dtype = DType::F16;

    let error = manifest.validate(&index).unwrap_err();
    assert_eq!(
        error,
        ManifestError::DTypeMismatch {
            name: "joint.head.bias".to_string(),
            expected: DType::F32,
            actual: DType::F16,
        }
    );
}

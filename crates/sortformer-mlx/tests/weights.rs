use std::collections::BTreeMap;

use nemotron_mlx::weights::{ArtifactError, Storage, TensorTransform};
use sortformer_mlx::weights::{convert_model, specs_from_inventory};

fn inventory(entries: &[(&str, &[usize])]) -> BTreeMap<String, Vec<usize>> {
    entries
        .iter()
        .map(|(name, shape)| (name.to_string(), shape.to_vec()))
        .collect()
}

#[test]
fn large_matrices_quantize_to_int8_group_128() {
    let specs = specs_from_inventory(&inventory(&[(
        "encoder.layers.0.self_attn.linear_q.weight",
        &[512, 512],
    )]));
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].storage, Storage::Int8Affine { group_size: 128 });
    assert_eq!(specs[0].transform, TensorTransform::Identity);
    assert_eq!(specs[0].artifact_shape, vec![512, 512]);
}

#[test]
fn pointwise_convolutions_squeeze_then_quantize() {
    let specs = specs_from_inventory(&inventory(&[(
        "encoder.layers.0.conv.pointwise_conv1.weight",
        &[1024, 512, 1],
    )]));
    assert_eq!(specs[0].storage, Storage::Int8Affine { group_size: 128 });
    assert_eq!(
        specs[0].transform,
        TensorTransform::SqueezeTrailingUnitDimensions
    );
    assert_eq!(specs[0].artifact_shape, vec![1024, 512]);
}

#[test]
fn narrow_and_odd_tensors_stay_f16() {
    let specs = specs_from_inventory(&inventory(&[
        (
            "transformer_encoder.layers.0.first_sub_layer.query_net.weight",
            &[192, 192],
        ),
        ("encoder.layers.0.conv.depthwise_conv.weight", &[512, 1, 9]),
        ("encoder.layers.0.self_attn.pos_bias_u", &[8, 64]),
        ("encoder.layers.0.norm_out.bias", &[512]),
        ("encoder.pre_encode.conv.0.weight", &[256, 1, 3, 3]),
    ]));
    assert_eq!(specs.len(), 5);
    for spec in &specs {
        assert_eq!(spec.storage, Storage::F16, "tensor {}", spec.name);
        assert_eq!(
            spec.transform,
            TensorTransform::Identity,
            "tensor {}",
            spec.name
        );
    }
}

#[test]
fn real_inventory_produces_a_plan_covering_every_tensor() {
    let inventory: BTreeMap<String, Vec<usize>> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sortformer_inventory.json"
    )))
    .unwrap();
    let specs = specs_from_inventory(&inventory);
    assert_eq!(specs.len(), inventory.len());
    let parameters: usize = specs
        .iter()
        .map(|spec| spec.source_shape.iter().product::<usize>())
        .sum();
    assert!(
        (110_000_000..125_000_000).contains(&parameters),
        "unexpected parameter count {parameters}"
    );
    let int8 = specs
        .iter()
        .filter(|spec| matches!(spec.storage, Storage::Int8Affine { .. }))
        .count();
    assert!(
        int8 > 100,
        "expected the conformer stack quantized, got {int8}"
    );
}

#[test]
fn convert_model_rejects_zero_group_size_before_touching_the_filesystem() {
    // The nonexistent source path proves the group_size == 0 guard runs
    // before any file I/O: if the guard were missing or reordered after
    // `read_inventory`, this would fail with an I/O error instead.
    let error = convert_model("/nonexistent/does-not-exist.safetensors", "/tmp/unused", 0)
        .expect_err("group_size 0 must be rejected");
    assert!(
        matches!(error, ArtifactError::InvalidQuantization(_)),
        "expected InvalidQuantization, got {error:?}"
    );
}

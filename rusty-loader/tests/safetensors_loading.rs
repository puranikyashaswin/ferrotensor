//! Loading safetensors checkpoints from fixtures generated in the test
//! (no GPU, no downloads).

use rusty_loader::{LoaderError, ModelLoader, WeightsContext};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir =
            std::env::temp_dir().join(format!("rusty-loader-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Serialize `(name, dtype, shape, raw little-endian bytes)` to `path`.
fn write_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.to_string(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes)
                    .expect("tensor view"),
            )
        })
        .collect();
    std::fs::write(
        path,
        safetensors::tensor::serialize(views, &None).expect("serialize"),
    )
    .expect("write");
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
        .collect()
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

#[test]
fn reads_f32_f16_and_bf16_tensors_as_f32() {
    let dir = TempDir::new("dtypes");
    // Values exactly representable in every format.
    let values = [1.0f32, -2.5, 0.125, 3.0, 0.0, -0.75];
    write_safetensors(
        &dir.0.join("model.safetensors"),
        &[
            ("a.f32", Dtype::F32, vec![2, 3], f32_bytes(&values)),
            ("a.f16", Dtype::F16, vec![3, 2], f16_bytes(&values)),
            ("a.bf16", Dtype::BF16, vec![6], bf16_bytes(&values)),
        ],
    );

    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("load dir");
    for (name, shape) in [
        ("a.f32", vec![2, 3]),
        ("a.f16", vec![3, 2]),
        ("a.bf16", vec![6]),
    ] {
        let (data, got_shape) = weights.read_f32_tensor(name).expect(name);
        assert_eq!(got_shape, shape, "{name}");
        assert_eq!(data, values, "{name}");
    }
    let meta = weights.get_tensor_metadata("a.f16").expect("metadata");
    assert_eq!(meta.dtype, Dtype::F16);
    assert_eq!(
        weights.get_tensor_info("a.bf16").map(|(s, _)| s),
        Some(vec![6])
    );
}

#[test]
fn loads_sharded_checkpoints_through_the_index() {
    let dir = TempDir::new("sharded");
    write_safetensors(
        &dir.0.join("model-00001-of-00002.safetensors"),
        &[("x", Dtype::F32, vec![2], f32_bytes(&[1.0, 2.0]))],
    );
    write_safetensors(
        &dir.0.join("model-00002-of-00002.safetensors"),
        &[("y", Dtype::F32, vec![1], f32_bytes(&[3.0]))],
    );
    let index = serde_json::json!({
        "metadata": {"total_size": 12},
        "weight_map": {
            "x": "model-00001-of-00002.safetensors",
            "y": "model-00002-of-00002.safetensors",
        }
    });
    std::fs::write(
        dir.0.join("model.safetensors.index.json"),
        index.to_string(),
    )
    .expect("write index");

    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("load sharded");
    assert_eq!(weights.read_f32_tensor("x").expect("x").0, vec![1.0, 2.0]);
    assert_eq!(weights.read_f32_tensor("y").expect("y").0, vec![3.0]);
}

#[test]
fn missing_weight_and_unsupported_dtype_are_errors() {
    let dir = TempDir::new("errors");
    write_safetensors(
        &dir.0.join("model.safetensors"),
        &[("q", Dtype::I8, vec![4], vec![1, 2, 3, 4])],
    );
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("load");
    assert!(matches!(
        weights.read_f32_tensor("nope"),
        Err(LoaderError::MissingWeight(name)) if name == "nope"
    ));
    assert!(matches!(
        weights.read_f32_tensor("q"),
        Err(LoaderError::UnsupportedDtype { .. })
    ));
}

#[test]
fn empty_model_dir_is_an_error() {
    let dir = TempDir::new("empty");
    let mut weights = WeightsContext::new();
    let err = weights.load_model_dir(&dir.0).expect_err("no weights");
    assert!(err.to_string().contains("model.safetensors"), "{err}");
}

#[test]
fn corrupt_file_is_an_error_not_a_panic() {
    let dir = TempDir::new("corrupt");
    std::fs::write(
        dir.0.join("model.safetensors"),
        b"definitely not safetensors",
    )
    .expect("write");
    let mut weights = WeightsContext::new();
    assert!(weights.load_model_dir(&dir.0).is_err());
}

#[test]
fn llama_config_fields_are_read() {
    let dir = TempDir::new("config");
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "num_hidden_layers": 30,
        "vocab_size": 49152,
        "rms_norm_eps": 1e-6,
        "rope_theta": 100000,
        "tie_word_embeddings": true,
    });
    let path = dir.0.join("config.json");
    std::fs::write(&path, config.to_string()).expect("write");
    let c = ModelLoader::load_config(&path).expect("config");
    assert_eq!(c.model_type.as_deref(), Some("llama"));
    assert_eq!(c.num_key_value_heads, Some(3));
    assert_eq!(c.rope_theta, Some(100000.0));
    assert_eq!(c.tie_word_embeddings, Some(true));
    assert!((c.norm_eps - 1e-6).abs() < 1e-12);
    assert!(c.layer_types.is_empty());
}

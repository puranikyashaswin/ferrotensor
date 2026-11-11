//! Broken or hostile model files must produce an error, never a panic, a
//! hang, a huge allocation or a read outside the model directory.
//!
//! Every input is built in a temp dir and loaded through the public loader
//! functions. No GPU is needed. The GPU-side checks (`upload_quantized`, and
//! the shape of a tensor against the config in the model builder) are not
//! reachable from here.

use ferrotensor_loader::{Lfm2Config, LoaderError, ModelLoader, Tokenizer, WeightsContext};
use safetensors::Dtype;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrotensor-bad-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------- helpers

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

/// A safetensors file with the header written by hand: 8-byte little-endian
/// header length, the header text, then the data.
fn raw_safetensors(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Writes `bytes` as `model.safetensors` and loads the directory; the load
/// must fail.
fn load_error(bytes: &[u8]) -> LoaderError {
    let dir = TempDir::new("weights");
    std::fs::write(dir.0.join("model.safetensors"), bytes).expect("write");
    let mut weights = WeightsContext::new();
    match weights.load_model_dir(&dir.0) {
        Ok(()) => panic!("a broken weights file was accepted"),
        Err(e) => e,
    }
}

fn valid_config() -> Value {
    json!({
        "model_type": "llama",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "vocab_size": 100,
    })
}

fn config_with(field: &str, value: Value) -> Value {
    let mut c = valid_config();
    c[field] = value;
    c
}

fn config_without(field: &str) -> Value {
    let mut c = valid_config();
    c.as_object_mut().expect("object").remove(field);
    c
}

/// Writes `contents` as `config.json` and loads it.
fn load_config_text(contents: &[u8]) -> Result<Lfm2Config, LoaderError> {
    let dir = TempDir::new("config");
    let path = dir.0.join("config.json");
    std::fs::write(&path, contents).expect("write");
    ModelLoader::load_config(&path)
}

fn config_error(contents: &Value) -> String {
    load_config_text(contents.to_string().as_bytes())
        .expect_err("a broken config was accepted")
        .to_string()
}

/// Writes `contents` as `tokenizer.json` and loads it.
fn load_tokenizer_text(contents: &[u8]) -> Result<Tokenizer, String> {
    let dir = TempDir::new("tokenizer");
    let path = dir.0.join("tokenizer.json");
    std::fs::write(&path, contents).expect("write");
    Tokenizer::from_file(&path)
}

/// A tokenizer made of a plain `model.vocab` map.
fn tokenizer_with_vocab(vocab: &Value) -> Tokenizer {
    let text = json!({ "model": { "vocab": vocab } }).to_string();
    load_tokenizer_text(text.as_bytes()).expect("tokenizer")
}

/// Runs `f` on another thread and fails the test, instead of hanging it, if
/// it does not finish.
fn finishes_in_time(what: &str, f: impl FnOnce() -> bool + Send + 'static) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(is_err) => assert!(is_err, "{what}: loading a named pipe did not fail"),
        Err(_) => panic!("{what}: loading a named pipe hung"),
    }
}

// ----------------------------------------------------------------- config

#[test]
fn valid_configs_are_still_accepted() {
    load_config_text(valid_config().to_string().as_bytes()).expect("valid config");

    // `head_dim` set explicitly: hidden size need not divide by the heads.
    let mut c = valid_config();
    c["hidden_size"] = json!(100);
    c["num_attention_heads"] = json!(8);
    c["head_dim"] = json!(16);
    load_config_text(c.to_string().as_bytes()).expect("explicit head_dim");

    // LFM2 fields, quantization, and unknown keys.
    let mut c = valid_config();
    c["block_dim"] = json!(64);
    c["block_ff_dim"] = json!(128);
    c["layer_types"] = json!(["conv", "full_attention"]);
    c["quantization"] = json!({"group_size": 64, "bits": 4, "mode": "affine"});
    c["something_new"] = json!({"nested": [1, 2, 3]});
    let cfg = load_config_text(c.to_string().as_bytes()).expect("lfm2 config");
    assert_eq!(cfg.layer_types.len(), 2);
}

#[test]
fn missing_config_file_is_an_error_that_names_the_file() {
    let dir = TempDir::new("no-config");
    let err = ModelLoader::load_config(dir.0.join("config.json")).expect_err("no config");
    assert!(err.to_string().contains("config.json"), "{err}");
}

#[test]
fn empty_config_file_is_an_error() {
    let err = load_config_text(b"").expect_err("empty config");
    assert!(err.to_string().contains("config.json"), "{err}");
}

#[test]
fn invalid_json_config_is_an_error() {
    for text in [
        &b"{not json"[..],
        b"{\"hidden_size\": 64,",
        b"\xff\xfe\x00garbage",
        b"[]",
        b"null",
        b"42",
    ] {
        let err = load_config_text(text).expect_err("invalid config");
        assert!(err.to_string().contains("config.json"), "{err}");
    }
}

#[test]
fn config_missing_a_required_field_names_the_field() {
    for field in [
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_hidden_layers",
        "vocab_size",
    ] {
        let msg = config_error(&config_without(field));
        assert!(msg.contains(field), "{field}: {msg}");
        assert!(msg.contains("config.json"), "{field}: {msg}");
    }
}

#[test]
fn config_field_with_the_wrong_type_is_an_error() {
    for (field, value) in [
        ("hidden_size", json!("64")),
        ("hidden_size", json!(64.5)),
        ("hidden_size", json!(-1)),
        ("hidden_size", json!(null)),
        ("num_attention_heads", json!([4])),
        ("num_hidden_layers", json!(true)),
        ("vocab_size", json!({"n": 100})),
        ("rms_norm_eps", json!("small")),
        ("layer_types", json!("conv")),
        ("quantization", json!(4)),
        ("tie_word_embeddings", json!("yes")),
    ] {
        let msg = config_error(&config_with(field, value.clone()));
        assert!(msg.contains("config.json"), "{field}={value}: {msg}");
    }
}

#[test]
fn config_with_zero_absurd_or_inconsistent_sizes_names_the_field() {
    let cases = [
        ("hidden_size", json!(0), "hidden_size"),
        ("num_attention_heads", json!(0), "num_attention_heads"),
        ("num_hidden_layers", json!(0), "num_hidden_layers"),
        ("vocab_size", json!(0), "vocab_size"),
        ("intermediate_size", json!(0), "intermediate_size"),
        // Billions of layers, or hidden sizes no model has.
        (
            "num_hidden_layers",
            json!(3_000_000_000u64),
            "num_hidden_layers",
        ),
        ("hidden_size", json!(1_000_000_000_000u64), "hidden_size"),
        (
            "intermediate_size",
            json!(1_000_000_000_000u64),
            "intermediate_size",
        ),
        ("vocab_size", json!(1_000_000_000_000u64), "vocab_size"),
        (
            "num_attention_heads",
            json!(1_000_000_000u64),
            "num_attention_heads",
        ),
        ("hidden_size", json!(u64::MAX), "hidden_size"),
        // Hidden size not divisible by the heads (no `head_dim` given).
        ("hidden_size", json!(65), "num_attention_heads"),
        // Key/value heads.
        ("num_key_value_heads", json!(0), "num_key_value_heads"),
        ("num_key_value_heads", json!(3), "num_key_value_heads"),
        ("num_key_value_heads", json!(8), "num_key_value_heads"),
        ("head_dim", json!(0), "head_dim"),
        ("block_dim", json!(0), "block_dim"),
        ("block_ff_dim", json!(0), "block_ff_dim"),
        (
            "quantization",
            json!({"group_size": 0, "bits": 4, "mode": "affine"}),
            "quantization.group_size",
        ),
        (
            "quantization",
            json!({"group_size": 64, "bits": 0, "mode": "affine"}),
            "quantization.bits",
        ),
        (
            "quantization",
            json!({"group_size": 64, "bits": 64, "mode": "affine"}),
            "quantization.bits",
        ),
    ];
    for (field, value, named) in cases {
        let msg = config_error(&config_with(field, value.clone()));
        assert!(msg.contains(named), "{field}={value}: {msg}");
        assert!(msg.contains("config.json"), "{field}={value}: {msg}");
    }
}

#[test]
fn config_that_is_a_directory_is_an_error() {
    let dir = TempDir::new("config-dir");
    let path = dir.0.join("config.json");
    std::fs::create_dir(&path).expect("mkdir");
    let err = ModelLoader::load_config(&path).expect_err("directory");
    assert!(err.to_string().contains("config.json"), "{err}");
}

#[test]
fn oversized_config_is_refused_without_being_read() {
    let dir = TempDir::new("config-big");
    let path = dir.0.join("config.json");
    // A sparse 65 MiB file: past the limit, almost free to create.
    let file = std::fs::File::create(&path).expect("create");
    file.set_len(65 * 1024 * 1024).expect("set_len");
    drop(file);
    let err = ModelLoader::load_config(&path).expect_err("too big");
    let msg = err.to_string();
    assert!(
        msg.contains("config.json") && msg.contains("limit"),
        "{msg}"
    );
}

#[test]
fn deeply_nested_json_is_an_error_not_a_stack_overflow() {
    let deep = "[".repeat(200_000);
    let err = load_config_text(deep.as_bytes()).expect_err("deep config");
    assert!(err.to_string().contains("config.json"), "{err}");
    let deep_objects = "{\"a\":".repeat(200_000);
    assert!(load_config_text(deep_objects.as_bytes()).is_err());
    assert!(load_tokenizer_text(deep.as_bytes()).is_err());
    assert!(load_tokenizer_text(deep_objects.as_bytes()).is_err());
}

// ---------------------------------------------------------------- weights

#[test]
fn missing_weights_file_is_an_error_that_names_the_file() {
    let dir = TempDir::new("no-weights");
    let mut weights = WeightsContext::new();
    let err = weights
        .load_file(dir.0.join("model.safetensors"))
        .expect_err("missing file");
    assert!(err.to_string().contains("model.safetensors"), "{err}");
    let err =
        ModelLoader::load_safetensors(dir.0.join("model.safetensors")).expect_err("missing file");
    assert!(err.to_string().contains("model.safetensors"), "{err}");
}

#[test]
fn weights_path_that_is_a_directory_is_an_error() {
    let dir = TempDir::new("weights-dir");
    std::fs::create_dir(dir.0.join("model.safetensors")).expect("mkdir");
    let mut weights = WeightsContext::new();
    let err = weights.load_model_dir(&dir.0).expect_err("directory");
    assert!(err.to_string().contains("model.safetensors"), "{err}");
}

#[test]
fn zero_byte_weights_file_is_an_error() {
    load_error(b"");
}

#[test]
fn weights_file_shorter_than_the_length_prefix_is_an_error() {
    load_error(&[1, 2, 3, 4]);
    load_error(&[0u8; 7]);
}

#[test]
fn weights_file_shorter_than_its_header_length_says_is_an_error() {
    // Says the header is 1000 bytes; only 2 follow.
    let mut bytes = 1000u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    load_error(&bytes);
}

#[test]
fn header_length_larger_than_the_file_or_any_sane_size_is_an_error() {
    for len in [u64::MAX, u64::MAX - 7, 1 << 40, 200_000_000] {
        let mut bytes = len.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        load_error(&bytes);
    }
}

#[test]
fn header_that_is_not_json_is_an_error() {
    load_error(&raw_safetensors("this is not json", &[]));
    load_error(&raw_safetensors("{\"t\":", &[]));
    load_error(&raw_safetensors("[]", &[]));
    // Not UTF-8.
    let mut bytes = 4u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[0xff, 0xfe, 0xfd, 0xfc]);
    load_error(&bytes);
}

#[test]
fn tensor_range_past_the_end_of_the_file_is_an_error() {
    // Declares 16 data bytes, the file holds 8.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
        &[0u8; 8],
    ));
    // Declares far more than any file could hold.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"I8","shape":[1099511627776],"data_offsets":[0,1099511627776]}}"#,
        &[0u8; 8],
    ));
}

#[test]
fn bytes_after_the_last_tensor_are_an_error() {
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0u8; 12],
    ));
}

#[test]
fn tensor_byte_length_that_does_not_match_shape_and_dtype_is_an_error() {
    // [4] F32 needs 16 bytes, the range holds 8.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    ));
    // [2] F16 needs 4 bytes, the range holds 8.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F16","shape":[2],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    ));
}

#[test]
fn shape_that_overflows_is_an_error() {
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}"#,
        &[],
    ));
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}"#,
        &[],
    ));
}

#[test]
fn broken_offsets_are_an_error() {
    // End before start.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
        &[0u8; 4],
    ));
    // Does not start at zero.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        &[0u8; 8],
    ));
    // Two tensors that overlap.
    load_error(&raw_safetensors(
        r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#,
        &[0u8; 12],
    ));
    // Negative and fractional offsets.
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[-4,0]}}"#,
        &[0u8; 4],
    ));
    load_error(&raw_safetensors(
        r#"{"t":{"dtype":"F32","shape":[1],"data_offsets":[0.5,4]}}"#,
        &[0u8; 4],
    ));
}

#[test]
fn header_entry_with_bad_fields_is_an_error() {
    for header in [
        // Unknown dtype name.
        r#"{"t":{"dtype":"F7","shape":[1],"data_offsets":[0,4]}}"#,
        // Shape of the wrong type or with negative sizes.
        r#"{"t":{"dtype":"F32","shape":"abc","data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
        // Missing fields.
        r#"{"t":{"dtype":"F32","shape":[1]}}"#,
        r#"{"t":{"shape":[1],"data_offsets":[0,4]}}"#,
        // Entry that is not an object.
        r#"{"t":7}"#,
    ] {
        load_error(&raw_safetensors(header, &[0u8; 4]));
    }
}

#[test]
fn unsupported_dtypes_are_an_error_that_names_the_tensor() {
    let dir = TempDir::new("dtypes");
    write_safetensors(
        &dir.0.join("model.safetensors"),
        &[
            ("bytes", Dtype::U8, vec![4], vec![1, 2, 3, 4]),
            ("ints", Dtype::I64, vec![1], 7i64.to_le_bytes().to_vec()),
            (
                "doubles",
                Dtype::F64,
                vec![1],
                1.5f64.to_le_bytes().to_vec(),
            ),
            ("flags", Dtype::BOOL, vec![2], vec![0, 1]),
        ],
    );
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("load");
    for name in ["bytes", "ints", "doubles", "flags"] {
        match weights.read_f32_tensor(name) {
            Err(LoaderError::UnsupportedDtype { name: n, .. }) => assert_eq!(n, name),
            other => panic!("{name}: expected UnsupportedDtype, got {other:?}"),
        }
    }
}

#[test]
fn a_tensor_the_model_needs_but_the_file_lacks_is_an_error() {
    let dir = TempDir::new("absent");
    write_safetensors(
        &dir.0.join("model.safetensors"),
        &[("present", Dtype::F32, vec![1], f32_bytes(&[1.0]))],
    );
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("load");

    let name = "model.embed_tokens.weight";
    assert!(matches!(
        weights.read_f32_tensor(name),
        Err(LoaderError::MissingWeight(n)) if n == name
    ));
    assert!(matches!(
        weights.get_tensor_metadata(name),
        Err(LoaderError::MissingWeight(n)) if n == name
    ));
    assert!(matches!(
        weights.read_bytes(name, |_, _| panic!("callback must not run")),
        Err(LoaderError::MissingWeight(n)) if n == name
    ));
    assert!(weights.get_tensor_info(name).is_none());
    // Names are matched exactly, not by prefix.
    assert!(weights.read_f32_tensor("present.weight").is_err());
    assert!(weights.read_f32_tensor("").is_err());
}

#[test]
fn a_tensor_with_the_wrong_shape_reports_its_stored_shape() {
    // The loader does not know the config; a caller (the model builder)
    // compares the stored shape with the one it expects. This checks the
    // stored shape is reported faithfully, so a mismatch can be caught.
    let dir = TempDir::new("shape");
    write_safetensors(
        &dir.0.join("model.safetensors"),
        &[(
            "model.embed_tokens.weight",
            Dtype::F32,
            vec![3, 3],
            f32_bytes(&[0.0; 9]),
        )],
    );
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("load");
    let config = valid_config();
    let expected = [
        config["vocab_size"].as_u64().expect("vocab") as usize,
        config["hidden_size"].as_u64().expect("hidden") as usize,
    ];
    let meta = weights
        .get_tensor_metadata("model.embed_tokens.weight")
        .expect("metadata");
    assert_eq!(meta.shape, vec![3, 3]);
    assert_ne!(meta.shape, expected);
    let (data, shape) = weights
        .read_f32_tensor("model.embed_tokens.weight")
        .expect("read");
    assert_eq!((data.len(), shape), (9, vec![3, 3]));
}

// ----------------------------------------------------------------- shards

/// A model dir `<root>/model` holding an index with `weight_map`.
fn model_dir_with_index(root: &TempDir, weight_map: Value) -> PathBuf {
    let model = root.0.join("model");
    std::fs::create_dir_all(&model).expect("mkdir");
    let index = json!({ "metadata": {"total_size": 4}, "weight_map": weight_map });
    std::fs::write(
        model.join("model.safetensors.index.json"),
        index.to_string(),
    )
    .expect("write index");
    model
}

#[test]
fn index_naming_a_missing_shard_is_an_error_that_names_the_shard() {
    let root = TempDir::new("missing-shard");
    let model = model_dir_with_index(&root, json!({"x": "model-00001-of-00002.safetensors"}));
    let mut weights = WeightsContext::new();
    let err = weights.load_model_dir(&model).expect_err("missing shard");
    assert!(
        err.to_string().contains("model-00001-of-00002.safetensors"),
        "{err}"
    );
}

#[test]
fn index_naming_a_shard_that_is_a_directory_is_an_error() {
    let root = TempDir::new("dir-shard");
    let model = model_dir_with_index(&root, json!({"x": "shard.safetensors"}));
    std::fs::create_dir(model.join("shard.safetensors")).expect("mkdir");
    let mut weights = WeightsContext::new();
    assert!(weights.load_model_dir(&model).is_err());
}

#[test]
fn index_shard_paths_that_leave_the_model_directory_are_refused() {
    let root = TempDir::new("escape");
    // A valid weights file outside the model directory. If the loader
    // followed a path to it, the load would succeed and "secret" would exist.
    let outside = root.0.join("outside.safetensors");
    write_safetensors(
        &outside,
        &[("secret", Dtype::F32, vec![1], f32_bytes(&[42.0]))],
    );
    let absolute = outside.to_str().expect("utf-8 temp path").to_string();

    for shard in [
        "../outside.safetensors",
        "sub/../../outside.safetensors",
        "..",
        "",
        "/",
        absolute.as_str(),
    ] {
        let model = model_dir_with_index(&root, json!({ "secret": shard }));
        let mut weights = WeightsContext::new();
        let err = weights
            .load_model_dir(&model)
            .expect_err("a shard outside the model directory was accepted");
        let msg = err.to_string();
        assert!(
            msg.contains("model.safetensors.index.json"),
            "{shard}: {msg}"
        );
        assert!(msg.contains("model directory"), "{shard}: {msg}");
        assert!(weights.get_tensor_info("secret").is_none(), "{shard}");
        assert!(weights.read_f32_tensor("secret").is_err(), "{shard}");
    }
}

#[test]
fn one_bad_shard_path_stops_the_load_before_any_shard_is_read() {
    let root = TempDir::new("escape-mixed");
    let outside = root.0.join("outside.safetensors");
    write_safetensors(
        &outside,
        &[("secret", Dtype::F32, vec![1], f32_bytes(&[1.0]))],
    );
    let model = model_dir_with_index(
        &root,
        json!({"x": "good.safetensors", "secret": "../outside.safetensors"}),
    );
    write_safetensors(
        &model.join("good.safetensors"),
        &[("x", Dtype::F32, vec![1], f32_bytes(&[2.0]))],
    );
    let mut weights = WeightsContext::new();
    assert!(weights.load_model_dir(&model).is_err());
    assert!(weights.get_tensor_info("x").is_none());
    assert!(weights.get_tensor_info("secret").is_none());
}

#[test]
fn index_that_lists_no_weights_is_an_error() {
    let root = TempDir::new("empty-index");
    let model = model_dir_with_index(&root, json!({}));
    let mut weights = WeightsContext::new();
    let err = weights.load_model_dir(&model).expect_err("empty index");
    assert!(
        err.to_string().contains("model.safetensors.index.json"),
        "{err}"
    );
}

#[test]
fn broken_index_files_are_an_error_that_names_the_index() {
    let root = TempDir::new("bad-index");
    let model = root.0.join("model");
    std::fs::create_dir_all(&model).expect("mkdir");
    let index = model.join("model.safetensors.index.json");
    let long_name = "x".repeat(10_000);
    let bad_shard_name = format!(r#"{{"weight_map":{{"a":"{long_name}"}}}}"#);
    for text in [
        "",
        "{not json",
        "[]",
        "null",
        r#"{"metadata":{}}"#,
        r#"{"weight_map":[]}"#,
        r#"{"weight_map":{"a":7}}"#,
        r#"{"weight_map":{"a":null}}"#,
        r#"{"weight_map":{"a":"has\u0000nul.safetensors"}}"#,
        bad_shard_name.as_str(),
    ] {
        std::fs::write(&index, text).expect("write index");
        let mut weights = WeightsContext::new();
        let err = weights
            .load_model_dir(&model)
            .expect_err("a broken index was accepted");
        let shown = err.to_string();
        // Either the index itself is named, or the shard path it asked for.
        assert!(
            shown.contains("model.safetensors.index.json") || shown.contains(".safetensors"),
            "{shown}"
        );
    }
}

#[test]
fn oversized_index_is_refused_without_being_read() {
    let root = TempDir::new("big-index");
    let model = root.0.join("model");
    std::fs::create_dir_all(&model).expect("mkdir");
    let file = std::fs::File::create(model.join("model.safetensors.index.json")).expect("create");
    file.set_len(65 * 1024 * 1024).expect("set_len");
    drop(file);
    let mut weights = WeightsContext::new();
    let err = weights.load_model_dir(&model).expect_err("too big");
    assert!(err.to_string().contains("limit"), "{err}");
}

// -------------------------------------------------------------- tokenizer

#[test]
fn missing_tokenizer_file_is_an_error_that_names_the_file() {
    let dir = TempDir::new("no-tokenizer");
    let err = Tokenizer::from_file(dir.0.join("tokenizer.json")).expect_err("missing");
    assert!(err.contains("tokenizer.json"), "{err}");
}

#[test]
fn empty_and_invalid_tokenizer_files_are_an_error() {
    for text in [
        &b""[..],
        b"{not json",
        b"\xff\xfe\x00garbage",
        b"[]",
        b"null",
        b"7",
        b"\"text\"",
    ] {
        let err = load_tokenizer_text(text).expect_err("invalid tokenizer");
        assert!(err.contains("tokenizer.json"), "{err}");
    }
}

#[test]
fn tokenizer_json_without_a_vocabulary_is_an_error() {
    for text in [
        r#"{}"#,
        r#"{"version":"1.0"}"#,
        r#"{"model":null}"#,
        r#"{"model":{}}"#,
        r#"{"model":{"type":"BPE"}}"#,
        r#"{"model":{"vocab":null}}"#,
        r#"{"model":{"vocab":[]}}"#,
        r#"{"model":{"vocab":"abc"}}"#,
        r#"{"vocab":{"a":0}}"#,
    ] {
        let err = load_tokenizer_text(text.as_bytes()).expect_err("tokenizer without vocabulary");
        assert!(err.contains("tokenizer.json"), "{text}: {err}");
    }
}

#[test]
fn tokenizer_with_an_empty_vocabulary_is_an_error() {
    // Plain `model.vocab` map.
    let err = load_tokenizer_text(br#"{"model":{"vocab":{}}}"#).expect_err("empty vocabulary");
    assert!(
        err.contains("tokenizer.json") && err.contains("empty"),
        "{err}"
    );

    // A full Hugging Face tokenizer with nothing in it.
    let err = load_tokenizer_text(br#"{"model":{"type":"BPE","vocab":{},"merges":[]}}"#)
        .expect_err("empty BPE vocabulary");
    assert!(
        err.contains("tokenizer.json") && err.contains("empty"),
        "{err}"
    );
}

#[test]
fn tokenizer_ids_that_repeat_are_an_error() {
    let err = load_tokenizer_text(br#"{"model":{"vocab":{"a":3,"b":3,"c":4}}}"#)
        .expect_err("repeated id");
    assert!(err.contains("tokenizer.json"), "{err}");
    assert!(err.contains("more than one token"), "{err}");
}

#[test]
fn tokenizer_ids_with_gaps_are_accepted_and_decode_safely() {
    // Real vocabularies have gaps (the model's embedding table is larger
    // than the vocabulary), so gaps are fine. The ids that are missing must
    // simply decode to nothing.
    let tok = tokenizer_with_vocab(&json!({"x": 10, "y": 30, "<unk>": 50}));
    assert_eq!(tok.encode_continuation("x y"), vec![10, 30]);
    assert_eq!(tok.decode(&[10, 30]), "xy");
    assert_eq!(tok.decode(&[10, 20, 30, 49]), "xy");
}

#[test]
fn encoding_empty_text_does_not_panic() {
    let tok = tokenizer_with_vocab(&json!({"x": 10, "y": 30}));
    assert!(tok.encode_continuation("").is_empty());
    assert!(tok.encode_continuation("   \n\t ").is_empty());
    // With special tokens: at most the beginning-of-sequence token.
    assert!(tok.encode("").len() <= 1);
    assert!(tok.encode("   ").len() <= 1);
}

#[test]
fn encoding_text_with_unknown_characters_does_not_panic() {
    let tok = tokenizer_with_vocab(&json!({"x": 10, "y": 30}));
    let ids = tok.encode_continuation("zz \u{1F600} \u{0} \u{202e}x");
    assert!(!ids.is_empty());
    let long = "q".repeat(100_000);
    assert!(!tok.encode_continuation(&long).is_empty());
}

#[test]
fn decoding_ids_outside_the_vocabulary_does_not_panic() {
    let tok = tokenizer_with_vocab(&json!({"x": 10, "y": 30}));
    assert_eq!(tok.decode(&[]), "");
    assert_eq!(tok.decode(&[999_999, u32::MAX]), "");
    assert_eq!(tok.decode(&[u32::MAX, 10, 999_999, 30]), "xy");
}

#[test]
fn hugging_face_tokenizer_survives_empty_text_and_unknown_ids() {
    let tok = load_tokenizer_text(
        br#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}}"#,
    )
    .expect("hugging face tokenizer");
    assert!(tok.hf_tokenizer.is_some());
    let _ = tok.encode("");
    let _ = tok.encode_continuation("");
    assert_eq!(tok.decode(&[]), "");
    assert_eq!(tok.decode(&[99, 1_000_000, u32::MAX]), "");
}

#[test]
fn tokenizer_that_is_a_directory_is_an_error() {
    let dir = TempDir::new("tokenizer-dir");
    let path = dir.0.join("tokenizer.json");
    std::fs::create_dir(&path).expect("mkdir");
    let err = Tokenizer::from_file(&path).expect_err("directory");
    assert!(err.contains("tokenizer.json"), "{err}");
}

#[test]
fn oversized_tokenizer_is_refused_without_being_read() {
    let dir = TempDir::new("tokenizer-big");
    let path = dir.0.join("tokenizer.json");
    let file = std::fs::File::create(&path).expect("create");
    file.set_len(257 * 1024 * 1024).expect("set_len");
    drop(file);
    let err = Tokenizer::from_file(&path).expect_err("too big");
    assert!(
        err.contains("tokenizer.json") && err.contains("limit"),
        "{err}"
    );
}

// ------------------------------------------------------------ named pipes

#[cfg(unix)]
#[test]
fn named_pipes_in_place_of_model_files_do_not_hang_the_loader() {
    fn make_pipe(path: &Path) -> bool {
        matches!(
            std::process::Command::new("mkfifo").arg(path).status(),
            Ok(status) if status.success()
        )
    }

    let dir = TempDir::new("pipes");
    let config = dir.0.join("config.json");
    let tokenizer = dir.0.join("tokenizer.json");
    let weights_dir = dir.0.join("weights");
    let index_dir = dir.0.join("index");
    std::fs::create_dir_all(&weights_dir).expect("mkdir");
    std::fs::create_dir_all(&index_dir).expect("mkdir");
    let weights = weights_dir.join("model.safetensors");
    let index = index_dir.join("model.safetensors.index.json");

    if ![&config, &tokenizer, &weights, &index]
        .iter()
        .all(|p| make_pipe(p))
    {
        // No `mkfifo` here: nothing to check.
        return;
    }

    finishes_in_time("config", move || {
        ModelLoader::load_config(config.as_path()).is_err()
    });
    finishes_in_time("tokenizer", move || {
        Tokenizer::from_file(tokenizer.as_path()).is_err()
    });
    finishes_in_time("weights", move || {
        WeightsContext::new()
            .load_model_dir(weights_dir.as_path())
            .is_err()
    });
    finishes_in_time("index", move || {
        WeightsContext::new()
            .load_model_dir(index_dir.as_path())
            .is_err()
    });
}

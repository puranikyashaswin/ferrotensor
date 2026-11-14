//! Loading a checkpoint uploads the same bytes as the original upload path.
//!
//! A tiny Llama-style checkpoint is written in each source dtype (F32, F16,
//! BF16), with tied and with separate `lm_head` weights, and built with f16
//! weights on and off. The GPU contents of `lm_head` and of one linear weight
//! are read back and compared byte for byte with what the original path
//! produced: read as f32, transpose into a second f32 copy, then upload the
//! copy as f32 or convert it to f16 (kept below as a test-only reference).
//! The host copy of the embedding table that the builder hands the model
//! is compared with the table read back from the GPU.
//! These tests need a GPU adapter, so they are `#[ignore]`d.

use crate::lfm::{Lfm2LayerOp, Lfm2Model};
use crate::model_builder::Lfm2ModelBuilder;
use ferrotensor_backend::{ComputeEngine, DType, UnifiedTensor, WgpuContext};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const VOCAB: usize = 13;
const HIDDEN: usize = 8;
const HEADS: usize = 2;
const FFN: usize = 12;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrotensor-graph-{tag}-{}-{nanos}",
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

/// Deterministic bit patterns: every value class (NaNs with payloads,
/// infinities, subnormals, zeros) turns up among them.
fn random_bits(n: usize, seed: u64) -> Vec<u64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect()
}

/// Raw little-endian bytes of `n` elements of `dtype`, starting with the
/// special values of that format.
fn tensor_bytes(dtype: Dtype, n: usize, seed: u64) -> Vec<u8> {
    let bits = random_bits(n, seed);
    match dtype {
        Dtype::F32 => {
            let edges = [
                0.0f32,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::from_bits(0x7fc1_2345), // NaN with payload
                f32::from_bits(0xff80_0001), // negative signalling NaN
                f32::from_bits(1),           // subnormal
                -f32::from_bits(0x0040_0000),
                65520.0, // overflows f16
                -1e10,
                2f32.powi(-25), // rounds to zero in f16
                f32::MAX,
            ];
            (0..n)
                .map(|i| {
                    edges
                        .get(i)
                        .copied()
                        .unwrap_or(f32::from_bits(bits[i] as u32))
                })
                .flat_map(f32::to_le_bytes)
                .collect()
        }
        Dtype::F16 | Dtype::BF16 => {
            let edges: [u16; 8] = [
                0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x0001, 0x7f81, 0xff80,
            ];
            (0..n)
                .map(|i| edges.get(i).copied().unwrap_or(bits[i] as u16))
                .flat_map(u16::to_le_bytes)
                .collect()
        }
        other => panic!("no test data for {other:?}"),
    }
}

/// Writes `config.json` and `model.safetensors` of a one-layer Llama-style
/// model whose weights are stored as `dtype`; `lm_head.weight` is written
/// only when `tied` is false.
fn write_checkpoint(dir: &Path, dtype: Dtype, tied: bool) {
    write_checkpoint_with(dir, dtype, tied, |n, seed| tensor_bytes(dtype, n, seed));
}

/// [`write_checkpoint`] with each tensor's bytes made by `bytes(n, seed)`.
fn write_checkpoint_with(
    dir: &Path,
    dtype: Dtype,
    tied: bool,
    bytes: impl Fn(usize, u64) -> Vec<u8>,
) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "num_hidden_layers": 1,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": tied,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write config");

    let mut shapes: Vec<(&str, Vec<usize>)> = vec![
        ("model.embed_tokens.weight", vec![VOCAB, HIDDEN]),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![HIDDEN, HIDDEN],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![HIDDEN, HIDDEN],
        ),
        ("model.layers.0.mlp.gate_proj.weight", vec![FFN, HIDDEN]),
        ("model.layers.0.mlp.up_proj.weight", vec![FFN, HIDDEN]),
        ("model.layers.0.mlp.down_proj.weight", vec![HIDDEN, FFN]),
        ("model.layers.0.input_layernorm.weight", vec![HIDDEN]),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![HIDDEN],
        ),
        ("model.norm.weight", vec![HIDDEN]),
    ];
    if !tied {
        shapes.push(("lm_head.weight", vec![VOCAB, HIDDEN]));
    }
    let data: Vec<(&str, Vec<usize>, Vec<u8>)> = shapes
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let n = shape.iter().product();
            (name, shape, bytes(n, i as u64 + 7))
        })
        .collect();
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = data
        .iter()
        .map(|(name, shape, bytes)| {
            let view = safetensors::tensor::TensorView::new(dtype, shape.clone(), bytes)
                .expect("tensor view");
            (name.to_string(), view)
        })
        .collect();
    let file = safetensors::tensor::serialize(views, &None).expect("serialize");
    std::fs::write(dir.join("model.safetensors"), file).expect("write weights");
}

/// Little-endian f32 bytes of `n` finite values in `[-0.5, 0.5)`: 4-bit
/// storage rejects NaN and infinities.
fn finite_f32_bytes(n: usize, seed: u64) -> Vec<u8> {
    random_bits(n, seed)
        .into_iter()
        .flat_map(|b| ((b >> 40) as f32 / (1u64 << 24) as f32 - 0.5).to_le_bytes())
        .collect()
}

/// The bytes the builder uploaded for a `[rows, cols]` linear weight before
/// it streamed the conversion: a transposed f32 copy, then its bytes as
/// `UnifiedTensor::new` reads them, or the f16 bytes
/// `UnifiedTensor::from_f32_as_f16` builds from it.
fn old_upload_bytes(values: &[f32], rows: usize, cols: usize, f16: bool) -> Vec<u8> {
    let mut transposed = vec![0.0f32; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = values[row * cols + col];
        }
    }
    if f16 {
        let mut bytes: Vec<u8> = transposed
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        bytes.resize(bytes.len().next_multiple_of(4).max(4), 0);
        bytes
    } else {
        transposed.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }
}

struct Gpu {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
}

fn gpu() -> Gpu {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = Arc::new(ComputeEngine::new(&ctx));
    Gpu { ctx, engine }
}

fn build(gpu: &Gpu, dir: &Path, f16: bool) -> (Lfm2Model, WeightsContext) {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    let model = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_f16_weights(f16)
        .build()
        .expect("build");
    (model, weights)
}

/// The raw bytes of a GPU tensor.
fn gpu_bytes(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<u8> {
    let words = pollster::block_on(t.to_vec(ctx));
    words.iter().flat_map(|w| w.to_ne_bytes()).collect()
}

fn assert_bytes_eq(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{what}: byte length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!("{what}: first differing byte at {i}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn uploaded_weights_match_the_old_path_in_every_dtype() {
    let gpu = gpu();
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        for tied in [true, false] {
            let dir = TempDir::new("upload-bytes");
            write_checkpoint(&dir.0, dtype, tied);
            for f16 in [false, true] {
                let case = format!("{dtype:?}, tied {tied}, f16 weights {f16}");
                let (model, weights) = build(&gpu, &dir.0, f16);

                let head_source = if tied {
                    "model.embed_tokens.weight"
                } else {
                    "lm_head.weight"
                };
                let (values, shape) = weights.read_f32_tensor(head_source).expect("read");
                let want = old_upload_bytes(&values, shape[0], shape[1], f16);
                let got = gpu_bytes(&gpu.ctx, &model.lm_head.weight.data);
                assert_bytes_eq(&format!("lm_head ({case})"), &got, &want);
                assert_eq!(model.lm_head.weight.data.shape, vec![HIDDEN, VOCAB]);

                let name = "model.layers.0.self_attn.q_proj.weight";
                let (values, shape) = weights.read_f32_tensor(name).expect("read");
                let want = old_upload_bytes(&values, shape[0], shape[1], f16);
                let Lfm2LayerOp::Attention(attn) = &model.layers[0].op else {
                    panic!("layer 0 is not attention");
                };
                let got = gpu_bytes(&gpu.ctx, &attn.q_proj.weight.data);
                assert_bytes_eq(&format!("q_proj ({case})"), &got, &want);
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn host_embeddings_come_from_the_checkpoint_and_match_the_gpu_table() {
    let gpu = gpu();
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        let dir = TempDir::new("embed-host");
        write_checkpoint(&dir.0, dtype, true);
        let (model, _weights) = build(&gpu, &dir.0, true);
        // Filled by the builder, so the first forward pass finds it and
        // does not read the table back from the GPU.
        let host = model
            .host_embeddings()
            .expect("the builder fills the host table");
        let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let gpu_table = gpu_bytes(&gpu.ctx, &model.embed_tokens.data);
        assert_bytes_eq(&format!("embed_host ({dtype:?})"), &host_bytes, &gpu_table);
    }
}

#[test]
#[ignore = "needs a GPU"]
fn models_assembled_by_hand_still_read_the_table_back_on_first_use() {
    let gpu = gpu();
    let dir = TempDir::new("embed-lazy");
    write_checkpoint(&dir.0, Dtype::F32, true);
    let (built, _weights) = build(&gpu, &dir.0, false);
    let model = Lfm2Model::new(
        built.embed_tokens,
        built.embedding_norm,
        built.layers,
        built.norm_f,
        built.lm_head,
    );
    assert!(model.host_embeddings().is_none());
    let _ = model.forward(&gpu.engine, &[1, 2, 3], 0, None, None);
    let host = model.host_embeddings().expect("filled on first use");
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let gpu_table = gpu_bytes(&gpu.ctx, &model.embed_tokens.data);
    assert_bytes_eq("embed_host (read back)", &host_bytes, &gpu_table);
}

#[test]
#[ignore = "needs a GPU"]
fn q4_load_keeps_the_tied_head_and_host_table_unquantised_without_the_file() {
    let gpu = gpu();
    let dir = TempDir::new("q4-tied");
    write_checkpoint_with(&dir.0, Dtype::F32, true, finite_f32_bytes);
    let config = ModelLoader::load_config(dir.0.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir.0).expect("weights");
    // Read before the build too: the build then reads pages that were
    // released after this read, which must fault them in unchanged.
    let (table, shape) = weights
        .read_f32_tensor("model.embed_tokens.weight")
        .expect("read");
    let model = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_f16_weights(true)
        .with_q4_weights(true)
        .build()
        .expect("build");
    // Unmapped and deleted, as the CLI drops the checkpoint once the model
    // is built: nothing below may read the file.
    drop(weights);
    std::fs::remove_file(dir.0.join("model.safetensors")).expect("delete weights");

    let table_bytes: Vec<u8> = table.iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(model.embed_tokens.data.dtype, DType::FP32);
    let gpu_table = gpu_bytes(&gpu.ctx, &model.embed_tokens.data);
    assert_bytes_eq("embed_tokens (q4)", &gpu_table, &table_bytes);
    let host = model
        .host_embeddings()
        .expect("the builder fills the host table");
    let host_bytes: Vec<u8> = host.iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_bytes_eq("embed_host (q4)", &host_bytes, &table_bytes);

    // The tied head follows the f16 setting, never 4-bit.
    assert_eq!(model.lm_head.weight.data.dtype, DType::FP16);
    let want = old_upload_bytes(&table, shape[0], shape[1], true);
    let got = gpu_bytes(&gpu.ctx, &model.lm_head.weight.data);
    assert_bytes_eq("lm_head (q4)", &got, &want);

    let Lfm2LayerOp::Attention(attn) = &model.layers[0].op else {
        panic!("layer 0 is not attention");
    };
    assert_eq!(attn.q_proj.weight.data.dtype, DType::Q4);

    let logits = model.forward(&gpu.engine, &[1, 2, 3], 0, None, None);
    let values = pollster::block_on(logits.data.to_vec(&gpu.ctx));
    assert!(!values.is_empty() && values.len().is_multiple_of(VOCAB));
    assert!(values.iter().all(|v| v.is_finite()), "non-finite logits");
}

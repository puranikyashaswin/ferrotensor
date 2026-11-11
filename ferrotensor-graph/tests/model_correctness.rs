//! Whole-model forward checks for the Llama-style decoder.
//!
//! A tiny random checkpoint is written to a temp dir and loaded through
//! [`Lfm2ModelBuilder`](ferrotensor_graph::model_builder::Lfm2ModelBuilder).
//! Its logits are compared with a plain f32 CPU implementation of the same
//! decoder written below from the math and the config (it calls no
//! Ferrotensor op), and with itself under changes that must not matter.
//!
//! These need a GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test --release -p ferrotensor-graph --test model_correctness -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{DType, UnifiedTensor};
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::Linear;
use std::collections::HashMap;

/// Tokens of the test prompt (every id is in the vocabulary).
const IDS: [u32; 8] = [3, 17, 5, 29, 0, 11, 23, 8];

/// Logits agree with the CPU reference within this absolute error.
///
/// Both sides compute in f32 with the same weights, so they differ only in
/// summation order (blocked matmuls, online softmax) and in the GPU's
/// `exp`, `pow`, `sin` and `cos`. Each of those is off by a few f32 ulps
/// (about 1e-7 relative); through two layers the logits, which are of order
/// 1, move by about 1e-5. 1e-3 leaves plenty of room for that, while any
/// real mistake (wrong RoPE pairing or base, wrong epsilon, missing scale,
/// wrong head grouping, a lost residual) moves logits by 1e-2 or more.
const REFERENCE_TOL: f32 = 1e-3;

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

fn get<'a>(weights: &'a Weights, name: &str) -> &'a [f32] {
    &weights
        .get(name)
        .unwrap_or_else(|| panic!("missing weight {name}"))
        .1
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y = W x` for every row `x` of `rows`, with `W` in Hugging Face layout
/// `[out, in]`.
fn linear(rows: &[f32], w: &[f32], in_dim: usize) -> Vec<f32> {
    rows.chunks(in_dim)
        .flat_map(|x| w.chunks(in_dim).map(move |w_row| dot(x, w_row)))
        .collect()
}

/// `x / sqrt(mean(x^2) + eps) * g` for every row `x`.
fn rms_norm(rows: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    rows.chunks(g.len())
        .flat_map(|x| {
            let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (mean_square + eps).sqrt();
            x.iter().zip(g).map(move |(v, w)| v * inv * w)
        })
        .collect()
}

/// Rotary embedding in the Llama "rotate half" form: inside each head, the
/// pair `(x[i], x[i + d/2])` is rotated by `pos * theta^(-2i/d)`. `rows` is
/// `[seq, heads * d]` and token `t` sits at position `t`.
fn rope(rows: &mut [f32], heads: usize, d: usize, theta: f32) {
    let half = d / 2;
    for (pos, token) in rows.chunks_mut(heads * d).enumerate() {
        for head in token.chunks_mut(d) {
            let (first, second) = head.split_at_mut(half);
            for (i, (x1, x2)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
                let angle = pos as f32 * theta.powf(-2.0 * i as f32 / d as f32);
                let (sin, cos) = angle.sin_cos();
                let (a, b) = (*x1, *x2);
                *x1 = a * cos - b * sin;
                *x2 = a * sin + b * cos;
            }
        }
    }
}

/// Causal attention, `softmax(q k^T / sqrt(d)) v` per head. `q` is
/// `[seq, HEADS * d]`, `k` and `v` are `[seq, KV_HEADS * d]`; query head `h`
/// reads key/value head `h / (HEADS / KV_HEADS)` (grouped-query attention).
fn attention(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let d = HEAD_DIM;
    let group = HEADS / KV_HEADS;
    let seq = q.len() / (HEADS * d);
    let scale = 1.0 / (d as f32).sqrt();
    let kv_row = |t: usize, kv_head: usize| (t * KV_HEADS + kv_head) * d;
    let mut out = vec![0.0; q.len()];
    for t in 0..seq {
        for head in 0..HEADS {
            let kv_head = head / group;
            let q_start = (t * HEADS + head) * d;
            let query = &q[q_start..q_start + d];
            let scores: Vec<f32> = (0..=t)
                .map(|j| dot(query, &k[kv_row(j, kv_head)..kv_row(j, kv_head) + d]) * scale)
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = exps.iter().sum();
            let o = &mut out[q_start..q_start + d];
            for (j, e) in exps.iter().enumerate() {
                let value = &v[kv_row(j, kv_head)..kv_row(j, kv_head) + d];
                for (oi, vi) in o.iter_mut().zip(value) {
                    *oi += e / total * vi;
                }
            }
        }
    }
    out
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn add_in_place(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Logits `[seq, VOCAB]` of the decoder for `ids`, the first at position 0.
fn reference_forward(w: &Weights, ids: &[u32]) -> Vec<f32> {
    let embed = get(w, "model.embed_tokens.weight");
    let mut x: Vec<f32> = ids
        .iter()
        .flat_map(|&id| embed[id as usize * HIDDEN..(id as usize + 1) * HIDDEN].to_vec())
        .collect();
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let lw = |name: &str| get(w, &format!("{p}.{name}.weight"));

        let h = rms_norm(&x, lw("input_layernorm"), EPS);
        let mut q = linear(&h, lw("self_attn.q_proj"), HIDDEN);
        let mut k = linear(&h, lw("self_attn.k_proj"), HIDDEN);
        let v = linear(&h, lw("self_attn.v_proj"), HIDDEN);
        rope(&mut q, HEADS, HEAD_DIM, THETA);
        rope(&mut k, KV_HEADS, HEAD_DIM, THETA);
        let attn = attention(&q, &k, &v);
        add_in_place(
            &mut x,
            &linear(&attn, lw("self_attn.o_proj"), HEADS * HEAD_DIM),
        );

        let h = rms_norm(&x, lw("post_attention_layernorm"), EPS);
        let gate = linear(&h, lw("mlp.gate_proj"), HIDDEN);
        let up = linear(&h, lw("mlp.up_proj"), HIDDEN);
        let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
        add_in_place(&mut x, &linear(&act, lw("mlp.down_proj"), FFN));
    }
    let h = rms_norm(&x, get(w, "model.norm.weight"), EPS);
    linear(&h, get(w, "lm_head.weight"), HIDDEN)
}

/// LoRA factors by layer prefix: `a` is `[in, rank]`, `b` is `[rank, out]`.
struct Adapter {
    a: Vec<f32>,
    b: Vec<f32>,
}

/// `weights` with every adapter folded in: `W'[o][i] = W[o][i] +
/// (alpha / rank) * sum_r a[i][r] * b[r][o]`. This is the update that
/// `Linear` documents (`x @ a @ b * alpha / rank`), written as a weight.
fn merge_adapters(
    weights: &Weights,
    adapters: &HashMap<String, Adapter>,
    rank: usize,
    alpha: f32,
) -> Weights {
    let scale = alpha / rank as f32;
    let mut merged = weights.clone();
    for (prefix, adapter) in adapters {
        let (shape, w) = merged
            .get_mut(&format!("{prefix}.weight"))
            .unwrap_or_else(|| panic!("no base weight for {prefix}"));
        let (out_dim, in_dim) = (shape[0], shape[1]);
        for (o, w_row) in w.chunks_mut(in_dim).enumerate() {
            for (i, w_oi) in w_row.iter_mut().enumerate() {
                let delta: f32 = (0..rank)
                    .map(|r| adapter.a[i * rank + r] * adapter.b[r * out_dim + o])
                    .sum();
                *w_oi += scale * delta;
            }
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn host(gpu: &Gpu, t: &Tensor) -> Vec<f32> {
    pollster::block_on(t.data.to_vec(&gpu.ctx))
}

/// Logits of `model` for `ids`, checked to be `[1, ids.len(), VOCAB]`.
fn logits(gpu: &Gpu, model: &Lfm2Model, ids: &[u32]) -> Vec<f32> {
    let out = model.forward(&gpu.engine, ids, 0, None, None);
    assert_eq!(out.data.shape, vec![1, ids.len(), VOCAB]);
    host(gpu, &out)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{what}: non-finite output"
    );
    let diff = max_abs_diff(actual, expected);
    assert!(diff <= tol, "{what}: max abs diff {diff} > {tol}");
}

/// Every `Linear` of the model that can carry an adapter, with the weight
/// name prefix it was loaded from.
fn linears_mut(model: &mut Lfm2Model) -> Vec<(String, &mut Linear)> {
    let mut out = Vec::new();
    for (l, layer) in model.layers.iter_mut().enumerate() {
        let p = format!("model.layers.{l}");
        if let Lfm2LayerOp::Attention(attn) = &mut layer.op {
            out.push((format!("{p}.self_attn.q_proj"), &mut attn.q_proj));
            out.push((format!("{p}.self_attn.k_proj"), &mut attn.k_proj));
            out.push((format!("{p}.self_attn.v_proj"), &mut attn.v_proj));
            out.push((format!("{p}.self_attn.o_proj"), &mut attn.o_proj));
        }
        out.push((format!("{p}.mlp.gate_proj"), &mut layer.mlp.gate_proj));
        out.push((format!("{p}.mlp.up_proj"), &mut layer.mlp.up_proj));
        out.push((format!("{p}.mlp.down_proj"), &mut layer.mlp.down_proj));
    }
    out.push(("lm_head".to_string(), &mut model.lm_head));
    out
}

/// Load the fixture with the given options.
fn setup(tag: &str, lora: bool, f16_weights: bool) -> (TempDir, Gpu, Weights, Lfm2Model) {
    let dir = TempDir::new(tag);
    let weights = tiny_llama_weights();
    write_tiny_llama(dir.path(), &weights);
    let gpu = gpu();
    let model = build_model(&gpu, dir.path(), lora.then(lora_everywhere), f16_weights);
    (dir, gpu, weights, model)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a GPU"]
fn full_forward_matches_a_cpu_reference() {
    let (_dir, gpu, weights, model) = setup("reference", false, false);
    assert!(model.supports_kv_cache(), "every block should be attention");
    let actual = logits(&gpu, &model, &IDS);
    let expected = reference_forward(&weights, &IDS);
    assert_close(&actual, &expected, REFERENCE_TOL, "logits vs CPU reference");
}

#[test]
#[ignore = "needs a GPU"]
fn earlier_logits_do_not_depend_on_later_tokens() {
    let (_dir, gpu, _weights, model) = setup("causal", false, false);
    let mut changed = IDS;
    changed[IDS.len() - 1] = (IDS[IDS.len() - 1] + 1) % VOCAB as u32;
    let a = logits(&gpu, &model, &IDS);
    let b = logits(&gpu, &model, &changed);
    let before = (IDS.len() - 1) * VOCAB;
    // Attention is causal and every other op works per token, so the rows
    // before the changed token run the exact same arithmetic: bit-identical.
    assert_eq!(
        a[..before].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b[..before].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "logits before the last position changed"
    );
    // And the last position does see its own token.
    assert!(
        max_abs_diff(&a[before..], &b[before..]) > 1e-3,
        "the last position did not react to its token"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn batched_forward_matches_each_sequence_alone() {
    let (_dir, gpu, _weights, model) = setup("batch", false, false);
    assert!(model.supports_batching());
    let first: [u32; 8] = IDS;
    let second: [u32; 8] = [30, 2, 19, 7, 7, 26, 1, 14];
    let batched = model.forward_batch(&gpu.engine, &[&first[..], &second[..]]);
    assert_eq!(batched.data.shape, vec![2, IDS.len(), VOCAB]);
    let batched = host(&gpu, &batched);
    let (b0, b1) = batched.split_at(IDS.len() * VOCAB);
    // Same math, only the number of rows per kernel launch changes, which
    // may change how sums are blocked; allow a little rounding.
    assert_close(b0, &logits(&gpu, &model, &first), 1e-4, "batch row 0");
    assert_close(b1, &logits(&gpu, &model, &second), 1e-4, "batch row 1");
}

#[test]
#[ignore = "needs a GPU"]
fn kv_cache_decoding_matches_the_full_forward() {
    let (_dir, gpu, _weights, model) = setup("kv-cache", false, false);
    let full = logits(&gpu, &model, &IDS);
    let mut cache = model.new_kv_cache(IDS.len()).expect("all-attention model");
    // A prompt of 4 tokens, then one token at a time.
    let prompt = 4;
    let mut decoded = host(
        &gpu,
        &model.forward(&gpu.engine, &IDS[..prompt], 0, Some(&mut cache), None),
    );
    for pos in prompt..IDS.len() {
        let step = model.forward(&gpu.engine, &IDS[pos..=pos], pos, Some(&mut cache), None);
        decoded.extend(host(&gpu, &step));
    }
    // A different attention launch (cache offsets) over the same values.
    assert_close(&decoded, &full, 1e-4, "cached decoding vs full forward");
}

#[test]
#[ignore = "needs a GPU"]
fn fresh_lora_adapters_do_not_change_the_output() {
    let (_dir, gpu, weights, with_lora) = setup("lora-fresh", true, false);
    assert_eq!(
        with_lora.lora_params().len(),
        2 * (LAYERS * 7 + 1),
        "an adapter (A and B) on every linear layer and the LM head"
    );
    let dir = TempDir::new("lora-fresh-base");
    write_tiny_llama(dir.path(), &weights);
    let base = build_model(&gpu, dir.path(), None, false);
    let a = logits(&gpu, &with_lora, &IDS);
    let b = logits(&gpu, &base, &IDS);
    // B starts at zero, so the adapter adds exactly 0.0 to each output and
    // the base matmuls are the same kernels on the same data.
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "fresh adapters changed the logits"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn trained_like_lora_delta_matches_the_merged_weight() {
    let (_dir, gpu, weights, mut model) = setup("lora-merged", true, false);
    let config = lora_everywhere();
    let mut adapters = HashMap::new();
    let mut seed = 1000;
    for (prefix, linear) in linears_mut(&mut model) {
        let lora = linear
            .lora
            .as_mut()
            .unwrap_or_else(|| panic!("{prefix} has no adapter"));
        assert_eq!((lora.rank, lora.alpha), (config.rank, config.alpha));
        let [in_dim, rank] = lora.a.data.shape[..] else {
            panic!("{prefix}: A should be 2-D");
        };
        let out_dim = lora.b.data.shape[1];
        seed += 2;
        let a = pseudo_random(in_dim * rank, seed, 0.3);
        let b = pseudo_random(rank * out_dim, seed + 1, 0.3);
        lora.a = Tensor::new(
            gpu.ctx.clone(),
            UnifiedTensor::new(&gpu.ctx, &a, &[in_dim, rank]),
        );
        lora.b = Tensor::new(
            gpu.ctx.clone(),
            UnifiedTensor::new(&gpu.ctx, &b, &[rank, out_dim]),
        );
        adapters.insert(prefix, Adapter { a, b });
    }
    assert_eq!(adapters.len(), LAYERS * 7 + 1);

    let actual = logits(&gpu, &model, &IDS);
    let merged = merge_adapters(&weights, &adapters, config.rank, config.alpha);
    let expected = reference_forward(&merged, &IDS);
    assert_close(
        &actual,
        &expected,
        REFERENCE_TOL,
        "LoRA logits vs merged weights",
    );
    // The adapters must matter, or the comparison above proves nothing.
    let base = reference_forward(&weights, &IDS);
    assert!(
        max_abs_diff(&expected, &base) > 0.1,
        "the adapters barely change the logits"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn forward_is_deterministic() {
    let (_dir, gpu, _weights, model) = setup("deterministic", false, false);
    let a = logits(&gpu, &model, &IDS);
    let b = logits(&gpu, &model, &IDS);
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "two runs gave different logits"
    );
}

/// `(shape, values)` of every f32 tensor in a safetensors file.
fn read_safetensors(path: &std::path::Path) -> Weights {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let file = safetensors::SafeTensors::deserialize(&bytes).expect("safetensors");
    let mut out = Weights::new();
    for (name, view) in file.tensors() {
        assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name} is not f32");
        let values = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        out.insert(name, (view.shape().to_vec(), values));
    }
    out
}

/// The PyTorch fixture of the gradient-parity tests in ferrotensor-trainer
/// (`scripts/gen_gradient_fixtures.py`) describes this file's model: same
/// config, bit-identical weights, and logits (without and with its LoRA
/// adapters) that equal the CPU reference above. This ties the PyTorch
/// mirror to an independent forward pass. CPU only, so it always runs.
#[test]
fn pytorch_gradient_fixture_matches_the_cpu_reference() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gradient_parity");

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).expect("config"))
            .expect("config json");
    for (key, value) in [
        ("hidden_size", HIDDEN),
        ("intermediate_size", FFN),
        ("num_attention_heads", HEADS),
        ("num_key_value_heads", KV_HEADS),
        ("num_hidden_layers", LAYERS),
        ("vocab_size", VOCAB),
    ] {
        assert_eq!(config[key].as_u64(), Some(value as u64), "config {key}");
    }
    assert_eq!(config["rms_norm_eps"].as_f64().map(|v| v as f32), Some(EPS));
    assert_eq!(config["rope_theta"].as_f64().map(|v| v as f32), Some(THETA));
    assert_eq!(config["tie_word_embeddings"].as_bool(), Some(false));

    let weights = tiny_llama_weights();
    let fixture = read_safetensors(&dir.join("model.safetensors"));
    let mut names: Vec<&String> = fixture.keys().collect();
    names.sort();
    let mut expected_names: Vec<&String> = weights.keys().collect();
    expected_names.sort();
    assert_eq!(names, expected_names, "fixture weight names");
    for (name, (shape, values)) in &weights {
        let (f_shape, f_values) = &fixture[name];
        assert_eq!(f_shape, shape, "{name} shape");
        assert!(
            f_values
                .iter()
                .zip(values)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: fixture weights differ from tiny_llama_weights()"
        );
    }

    let cases: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("cases.json")).expect("cases"))
            .expect("cases json");
    let first_row: Vec<u32> = cases["cases"]["single"]["rows"][0]
        .as_array()
        .expect("rows")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    assert_eq!(first_row, IDS, "the fixture's first row is IDS");
    let config = lora_everywhere();
    assert_eq!(
        cases["lora_rank"].as_u64(),
        Some(config.rank as u64),
        "LoRA rank"
    );
    assert_eq!(
        cases["lora_alpha"].as_f64().map(|v| v as f32),
        Some(config.alpha),
        "LoRA alpha"
    );

    let raw = read_safetensors(&dir.join("adapters.safetensors"));
    let adapters: HashMap<String, Adapter> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".lora_a"))
        .map(|prefix| {
            let a = raw[&format!("{prefix}.lora_a")].1.clone();
            let b = raw[&format!("{prefix}.lora_b")].1.clone();
            (prefix.to_string(), Adapter { a, b })
        })
        .collect();
    assert_eq!(
        adapters.len(),
        LAYERS * 7 + 1,
        "an adapter per linear layer"
    );

    // Two f32 CPU implementations of the same math: they differ only in
    // summation order and libm, about 1e-6 on logits of order 1 (both are
    // within 1.5e-6 of an f64 evaluation).
    let tol = 1e-4;
    let expected = read_safetensors(&dir.join("expected.safetensors"));
    let base = reference_forward(&weights, &IDS);
    assert_close(
        &expected["logits.base"].1,
        &base,
        tol,
        "PyTorch base logits",
    );
    let merged = merge_adapters(&weights, &adapters, config.rank, config.alpha);
    let with_lora = reference_forward(&merged, &IDS);
    assert_close(
        &expected["logits.single"].1,
        &with_lora,
        tol,
        "PyTorch LoRA logits",
    );
}

#[test]
#[ignore = "needs a GPU"]
fn f16_base_weights_stay_close_to_f32() {
    let (_dir, gpu, weights, model) = setup("f16", false, true);
    assert_eq!(model.lm_head.weight.data.dtype, DType::FP16);
    let actual = logits(&gpu, &model, &IDS);
    let expected = reference_forward(&weights, &IDS);
    // Rounding a weight to f16 changes it by at most 2^-11 (about 5e-4) of
    // its value. A 32- or 64-term dot product of such errors with random
    // signs moves an output of order 1 by a few 1e-4, and about ten such
    // products lie between the input and the logits. That gives errors of
    // order 1e-3, so 1e-2 is a safe bound that still catches a broken f16
    // path (wrong unpacking or a missing transpose gives errors of order 1).
    assert_close(&actual, &expected, 1e-2, "f16 logits vs f32 reference");
}

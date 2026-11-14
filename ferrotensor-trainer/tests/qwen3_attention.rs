//! Qwen3-style attention (per-head RMSNorm of queries and keys before RoPE)
//! on a tiny random model, checked against an f64 CPU reference of the whole
//! forward pass written here, independently of the GPU code.
//!
//! The GPU tests are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test qwen3_attention -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::Architecture;
use ferrotensor_graph::Attention;
use ferrotensor_loader::ModelLoader;
use std::collections::HashMap;
use std::path::Path;

/// Checkpoint tensors as f64, keyed by name: `(shape, values)`.
type Weights = HashMap<String, (Vec<usize>, Vec<f64>)>;

fn read_weights(dir: &Path) -> Weights {
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("read weights");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse weights");
    st.tensors()
        .into_iter()
        .map(|(name, view)| {
            let (words, _) = view.data().as_chunks::<4>();
            let values = words
                .iter()
                .map(|c| f32::from_le_bytes(*c) as f64)
                .collect();
            (name, (view.shape().to_vec(), values))
        })
        .collect()
}

/// A LoRA adapter as the GPU model holds it: `a` `[in, rank]`, `b`
/// `[rank, out]`, update `x @ a @ b * scale`.
#[derive(Clone)]
struct RefLora {
    a: Vec<f64>,
    b: Vec<f64>,
    rank: usize,
    scale: f64,
}

/// The f64 reference decoder, following Hugging Face `Qwen3ForCausalLM` /
/// `LlamaForCausalLM`.
struct Reference<'a> {
    w: &'a Weights,
    /// Adapters keyed by projection prefix, e.g. `model.layers.0.self_attn.q_proj`.
    lora: HashMap<String, RefLora>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
    theta: f64,
    /// Apply `q_norm`/`k_norm` (off only for the control that shows they matter).
    qk_norm: bool,
}

impl Reference<'_> {
    fn get(&self, name: &str) -> &[f64] {
        &self
            .w
            .get(name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
    }

    /// `x @ W^T` for a PyTorch `[out, in]` weight, plus the adapter if any.
    fn linear(&self, x: &[f64], weight: &str, lora: Option<&str>) -> Vec<f64> {
        let (shape, w) = self
            .w
            .get(weight)
            .unwrap_or_else(|| panic!("missing {weight}"));
        let (out, inp) = (shape[0], shape[1]);
        let rows = x.len() / inp;
        let mut y = vec![0.0; rows * out];
        for r in 0..rows {
            let xr = &x[r * inp..(r + 1) * inp];
            for o in 0..out {
                let wo = &w[o * inp..(o + 1) * inp];
                y[r * out + o] = xr.iter().zip(wo).map(|(a, b)| a * b).sum();
            }
            if let Some(l) = lora.and_then(|p| self.lora.get(p)) {
                let t: Vec<f64> = (0..l.rank)
                    .map(|k| (0..inp).map(|i| xr[i] * l.a[i * l.rank + k]).sum())
                    .collect();
                for o in 0..out {
                    let d: f64 = (0..l.rank).map(|k| t[k] * l.b[k * out + o]).sum();
                    y[r * out + o] += l.scale * d;
                }
            }
        }
        y
    }

    /// RMSNorm over consecutive chunks of `w.len()` elements.
    fn rms_norm(&self, x: &[f64], w: &[f64]) -> Vec<f64> {
        let d = w.len();
        let mut y = Vec::with_capacity(x.len());
        for row in x.chunks(d) {
            let ms = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let inv = 1.0 / (ms + self.eps).sqrt();
            y.extend(row.iter().zip(w).map(|(v, g)| v * inv * g));
        }
        y
    }

    /// Rotate-half RoPE of `[seq, heads * head_dim]` at positions `0..seq`.
    fn rope(&self, x: &mut [f64], heads: usize) {
        let hd = self.head_dim;
        let half = hd / 2;
        let seq = x.len() / (heads * hd);
        for t in 0..seq {
            for h in 0..heads {
                let base = (t * heads + h) * hd;
                for i in 0..half {
                    let angle = t as f64 * self.theta.powf(-2.0 * i as f64 / hd as f64);
                    let (s, c) = angle.sin_cos();
                    let (a, b) = (x[base + i], x[base + i + half]);
                    x[base + i] = a * c - b * s;
                    x[base + i + half] = a * s + b * c;
                }
            }
        }
    }

    /// Causal grouped-query attention: query head `h` reads key/value head
    /// `h / (heads / kv_heads)`, as `repeat_kv` arranges them.
    fn attention(&self, q: &[f64], k: &[f64], v: &[f64]) -> Vec<f64> {
        let (hd, nh, nkv) = (self.head_dim, self.heads, self.kv_heads);
        let seq = q.len() / (nh * hd);
        let scale = 1.0 / (hd as f64).sqrt();
        let mut out = vec![0.0; seq * nh * hd];
        for h in 0..nh {
            let g = h / (nh / nkv);
            for t in 0..seq {
                let qv = &q[(t * nh + h) * hd..][..hd];
                let scores: Vec<f64> = (0..=t)
                    .map(|u| {
                        let kv = &k[(u * nkv + g) * hd..][..hd];
                        qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f64>() * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (u, e) in exps.iter().enumerate() {
                    let vv = &v[(u * nkv + g) * hd..][..hd];
                    let dst = &mut out[(t * nh + h) * hd..][..hd];
                    dst.iter_mut().zip(vv).for_each(|(o, x)| *o += e / sum * x);
                }
            }
        }
        out
    }

    /// Logits `[seq, vocab]` for `ids`, starting at position 0.
    fn forward(&self, ids: &[u32]) -> Vec<f64> {
        let (eshape, embed) = &self.w["model.embed_tokens.weight"];
        let hidden = eshape[1];
        let mut x: Vec<f64> = ids
            .iter()
            .flat_map(|&id| embed[id as usize * hidden..][..hidden].iter().copied())
            .collect();
        for l in 0..LAYERS {
            let p = format!("model.layers.{l}");
            let a = format!("{p}.self_attn");
            let h = self.rms_norm(&x, self.get(&format!("{p}.input_layernorm.weight")));
            let proj = |name: &str, input: &[f64]| {
                let prefix = format!("{a}.{name}");
                self.linear(input, &format!("{prefix}.weight"), Some(&prefix))
            };
            let mut q = proj("q_proj", &h);
            let mut k = proj("k_proj", &h);
            let v = proj("v_proj", &h);
            if self.qk_norm {
                q = self.rms_norm(&q, self.get(&format!("{a}.q_norm.weight")));
                k = self.rms_norm(&k, self.get(&format!("{a}.k_norm.weight")));
            }
            self.rope(&mut q, self.heads);
            self.rope(&mut k, self.kv_heads);
            let attn = self.attention(&q, &k, &v);
            let o = proj("o_proj", &attn);
            x.iter_mut().zip(&o).for_each(|(x, o)| *x += o);

            let h = self.rms_norm(
                &x,
                self.get(&format!("{p}.post_attention_layernorm.weight")),
            );
            let gate = self.linear(&h, &format!("{p}.mlp.gate_proj.weight"), None);
            let up = self.linear(&h, &format!("{p}.mlp.up_proj.weight"), None);
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let down = self.linear(&act, &format!("{p}.mlp.down_proj.weight"), None);
            x.iter_mut().zip(&down).for_each(|(x, d)| *x += d);
        }
        let x = self.rms_norm(&x, self.get("model.norm.weight"));
        let head = if self.w.contains_key("lm_head.weight") {
            "lm_head.weight"
        } else {
            "model.embed_tokens.weight"
        };
        self.linear(&x, head, None)
    }

    /// Mean cross-entropy of next-token `targets` over every position.
    fn loss(&self, ids: &[u32], targets: &[u32]) -> f64 {
        let logits = self.forward(ids);
        let vocab = logits.len() / ids.len();
        let total: f64 = logits
            .chunks(vocab)
            .zip(targets)
            .map(|(row, &t)| {
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
                lse - row[t as usize]
            })
            .sum();
        total / ids.len() as f64
    }
}

/// The reference for the checkpoint in `dir`, with the model's adapters.
fn reference<'a>(gpu: &Gpu, dir: &Path, w: &'a Weights, model: &Lfm2Model) -> Reference<'a> {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let heads = config.num_attention_heads;
    let cfg = lora_config();
    let named = model.named_lora_params();
    let lora = named
        .keys()
        .filter_map(|k| k.strip_suffix(".lora_A.weight"))
        .map(|prefix| {
            let to64 = |name: String| -> Vec<f64> {
                host(gpu, &named[&name]).iter().map(|&v| v as f64).collect()
            };
            let l = RefLora {
                a: to64(format!("{prefix}.lora_A.weight")),
                b: to64(format!("{prefix}.lora_B.weight")),
                rank: cfg.rank,
                scale: cfg.alpha as f64 / cfg.rank as f64,
            };
            (prefix.to_string(), l)
        })
        .collect();
    Reference {
        w,
        lora,
        heads,
        kv_heads: config.num_key_value_heads.unwrap_or(heads),
        head_dim: config.head_dim.unwrap_or(config.hidden_size / heads),
        eps: config.norm_eps as f64,
        theta: config.rope_theta.unwrap_or(10000.0),
        qk_norm: true,
    }
}

fn attention(model: &Lfm2Model, layer: usize) -> &Attention {
    match &model.layers[layer].op {
        Lfm2LayerOp::Attention(a) => a,
        Lfm2LayerOp::Conv(_) => panic!("layer {layer} is not attention"),
    }
}

fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - y).abs())
        .fold(0.0, f64::max)
}

/// Fill every LoRA `B` (zero at initialisation) with small random values,
/// so the adapters change the output and `A` gets a non-zero gradient.
fn randomise_lora_b(gpu: &Gpu, model: &Lfm2Model) {
    let mut names: Vec<String> = model.named_lora_params().into_keys().collect();
    names.sort();
    let named = model.named_lora_params();
    for (i, name) in names.iter().filter(|n| n.contains("lora_B")).enumerate() {
        let t = &named[name];
        let n: usize = t.data.shape.iter().product();
        let values = pseudo_random(n, 500 + i as u64, 0.1);
        t.data.write_data(&gpu.ctx, bytemuck::cast_slice(&values));
    }
}

#[test]
fn qwen3_config_with_layer_types_is_llama_style() {
    let dir = TempDir::new("qwen3-detect");
    write_tiny_qwen3(dir.path());
    let config = ModelLoader::load_config(dir.path().join("config.json")).expect("config");
    assert_eq!(config.layer_types.len(), LAYERS);
    assert_eq!(Architecture::detect(&config), Architecture::Llama);
}

#[test]
#[ignore = "needs a GPU"]
fn qwen3_norms_are_loaded_frozen_and_named() {
    let dir = TempDir::new("qwen3-load");
    write_tiny_qwen3(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let named = model.named_params();
    for l in 0..LAYERS {
        let a = attention(&model, l);
        assert_eq!(a.head_dim, QWEN3_HEAD_DIM);
        for (norm, name) in [(&a.q_norm, "q_norm"), (&a.k_norm, "k_norm")] {
            let norm = norm.as_ref().expect("norm loaded");
            assert_eq!(norm.weight.data.shape, [QWEN3_HEAD_DIM]);
            assert!(!norm.weight.requires_grad, "{name} should be frozen");
            assert!(named.contains_key(&format!("model.layers.{l}.self_attn.{name}.weight")));
        }
    }

    let llama_dir = TempDir::new("qwen3-load-llama");
    write_tiny_llama(llama_dir.path());
    let llama = build_model(&gpu, llama_dir.path());
    for l in 0..LAYERS {
        let a = attention(&llama, l);
        assert!(a.q_norm.is_none() && a.k_norm.is_none());
    }
    assert!(!llama
        .named_params()
        .keys()
        .any(|k| k.contains(".q_norm.") || k.contains(".k_norm.")));
}

#[test]
#[ignore = "needs a GPU"]
fn qwen3_logits_match_the_cpu_reference() {
    let dir = TempDir::new("qwen3-ref");
    write_tiny_qwen3(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    randomise_lora_b(&gpu, &model);
    let w = read_weights(dir.path());
    let mut r = reference(&gpu, dir.path(), &w, &model);

    let seq = pattern_sequence();
    let ids = &seq[..12];
    let gpu_logits = host(&gpu, &model.forward(&gpu.engine, ids, 0, None, None));
    let want = r.forward(ids);
    let err = max_abs_diff(&gpu_logits, &want);
    assert!(err < 1e-4, "GPU vs f64 reference: max |diff| {err}");

    // Control: without q/k norms the reference is far from the GPU logits,
    // so the comparison above does check the norms.
    r.qk_norm = false;
    let without = max_abs_diff(&gpu_logits, &r.forward(ids));
    assert!(
        without > 1e-2,
        "q/k norms barely change the logits: {without}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn llama_logits_match_the_cpu_reference() {
    let dir = TempDir::new("qwen3-ref-llama");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    randomise_lora_b(&gpu, &model);
    let w = read_weights(dir.path());
    let mut r = reference(&gpu, dir.path(), &w, &model);
    r.qk_norm = false;

    let seq = pattern_sequence();
    let ids = &seq[..12];
    let gpu_logits = host(&gpu, &model.forward(&gpu.engine, ids, 0, None, None));
    let err = max_abs_diff(&gpu_logits, &r.forward(ids));
    assert!(err < 1e-4, "GPU vs f64 reference: max |diff| {err}");
}

#[test]
#[ignore = "needs a GPU"]
fn qwen3_kv_cache_decoding_matches_full_recompute() {
    let dir = TempDir::new("qwen3-kv");
    write_tiny_qwen3(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let seq = pattern_sequence();
    let full = host(&gpu, &model.forward(&gpu.engine, &seq[..10], 0, None, None));

    let mut cache = model.new_kv_cache(16).expect("qwen3 supports a kv cache");
    let prefix = host(
        &gpu,
        &model.forward(&gpu.engine, &seq[..6], 0, Some(&mut cache), None),
    );
    let mut cached = prefix[..6 * VOCAB].to_vec();
    for pos in 6..10 {
        let step = host(
            &gpu,
            &model.forward(&gpu.engine, &seq[pos..pos + 1], pos, Some(&mut cache), None),
        );
        cached.extend_from_slice(&step);
    }
    for (i, (x, y)) in full.iter().zip(&cached).enumerate() {
        assert!((x - y).abs() < 1e-3, "logit {i}: full {x} vs cached {y}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn qwen3_batched_forward_matches_one_sequence_at_a_time() {
    let dir = TempDir::new("qwen3-batch");
    write_tiny_qwen3(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    randomise_lora_b(&gpu, &model);
    let seq = pattern_sequence();
    let rows: [&[u32]; 3] = [&seq[..9], &seq[4..13], &seq[11..20]];
    let batched = host(&gpu, &model.forward_batch(&gpu.engine, &rows));
    let per_row = rows.len();
    assert_eq!(batched.len(), per_row * 9 * VOCAB);
    for (b, row) in rows.iter().enumerate() {
        let single = host(&gpu, &model.forward(&gpu.engine, row, 0, None, None));
        let got = &batched[b * 9 * VOCAB..(b + 1) * 9 * VOCAB];
        for (i, (x, y)) in single.iter().zip(got).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "row {b} logit {i}: single {x} vs batched {y}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn qwen3_lora_gradients_through_the_norms_match_finite_differences() {
    let dir = TempDir::new("qwen3-grad");
    write_tiny_qwen3(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    randomise_lora_b(&gpu, &model);
    let w = read_weights(dir.path());
    let r = reference(&gpu, dir.path(), &w, &model);

    let seq = pattern_sequence();
    let (ids, targets) = (&seq[..10], &seq[1..11]);
    let logits = model.forward(&gpu.engine, ids, 0, None, None);
    let loss = ferrotensor_trainer::masked_cross_entropy(&logits, targets, &[1; 10], &gpu.engine)
        .expect("loss");
    let want_loss = r.loss(ids, targets);
    assert!(
        (loss.value as f64 - want_loss).abs() < 1e-4,
        "loss: GPU {} vs reference {want_loss}",
        loss.value
    );
    loss.tensor.backward(&gpu.engine);

    // The norm weights are frozen base weights: no gradient.
    for l in 0..LAYERS {
        let a = attention(&model, l);
        for norm in [&a.q_norm, &a.k_norm] {
            let norm = norm.as_ref().expect("norm loaded");
            assert!(norm.weight.grad.borrow().is_none());
        }
    }

    let named = model.named_lora_params();
    let h = 1e-5;
    let mut checked = 0;
    for l in 0..LAYERS {
        for proj in ["q_proj", "k_proj"] {
            let prefix = format!("model.layers.{l}.self_attn.{proj}");
            for part in ["A", "B"] {
                let name = format!("{prefix}.lora_{part}.weight");
                let grad = named[&name].grad.borrow().clone().expect("LoRA gradient");
                let grad = pollster::block_on(grad.to_vec(&gpu.ctx));
                // The three largest entries, where a wrong gradient shows
                // most clearly against the finite-difference noise.
                let mut order: Vec<usize> = (0..grad.len()).collect();
                order.sort_by(|&i, &j| grad[j].abs().total_cmp(&grad[i].abs()));
                for &i in &order[..3] {
                    let fd = {
                        let mut plus = Reference {
                            lora: r.lora.clone(),
                            ..r
                        };
                        let bump = |plus: &mut Reference<'_>, delta: f64| {
                            let l = plus.lora.get_mut(&prefix).expect("adapter");
                            let v = if part == "A" { &mut l.a } else { &mut l.b };
                            v[i] += delta;
                        };
                        bump(&mut plus, h);
                        let up = plus.loss(ids, targets);
                        bump(&mut plus, -2.0 * h);
                        let down = plus.loss(ids, targets);
                        (up - down) / (2.0 * h)
                    };
                    let g = grad[i] as f64;
                    assert!(
                        fd.abs() > 1e-4,
                        "{name}[{i}]: gradient too small to check ({fd})"
                    );
                    assert!(
                        (g - fd).abs() <= 1e-6 + 2e-2 * fd.abs(),
                        "{name}[{i}]: GPU {g} vs finite difference {fd}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, LAYERS * 2 * 2 * 3);
}

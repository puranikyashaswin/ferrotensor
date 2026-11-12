//! Frozen base weights stored at 4 bits (`Lfm2ModelBuilder::with_q4_weights`).
//!
//! 4-bit weights change the model, so the reference is not the original
//! checkpoint: it is the same model loaded in f32 from a copy of the
//! checkpoint whose decoder-block weights were replaced on the host by
//! `q4::dequantize(q4::quantize(w))`. The 4-bit model must match that
//! reference to f32 tolerance in logits, loss, LoRA gradients, training
//! and generation. How far either is from the original weights is the
//! quantisation effect; it is printed, not asserted.
//!
//! Most tests need a GPU, so they are `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test q4_base_weights -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::memory::wgpu_buffer_size;
use ferrotensor_backend::{q4, DType, MemoryCategory, UnifiedTensor};
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::Lfm2ModelBuilder;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::memory::{tag_trainable, MemoryRecorder};
use std::collections::HashMap;
use std::io::Write as _;
use std::path::Path;

/// File-name suffixes of the weights the builder stores at 4 bits in the
/// tiny Llama-style model (every decoder-block projection; the tied head
/// is not one of them).
const QUANTISED: [&str; 7] = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
];

/// Written to the stream directly so it shows in the CI log.
fn report(line: &str) {
    let _ = writeln!(std::io::stderr(), "{line}");
}

fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Every tensor of `dir/model.safetensors` as `(name, shape, f32 values)`,
/// sorted by name.
fn read_tensors(dir: &Path) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(dir.join("model.safetensors")).expect("read weights");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse weights");
    let mut out: Vec<(String, Vec<usize>, Vec<f32>)> = st
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}");
            let values = view
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (name, view.shape().to_vec(), values)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Writes `config.json` (copied from `src`) and `tensors` as F32
/// safetensors into `dst`, the way `common::write_tiny_llama` does.
fn write_model(src: &Path, dst: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    std::fs::copy(src.join("config.json"), dst.join("config.json")).expect("copy config");
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(n, s, v)| {
            (
                n.clone(),
                s.clone(),
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            )
        })
        .collect();
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = bytes
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                    .expect("tensor view"),
            )
        })
        .collect();
    let data = safetensors::tensor::serialize(views, &None).expect("serialize");
    std::fs::write(dst.join("model.safetensors"), data).expect("write weights");
}

/// The builder's 4-bit round trip of a PyTorch-order `[out, in]` weight:
/// transposed to the logical `[in, out]`, quantised and decoded there, and
/// transposed back.
fn round_trip(values: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let logical = transpose(values, out, inp);
    let words = q4::quantize(&logical, inp, out).expect("quantise");
    transpose(&q4::dequantize(&words, inp, out), inp, out)
}

/// Copies the model in `src` to `dst` with every decoder-block projection
/// replaced by its 4-bit round trip; every other tensor is unchanged.
/// Returns how many tensors were replaced.
fn write_dequantised_copy(src: &Path, dst: &Path) -> usize {
    let mut replaced = 0;
    let tensors: Vec<_> = read_tensors(src)
        .into_iter()
        .map(|(name, shape, values)| {
            if QUANTISED.iter().any(|s| name.ends_with(s)) {
                let [out, inp] = shape[..] else {
                    panic!("{name} is not 2-D: {shape:?}");
                };
                replaced += 1;
                let values = round_trip(&values, out, inp);
                (name, shape, values)
            } else {
                (name, shape, values)
            }
        })
        .collect();
    write_model(src, dst, &tensors);
    replaced
}

fn builder<'a>(gpu: &Gpu, weights: &'a WeightsContext, dir: &Path) -> Lfm2ModelBuilder<'a> {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), weights, config)
}

fn load_weights(dir: &Path) -> WeightsContext {
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    weights
}

/// The tiny model in `dir` with LoRA on q/k/v/o and f32 weights, the
/// decoder blocks at 4 bits when `q4`.
fn build(gpu: &Gpu, dir: &Path, q4: bool) -> Lfm2Model {
    let weights = load_weights(dir);
    builder(gpu, &weights, dir)
        .with_lora(lora_config())
        .with_q4_weights(q4)
        .build()
        .expect("build")
}

/// Non-zero LoRA B matrices, so every LoRA A gradient is non-zero too.
fn set_lora_b(gpu: &Gpu, model: &Lfm2Model) {
    let (names, params) = sorted_lora_params(model);
    for (i, (name, t)) in names.iter().zip(&params).enumerate() {
        if !name.ends_with("lora_B.weight") {
            continue;
        }
        let numel = t.data.shape.iter().product();
        let bytes: Vec<u8> = pseudo_random(numel, 1000 + i as u64, 0.1)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        t.data.write_data(&gpu.ctx, &bytes);
    }
}

/// One forward and backward pass: logits, loss and the gradient of every
/// LoRA tensor (sorted by name).
struct Pass {
    logits: Vec<f32>,
    loss: f32,
    grads: Vec<(String, Option<Vec<f32>>)>,
}

/// Runs one row through `forward` or several equally long rows through
/// `forward_batch`, then the loss and its backward pass.
fn pass(gpu: &Gpu, model: &Lfm2Model, rows: &[Vec<u32>], targets: &[Vec<u32>]) -> Pass {
    let logits_t = if rows.len() == 1 {
        model.forward(&gpu.engine, &rows[0], 0, None, None)
    } else {
        let refs: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
        model.forward_batch(&gpu.engine, &refs)
    };
    let logits = host(gpu, &logits_t);
    let (tensor, loss): (Tensor, f32) = if rows.len() == 1 {
        let mask = vec![1u8; targets[0].len()];
        let l =
            ferrotensor_trainer::masked_cross_entropy(&logits_t, &targets[0], &mask, &gpu.engine)
                .expect("loss");
        (l.tensor, l.value)
    } else {
        let flat: Vec<u32> = targets.iter().flatten().copied().collect();
        let mask = vec![1u8; flat.len()];
        let l = ferrotensor_trainer::masked_cross_entropy_batch(
            &logits_t,
            &flat,
            &mask,
            rows.len(),
            &gpu.engine,
        )
        .expect("loss");
        (l.tensor, l.per_sequence.iter().map(|(v, _)| v).sum())
    };
    tensor.backward(&gpu.engine);
    let (names, params) = sorted_lora_params(model);
    let grads = names
        .into_iter()
        .zip(params)
        .map(|(name, t)| {
            let g = t
                .grad
                .borrow()
                .as_ref()
                .map(|g| pollster::block_on(g.to_vec(&gpu.ctx)));
            (name, g)
        })
        .collect();
    Pass {
        logits,
        loss,
        grads,
    }
}

/// Largest `|x - y| - rel * |y|` and its index.
fn worst(x: &[f32], y: &[f32], rel: f32) -> (f32, usize) {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .map(|(x, y)| (x - y).abs() - rel * y.abs())
        .zip(0..)
        .fold((f32::MIN, 0), |m, (d, i)| if d > m.0 { (d, i) } else { m })
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Asserts the 4-bit pass equals the dequantised-f32 pass to f32
/// tolerance and prints the largest differences.
fn assert_matches(label: &str, q: &Pass, r: &Pass) {
    let (d, at) = worst(&q.logits, &r.logits, 1e-3);
    assert!(
        d <= 1e-4,
        "{label}: logit {at}: 4-bit {} vs dequantised f32 {}",
        q.logits[at],
        r.logits[at]
    );
    assert!(
        (q.loss - r.loss).abs() <= 1e-4 * r.loss.abs(),
        "{label}: loss 4-bit {} vs dequantised f32 {}",
        q.loss,
        r.loss
    );
    assert_eq!(q.grads.len(), LAYERS * 4 * 2, "q/k/v/o adapters, A and B");
    assert_eq!(q.grads.len(), r.grads.len());
    let mut max_grad = 0.0f32;
    for ((name, x), (name_r, y)) in q.grads.iter().zip(&r.grads) {
        assert_eq!(name, name_r);
        let (Some(x), Some(y)) = (x, y) else {
            panic!(
                "{label}: {name}: gradient present 4-bit {}, dequantised f32 {}",
                x.is_some(),
                y.is_some()
            );
        };
        assert!(y.iter().any(|v| *v != 0.0), "{label}: {name} is all zero");
        let (d, at) = worst(x, y, 1e-3);
        assert!(
            d <= 1e-5,
            "{label}: {name}[{at}]: 4-bit {} vs dequantised f32 {}",
            x[at],
            y[at]
        );
        max_grad = max_grad.max(max_abs_diff(x, y));
    }
    report(&format!(
        "{label}: 4-bit vs dequantised f32: max |logit diff| {:.3e}, loss {} vs {} (diff {:.3e}), max |LoRA grad diff| {:.3e}",
        max_abs_diff(&q.logits, &r.logits),
        q.loss,
        r.loss,
        (q.loss - r.loss).abs(),
        max_grad
    ));
}

/// Tiny model in `a`, its dequantised copy in `b`.
fn fixtures(tag: &str) -> (TempDir, TempDir) {
    let a = TempDir::new(&format!("{tag}-orig"));
    let b = TempDir::new(&format!("{tag}-deq"));
    write_tiny_llama(a.path());
    assert_eq!(write_dequantised_copy(a.path(), b.path()), LAYERS * 7);
    (a, b)
}

fn one_row() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let seq = pattern_sequence();
    (vec![seq[..23].to_vec()], vec![seq[1..].to_vec()])
}

fn three_rows() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let seq = pattern_sequence();
    let rows = [0, 3, 6].map(|s| seq[s..s + 16].to_vec()).to_vec();
    let targets = [0, 3, 6].map(|s| seq[s + 1..s + 17].to_vec()).to_vec();
    (rows, targets)
}

fn compare_with_dequantised(tag: &str, rows: &[Vec<u32>], targets: &[Vec<u32>]) {
    let (a, b) = fixtures(tag);
    let gpu = gpu();
    let m_q4 = build(&gpu, a.path(), true);
    let m_ref = build(&gpu, b.path(), false);
    let m_orig = build(&gpu, a.path(), false);
    for m in [&m_q4, &m_ref, &m_orig] {
        set_lora_b(&gpu, m);
    }
    let q = pass(&gpu, &m_q4, rows, targets);
    let r = pass(&gpu, &m_ref, rows, targets);
    assert_matches(tag, &q, &r);
    let o = pass(&gpu, &m_orig, rows, targets);
    report(&format!(
        "{tag}: quantisation effect (not a check): loss 4-bit {} vs original f32 weights {} (diff {:+.3e}, {:+.3}%), max |logit diff| {:.3e}",
        q.loss,
        o.loss,
        q.loss - o.loss,
        100.0 * (q.loss - o.loss) / o.loss,
        max_abs_diff(&q.logits, &o.logits)
    ));
}

#[test]
#[ignore = "needs a GPU"]
fn q4_weights_are_stored_at_four_bits() {
    let dir = TempDir::new("q4-dtypes");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let kv = HIDDEN / HEADS * KV_HEADS;
    for f16 in [false, true] {
        let weights = load_weights(dir.path());
        let model = builder(&gpu, &weights, dir.path())
            .with_lora(lora_config())
            .with_f16_weights(f16)
            .with_q4_weights(true)
            .build()
            .expect("build");
        for layer in &model.layers {
            let Lfm2LayerOp::Attention(attn) = &layer.op else {
                panic!("llama blocks are attention");
            };
            // Logical [in, out] shapes.
            for (name, linear, k, n) in [
                ("q_proj", &attn.q_proj, HIDDEN, HIDDEN),
                ("k_proj", &attn.k_proj, HIDDEN, kv),
                ("v_proj", &attn.v_proj, HIDDEN, kv),
                ("o_proj", &attn.o_proj, HIDDEN, HIDDEN),
                ("gate_proj", &layer.mlp.gate_proj, HIDDEN, FFN),
                ("up_proj", &layer.mlp.up_proj, HIDDEN, FFN),
                ("down_proj", &layer.mlp.down_proj, FFN, HIDDEN),
            ] {
                let w = &linear.weight;
                assert_eq!(w.data.dtype, DType::Q4, "{name}");
                assert_eq!(w.data.shape, vec![k, n], "{name}: logical shape");
                assert_eq!(
                    w.data.size_in_bytes(),
                    4 * q4::packed_words(k, n) as u64,
                    "{name}"
                );
                assert!(!w.requires_grad, "{name} must be frozen");
                if let Some(lora) = &linear.lora {
                    assert_eq!(lora.a.data.dtype, DType::FP32, "{name} LoRA A");
                    assert_eq!(lora.b.data.dtype, DType::FP32, "{name} LoRA B");
                }
            }
            assert_eq!(layer.operator_norm.weight.data.dtype, DType::FP32);
            assert_eq!(layer.ffn_norm.weight.data.dtype, DType::FP32);
        }
        // 32 x 32 at 4 bits: 1024 / 64 = 16 groups of 8 code words and 1
        // scale word each.
        assert_eq!(4 * q4::packed_words(HIDDEN, HIDDEN), 576);
        let head = if f16 { DType::FP16 } else { DType::FP32 };
        assert_eq!(model.lm_head.weight.data.dtype, head, "lm_head (f16 {f16})");
        assert_eq!(model.embed_tokens.data.dtype, DType::FP32);
        assert_eq!(model.norm_f.weight.data.dtype, DType::FP32);
    }
}

#[test]
#[ignore = "needs a GPU"]
fn q4_model_matches_dequantised_f32_model_batch_1() {
    let (rows, targets) = one_row();
    compare_with_dequantised("batch 1", &rows, &targets);
}

#[test]
#[ignore = "needs a GPU"]
fn q4_model_matches_dequantised_f32_model_batch_3() {
    let (rows, targets) = three_rows();
    compare_with_dequantised("batch 3", &rows, &targets);
}

#[test]
#[ignore = "needs a GPU"]
fn q4_differs_from_the_unquantised_model() {
    let dir = TempDir::new("q4-differs");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let q = build(&gpu, dir.path(), true);
    let full = build(&gpu, dir.path(), false);
    let seq = pattern_sequence();
    let lq = host(&gpu, &q.forward(&gpu.engine, &seq[..23], 0, None, None));
    let lf = host(&gpu, &full.forward(&gpu.engine, &seq[..23], 0, None, None));
    // Proves the 4-bit weights are really used.
    let diff = max_abs_diff(&lq, &lf);
    assert!(diff > 1e-4, "4-bit logits equal the f32 ones ({diff})");
    let mask = vec![1u8; 23];
    let loss = |l: &[f32]| {
        ferrotensor_trainer::masked_cross_entropy_from_logits(l, VOCAB, &seq[1..], &mask)
            .expect("loss")
            .0
    };
    let (q_loss, f_loss) = (loss(&lq), loss(&lf));
    assert!(q_loss.is_finite());
    report(&format!(
        "quantisation effect (not a check): loss 4-bit {q_loss} vs f32 {f_loss} (diff {:+.3e}), max |logit diff| {diff:.3e}",
        q_loss - f_loss
    ));
}

/// Trainer-level comparison: several optimizer steps with gradient
/// accumulation, on single sequences and on batches.
#[test]
#[ignore = "needs a GPU"]
fn q4_training_matches_dequantised_f32_training() {
    let (a, b) = fixtures("q4-train");
    let gpu = gpu();
    let seq = pattern_sequence();
    // Six sequences of three lengths; two per optimizer step.
    let windows: Vec<(Vec<u32>, Vec<u32>)> = [(0, 20), (2, 23), (1, 15), (4, 24), (0, 23), (3, 18)]
        .iter()
        .map(|&(s, e)| (seq[s..e - 1].to_vec(), seq[s + 1..e].to_vec()))
        .collect();
    let mut losses = Vec::new();
    for dir in [a.path(), b.path()] {
        let model = build(&gpu, dir, dir == a.path());
        let (_, params) = sorted_lora_params(&model);
        let mut t = trainer(&gpu, model, 1e-2).with_grad_accum_steps(2);
        let mut run = Vec::new();
        for (input, target) in &windows {
            let stats = t.train_step(input, target, &params).expect("step");
            assert!(stats.loss.is_finite());
            run.push((stats.loss, stats.optimizer_step, stats.grad_norm));
        }
        losses.push(run);
    }
    let (q, r) = (&losses[0], &losses[1]);
    for (i, (x, y)) in q.iter().zip(r).enumerate() {
        assert_eq!(x.1, y.1, "call {i}: optimizer step");
        assert!(
            (x.0 - y.0).abs() <= 1e-4 * y.0.abs(),
            "call {i}: loss 4-bit {} vs dequantised f32 {}",
            x.0,
            y.0
        );
        if let (Some(gx), Some(gy)) = (x.2, y.2) {
            assert!(
                (gx - gy).abs() <= 1e-3 * gy,
                "call {i}: grad norm {gx} vs {gy}"
            );
        }
    }
    assert_eq!(q.iter().filter(|s| s.1).count(), 3, "three optimizer steps");
    report(&format!(
        "training, 3 steps x accum 2: losses 4-bit {:?} vs dequantised f32 {:?}",
        q.iter().map(|s| s.0).collect::<Vec<_>>(),
        r.iter().map(|s| s.0).collect::<Vec<_>>()
    ));
}

#[test]
#[ignore = "needs a GPU"]
fn q4_base_weights_still_learn() {
    let dir = TempDir::new("q4-learn");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build(&gpu, dir.path(), true);
    let (_, params) = sorted_lora_params(&model);
    let mut t = trainer(&gpu, model, 1e-2);
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let losses: Vec<f32> = (0..40)
        .map(|_| t.train_step(input, target, &params).expect("step").loss)
        .collect();
    assert!(losses.iter().all(|l| l.is_finite()));
    assert!(
        losses[39] < 0.5 * losses[0],
        "4-bit base weights should still learn the sequence: {losses:?}"
    );
}

/// Greedy decoding with a KV cache, as `ferrotensor generate` does it:
/// the generated tokens and the logits each one was chosen from.
fn generate(gpu: &Gpu, model: &Lfm2Model, prompt: &[u32], n: usize) -> (Vec<u32>, Vec<f32>) {
    let mut cache = model.new_kv_cache(prompt.len() + n).expect("kv cache");
    let mut tokens = prompt.to_vec();
    let mut generated = Vec::new();
    let mut seen = Vec::new();
    let mut logits = host(
        gpu,
        &model.forward(&gpu.engine, prompt, 0, Some(&mut cache), None),
    );
    for _ in 0..n {
        let last = &logits[logits.len() - VOCAB..];
        seen.extend_from_slice(last);
        let next = (0..VOCAB)
            .max_by(|&i, &j| last[i].total_cmp(&last[j]))
            .expect("vocab") as u32;
        generated.push(next);
        tokens.push(next);
        logits = host(
            gpu,
            &model.forward(
                &gpu.engine,
                &[next],
                tokens.len() - 1,
                Some(&mut cache),
                None,
            ),
        );
    }
    (generated, seen)
}

#[test]
#[ignore = "needs a GPU"]
fn q4_kv_cache_generation_matches_dequantised_f32() {
    let (a, b) = fixtures("q4-generate");
    let gpu = gpu();
    let m_q4 = build(&gpu, a.path(), true);
    let m_ref = build(&gpu, b.path(), false);
    for m in [&m_q4, &m_ref] {
        set_lora_b(&gpu, m);
    }
    let prompt = &pattern_sequence()[..5];
    let (tq, lq) = generate(&gpu, &m_q4, prompt, 10);
    let (tr, lr) = generate(&gpu, &m_ref, prompt, 10);
    let (d, at) = worst(&lq, &lr, 1e-3);
    assert!(
        d <= 1e-4,
        "logit {at}: 4-bit {} vs dequantised f32 {}",
        lq[at],
        lr[at]
    );
    assert_eq!(tq, tr, "greedy tokens");
    // The cached 4-bit decode equals a full recompute of the same tokens.
    let mut all = prompt.to_vec();
    all.extend_from_slice(&tq[..tq.len() - 1]);
    let full = host(&gpu, &m_q4.forward(&gpu.engine, &all, 0, None, None));
    let full_rows = &full[(prompt.len() - 1) * VOCAB..];
    let (d, at) = worst(&lq, full_rows, 1e-3);
    assert!(
        d <= 1e-4,
        "logit {at}: cached {} vs full {}",
        lq[at],
        full_rows[at]
    );
    report(&format!(
        "kv-cache generation: 4-bit tokens {tq:?} = dequantised f32 tokens {tr:?}, max |logit diff| {:.3e}",
        max_abs_diff(&lq, &lr)
    ));
}

/// Logits and LoRA gradients of a build, as raw bits.
fn bits(gpu: &Gpu, model: &Lfm2Model) -> (Vec<DType>, Vec<u32>, Vec<Vec<u32>>) {
    set_lora_b(gpu, model);
    let (rows, targets) = one_row();
    let p = pass(gpu, model, &rows, &targets);
    let mut dtypes: Vec<(String, DType)> = model
        .named_params()
        .into_iter()
        .map(|(n, t)| (n, t.data.dtype))
        .collect();
    dtypes.sort_by(|a, b| a.0.cmp(&b.0));
    (
        dtypes.into_iter().map(|(_, d)| d).collect(),
        p.logits.iter().map(|x| x.to_bits()).collect(),
        p.grads
            .iter()
            .map(|(_, g)| {
                g.as_ref()
                    .expect("grad")
                    .iter()
                    .map(|x| x.to_bits())
                    .collect()
            })
            .collect(),
    )
}

#[test]
#[ignore = "needs a GPU"]
fn q4_off_is_bit_identical_to_not_asking() {
    let dir = TempDir::new("q4-off");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    for f16 in [false, true] {
        let weights = load_weights(dir.path());
        let plain = builder(&gpu, &weights, dir.path())
            .with_lora(lora_config())
            .with_f16_weights(f16)
            .build()
            .expect("build");
        let off = builder(&gpu, &weights, dir.path())
            .with_lora(lora_config())
            .with_f16_weights(f16)
            .with_q4_weights(false)
            .build()
            .expect("build");
        let (dp, lp, gp) = bits(&gpu, &plain);
        let (doff, loff, goff) = bits(&gpu, &off);
        assert!(!dp.contains(&DType::Q4), "no 4-bit tensor without the flag");
        assert_eq!(dp.contains(&DType::FP16), f16);
        assert_eq!(dp, doff, "dtypes (f16 {f16})");
        assert_eq!(lp, loff, "logits (f16 {f16})");
        assert_eq!(gp, goff, "LoRA gradients (f16 {f16})");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn q4_with_trainable_base_is_an_error() {
    let dir = TempDir::new("q4-trainable");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let weights = load_weights(dir.path());
    let err = builder(&gpu, &weights, dir.path())
        .with_trainable_base(true)
        .with_q4_weights(true)
        .build()
        .err()
        .expect("4-bit with a trainable base must fail");
    assert!(err.0.contains("cannot train"), "{err}");
    assert!(err.0.contains("4-bit"), "{err}");
}

#[test]
#[ignore = "needs a GPU"]
fn pre_quantized_checkpoints_are_rejected() {
    let a = TempDir::new("scales-orig");
    let b = TempDir::new("scales");
    write_tiny_llama(a.path());
    let mut tensors = read_tensors(a.path());
    tensors.push((
        "model.layers.0.self_attn.q_proj.scales".into(),
        vec![1],
        vec![1.0],
    ));
    write_model(a.path(), b.path(), &tensors);
    let gpu = gpu();
    for q4 in [false, true] {
        let weights = load_weights(b.path());
        let err = builder(&gpu, &weights, b.path())
            .with_lora(lora_config())
            .with_q4_weights(q4)
            .build()
            .err()
            .expect("a .scales tensor must be rejected");
        assert!(err.0.contains("pre-quantized"), "{err}");
        assert!(err.0.contains("q_proj"), "{err}");
    }
}

#[test]
#[ignore = "needs a GPU"]
#[should_panic(expected = "a 4-bit right operand must not require a gradient")]
fn q4_weight_that_requires_a_gradient_is_rejected() {
    let gpu = gpu();
    let w = UnifiedTensor::from_f32_as_q4(&gpu.ctx, &pseudo_random(64 * 2, 7, 0.1), &[64, 2])
        .expect("quantise");
    let w = Tensor::new(gpu.ctx.clone(), w);
    assert!(w.requires_grad);
    let x = Tensor::frozen(
        gpu.ctx.clone(),
        UnifiedTensor::new(&gpu.ctx, &pseudo_random(3 * 64, 8, 1.0), &[3, 64]),
    );
    let _ = Tensor::matmul(&x, &w, &gpu.engine);
}

#[test]
#[ignore = "needs a GPU"]
fn memory_report_counts_the_q4_weights() {
    let dir = TempDir::new("q4-memory");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let mut recorder = MemoryRecorder::start(gpu.ctx.clone());
    let model = {
        let _tag = gpu.ctx.memory().scope(MemoryCategory::BaseWeights);
        let weights = load_weights(dir.path());
        builder(&gpu, &weights, dir.path())
            .with_lora(lora_config())
            .with_f16_weights(true)
            .with_q4_weights(true)
            .build()
            .expect("build")
    };
    let (_, params) = sorted_lora_params(&model);
    tag_trainable(&params);
    let load = recorder.record("after model load").clone();
    recorder.stop();
    let table = load.to_table();
    report(&table);

    let kv = HIDDEN / HEADS * KV_HEADS;
    // Logical [in, out] of each quantised projection.
    let shapes = [
        (HIDDEN, HIDDEN),
        (HIDDEN, kv),
        (HIDDEN, kv),
        (HIDDEN, HIDDEN),
        (HIDDEN, FFN),
        (HIDDEN, FFN),
        (FFN, HIDDEN),
    ];
    let q4_bytes: u64 = LAYERS as u64
        * shapes
            .iter()
            .map(|&(k, n)| 4 * q4::packed_words(k, n) as u64)
            .sum::<u64>();
    // The tied lm_head [HIDDEN, VOCAB] is the only f16 weight.
    let f16_bytes = wgpu_buffer_size(((HIDDEN * VOCAB) as u64 * 2).next_multiple_of(4));
    for s in [&load.peak, &load.live] {
        let b = s.category(MemoryCategory::BaseWeights);
        assert_eq!(b.q4_bytes, q4_bytes, "4-bit base weights");
        assert_eq!(b.f16_bytes, f16_bytes, "f16 lm_head");
        assert_eq!(s.category(MemoryCategory::Trainable).q4_bytes, 0);
    }
    let row = table
        .lines()
        .find(|l| l.contains("of which 4-bit"))
        .expect("an \"of which 4-bit\" row");
    report(&format!(
        "4-bit base weights: {q4_bytes} bytes in {} tensors; report row: {}",
        LAYERS * shapes.len(),
        row.trim()
    ));
}

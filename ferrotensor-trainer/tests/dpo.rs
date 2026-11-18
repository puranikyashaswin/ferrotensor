//! DPO: the reference pass with the adapters off, the loss and its gradient
//! on raw logits against a host formula, parity of the loss and every LoRA
//! gradient with PyTorch autograd, and the training loop.
//!
//! The PyTorch reference is written by `scripts/gen_dpo_fixture.py` from
//! the gradient-parity fixture (the model and adapters of
//! `ferrotensor-graph/tests/fixtures/gradient_parity`). CI runs the script
//! and passes its output directory in `FERROTENSOR_DPO_FIXTURE`; without it
//! the parity tests print a note and pass.
//!
//! These need a GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test --release -p ferrotensor-trainer --test dpo -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use common::{gpu, Gpu, TempDir};
use ferrotensor_autograd::{no_grad, Tensor};
use ferrotensor_backend::UnifiedTensor;
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_graph::{adapters_enabled, without_adapters, Linear};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::dpo::{
    dpo_loss, reference_log_probs, sequence_log_probs, ScoredRows, DEFAULT_BETA,
};
use ferrotensor_trainer::PreferencePair;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The loss and every LoRA gradient must be within this of PyTorch.
///
/// Both sides compute in f32 from the same weights; the SFT gradient-parity
/// tests on this model land between 1e-7 and 7e-7. The DPO gradients are
/// the SFT row gradients times `beta * sigmoid(-margin) / pairs`, so they
/// are smaller, and the loss depends on the log-probabilities only through
/// `beta` times their differences.
const PARITY_ATOL: f64 = 1e-6;

/// Gradients pass when `|a - e| <= PARITY_ATOL + GRAD_RTOL * |e|`; the loss
/// keeps the plain `PARITY_ATOL`.
///
/// A gradient is `beta * sigmoid(-z) / pairs` times SFT-style row gradients,
/// where `z` is the implicit reward margin. `z` is built from four sequence
/// log-probabilities, each a sum of response-token terms of size 3 to 4, so
/// each carries an f32 error of a few 1e-7 (the loss shows 6e-7). The error
/// in `z` moves `sigmoid(-z)` by a relative amount of the same size, and
/// the same factor multiplies every element of the gradients of that pair,
/// so the error scales with the gradient: it is relative, not absolute.
/// With gradients of 0.25 to 0.55 a relative error of 2.5e-6 is 1.4e-6
/// absolute. 1e-5 is about four times that and still 100 times tighter than
/// the 1e-3 used for the SFT gradient tests, so a wrong sign, a missing
/// pair, a wrong scale or a wrong mask still fails by orders of magnitude.
const GRAD_RTOL: f64 = 1e-5;

/// The sequence log-probabilities are sums of eight f32 cross-entropies of
/// about 3.5 each, so f32 rounding alone is of order 1e-6 there. They are
/// printed and checked loosely; the loss and gradients carry the gate.
const LOG_PROB_ATOL: f64 = 1e-4;

/// How far a gradient is from its reference.
struct GradDiff {
    /// Largest `|a - e|`.
    abs: f64,
    /// Largest `|e|`.
    scale: f64,
    /// Largest `|a - e| / |e|` over elements with `|e| >= scale / 10`.
    rel: f64,
    /// Largest `|a - e| / (PARITY_ATOL + GRAD_RTOL * |e|)`; passes at <= 1.
    ratio: f64,
}

fn gradient_diff(actual: &[f32], expected: &[f32]) -> GradDiff {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let scale = expected
        .iter()
        .fold(0.0f64, |m, &e| m.max((e as f64).abs()));
    let mut diff = GradDiff {
        abs: 0.0,
        scale,
        rel: 0.0,
        ratio: 0.0,
    };
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e) = (a as f64, e as f64);
        let d = (a - e).abs();
        if d.is_nan() {
            // NaN must fail.
            diff.abs = f64::INFINITY;
            diff.ratio = f64::INFINITY;
            continue;
        }
        diff.abs = diff.abs.max(d);
        diff.ratio = diff.ratio.max(d / (PARITY_ATOL + GRAD_RTOL * e.abs()));
        if e.abs() >= scale / 10.0 {
            diff.rel = diff.rel.max(d / e.abs());
        }
    }
    diff
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn gradient_fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ferrotensor-graph/tests/fixtures/gradient_parity")
}

/// The adapter shape, from either fixture's `cases.json`.
#[derive(Deserialize)]
struct LoraMeta {
    lora_rank: usize,
    lora_alpha: f32,
}

#[derive(Deserialize)]
struct Meta {
    #[serde(flatten)]
    lora: LoraMeta,
    cases: HashMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    rows: Vec<Vec<u32>>,
    targets: Vec<Vec<u32>>,
    masks: Vec<Vec<u8>>,
    beta: f32,
    loss: f64,
    policy_log_probs: Vec<f64>,
    reference_log_probs: Vec<f64>,
}

/// `(shape, values)` by name.
type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn read_tensors(path: &Path) -> Tensors {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let file = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("{} is not a safetensors file: {e}", path.display()));
    let mut out = Tensors::new();
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

fn host(gpu: &Gpu, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&gpu.ctx))
}

/// The gradient-parity checkpoint with an adapter on every linear layer and
/// the LM head, holding the fixture's non-zero LoRA factors.
fn parity_model(gpu: &Gpu, rank: usize, alpha: f32) -> Lfm2Model {
    let dir = gradient_fixture_dir();
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir).expect("weights");
    let lora = LoRaConfig {
        rank,
        alpha,
        target_modules: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
        .map(String::from)
        .to_vec(),
    };
    let mut model = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_lora(lora)
        .build()
        .expect("build");
    let adapters = read_tensors(&dir.join("adapters.safetensors"));
    let set = |prefix: String, linear: &mut Linear| {
        let lora = linear
            .lora
            .as_mut()
            .unwrap_or_else(|| panic!("{prefix} has no adapter"));
        for (suffix, slot) in [("lora_a", &mut lora.a), ("lora_b", &mut lora.b)] {
            let (shape, values) = adapters
                .get(&format!("{prefix}.{suffix}"))
                .unwrap_or_else(|| panic!("fixture has no {prefix}.{suffix}"));
            assert_eq!(&slot.data.shape, shape, "{prefix}.{suffix} shape");
            *slot = Tensor::new(gpu.ctx.clone(), UnifiedTensor::new(&gpu.ctx, values, shape));
        }
    };
    for (l, layer) in model.layers.iter_mut().enumerate() {
        let p = format!("model.layers.{l}");
        if let Lfm2LayerOp::Attention(attn) = &mut layer.op {
            set(format!("{p}.self_attn.q_proj"), &mut attn.q_proj);
            set(format!("{p}.self_attn.k_proj"), &mut attn.k_proj);
            set(format!("{p}.self_attn.v_proj"), &mut attn.v_proj);
            set(format!("{p}.self_attn.o_proj"), &mut attn.o_proj);
        }
        set(format!("{p}.mlp.gate_proj"), &mut layer.mlp.gate_proj);
        set(format!("{p}.mlp.up_proj"), &mut layer.mlp.up_proj);
        set(format!("{p}.mlp.down_proj"), &mut layer.mlp.down_proj);
    }
    set("lm_head".to_string(), &mut model.lm_head);
    model
}

/// Every LoRA factor under the fixture's names.
fn lora_tensors(model: &Lfm2Model) -> Vec<(String, Tensor)> {
    let mut linears: Vec<(String, &Linear)> = Vec::new();
    for (l, layer) in model.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        let Lfm2LayerOp::Attention(attn) = &layer.op else {
            panic!("layer {l} should be attention");
        };
        linears.push((format!("{p}.self_attn.q_proj"), &attn.q_proj));
        linears.push((format!("{p}.self_attn.k_proj"), &attn.k_proj));
        linears.push((format!("{p}.self_attn.v_proj"), &attn.v_proj));
        linears.push((format!("{p}.self_attn.o_proj"), &attn.o_proj));
        linears.push((format!("{p}.mlp.gate_proj"), &layer.mlp.gate_proj));
        linears.push((format!("{p}.mlp.up_proj"), &layer.mlp.up_proj));
        linears.push((format!("{p}.mlp.down_proj"), &layer.mlp.down_proj));
    }
    linears.push(("lm_head".to_string(), &model.lm_head));
    let mut out = Vec::new();
    for (prefix, linear) in linears {
        let lora = linear
            .lora
            .as_ref()
            .unwrap_or_else(|| panic!("{prefix} has no adapter"));
        out.push((format!("{prefix}.lora_a"), lora.a.clone()));
        out.push((format!("{prefix}.lora_b"), lora.b.clone()));
    }
    out
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let mut err = 0.0f64;
    let mut scale = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let d = (a as f64 - e as f64).abs();
        // NaN must fail.
        err = if d.is_nan() {
            f64::INFINITY
        } else {
            err.max(d)
        };
        scale = scale.max((e as f64).abs());
    }
    (err, scale)
}

// ---------------------------------------------------------------------------
// The reference pass
// ---------------------------------------------------------------------------

/// With the adapters switched off the model is the base model: its logits
/// are the fixture's base logits (PyTorch with every LoRA B zero), nothing
/// is recorded for backward, and outside the switch the adapters are back.
#[test]
#[ignore = "needs a GPU"]
fn reference_pass_runs_the_base_model_without_a_graph() {
    let gpu = gpu();
    let meta: LoraMeta = serde_json::from_str(
        &std::fs::read_to_string(gradient_fixture_dir().join("cases.json")).expect("cases.json"),
    )
    .unwrap_or_else(|e| panic!("parse cases.json: {e}"));
    let model = parity_model(&gpu, meta.lora_rank, meta.lora_alpha);
    let expected = read_tensors(&gradient_fixture_dir().join("expected.safetensors"));
    let row0: Vec<u32> = vec![3, 17, 5, 29, 0, 11, 23, 8];

    assert!(adapters_enabled());
    let reference = no_grad(|| {
        without_adapters(|| {
            assert!(!adapters_enabled());
            model.forward(&gpu.engine, &row0, 0, None, None)
        })
    });
    assert!(adapters_enabled(), "the switch must restore the adapters");
    assert!(
        reference.creator.is_none(),
        "reference logits recorded a graph"
    );
    assert!(
        !reference.requires_grad,
        "reference logits require a gradient"
    );
    let (base_shape, base) = &expected["logits.base"];
    assert_eq!(&reference.data.shape, base_shape);
    let (err, _) = max_abs_diff(&host(&gpu, &reference.data), base);
    assert!(err < 1e-4, "base logits differ from PyTorch by {err:e}");

    let policy = model.forward(&gpu.engine, &row0, 0, None, None);
    assert!(policy.creator.is_some() && policy.requires_grad);
    let (_, with_adapters) = &expected["logits.single"];
    let (err, _) = max_abs_diff(&host(&gpu, &policy.data), with_adapters);
    assert!(err < 1e-4, "adapter logits differ from PyTorch by {err:e}");
    let (gap, _) = max_abs_diff(&host(&gpu, &policy.data), base);
    assert!(gap > 1e-2, "the adapters should change the logits");

    // reference_log_probs equals the log-softmax of the base logits, in a
    // batch and one row at a time.
    let row1: Vec<u32> = vec![30, 2, 19, 7, 7, 26, 1, 14];
    let targets: Vec<u32> = [&row0[1..], &[12u32][..], &row1[1..], &[31u32][..]].concat();
    let mask: Vec<u8> = [[0, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 1]].concat();
    let rows: Vec<&[u32]> = vec![row0.as_slice(), row1.as_slice()];
    let batched = reference_log_probs(&model, &gpu.engine, &rows, &targets, &mask)
        .expect("reference log-probs");
    let vocab = base_shape[2];
    let (lp0, n0) = host_log_prob(base, vocab, &targets[..8], &mask[..8]);
    assert_eq!(batched[0].1, n0);
    assert!(
        (batched[0].0 - lp0).abs() < LOG_PROB_ATOL,
        "{} vs {lp0}",
        batched[0].0
    );
    for (i, row) in rows.iter().enumerate() {
        let one = reference_log_probs(
            &model,
            &gpu.engine,
            &[*row],
            &targets[i * 8..(i + 1) * 8],
            &mask[i * 8..(i + 1) * 8],
        )
        .expect("reference log-probs");
        assert!(
            (one[0].0 - batched[i].0).abs() < LOG_PROB_ATOL,
            "row {i}: alone {} vs batched {}",
            one[0].0,
            batched[i].0
        );
    }
}

// ---------------------------------------------------------------------------
// The loss on raw logits
// ---------------------------------------------------------------------------

/// Sum over scored rows of `log softmax(logits)[target]`, in f64.
fn host_log_prob(logits: &[f32], vocab: usize, targets: &[u32], mask: &[u8]) -> (f64, usize) {
    let mut sum = 0.0;
    let mut n = 0;
    for (r, row) in logits.chunks(vocab).enumerate() {
        if mask[r] == 0 {
            continue;
        }
        let max = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
        let lse = max
            + row
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln();
        sum += row[targets[r] as usize] as f64 - lse;
        n += 1;
    }
    (sum, n)
}

/// `sequence_log_probs` and the `dpo_loss` gradient on leaf logits equal the
/// formulas computed on the host, with one forward pass for all sequences
/// and with one pass per sequence; masked rows get no gradient.
#[test]
#[ignore = "needs a GPU"]
fn dpo_loss_and_gradient_match_the_host_formula() {
    let gpu = gpu();
    let (sequences, seq, vocab) = (4usize, 5usize, 7usize);
    let values = common::pseudo_random(sequences * seq * vocab, 77, 3.0);
    let targets: Vec<u32> = (0..sequences * seq)
        .map(|i| ((i * 5 + 2) % vocab) as u32)
        .collect();
    let mask: Vec<u8> = [
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    .concat();
    let reference = [-9.5f64, -8.0, -11.25, -7.5];
    let beta = 0.3f32;

    // Host: log-probs, loss and d loss / d logits.
    let rows = |s: usize| s * seq * vocab..(s + 1) * seq * vocab;
    let logp: Vec<f64> = (0..sequences)
        .map(|s| {
            host_log_prob(
                &values[rows(s)],
                vocab,
                &targets[s * seq..(s + 1) * seq],
                &mask[s * seq..(s + 1) * seq],
            )
            .0
        })
        .collect();
    let pairs = sequences / 2;
    let b = beta as f64;
    let mut loss = 0.0;
    let mut d_logp = vec![0.0f64; sequences];
    for i in 0..pairs {
        let z = b * ((logp[2 * i] - reference[2 * i]) - (logp[2 * i + 1] - reference[2 * i + 1]));
        loss += (1.0 + (-z).exp()).ln() / pairs as f64;
        let s = 1.0 / (1.0 + z.exp()); // sigmoid(-z)
        d_logp[2 * i] = -b * s / pairs as f64;
        d_logp[2 * i + 1] = b * s / pairs as f64;
    }
    let mut want = vec![0.0f64; values.len()];
    for r in 0..sequences * seq {
        if mask[r] == 0 {
            continue;
        }
        let row = &values[r * vocab..(r + 1) * vocab];
        let max = row.iter().fold(f64::NEG_INFINITY, |m, &x| m.max(x as f64));
        let z: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
        for v in 0..vocab {
            let softmax = (row[v] as f64 - max).exp() / z;
            let onehot = if v == targets[r] as usize { 1.0 } else { 0.0 };
            // d logp / d logit = onehot - softmax
            want[r * vocab + v] = d_logp[r / seq] * (onehot - softmax);
        }
    }

    for per_sequence in [false, true] {
        let leaves: Vec<Tensor> = if per_sequence {
            (0..sequences)
                .map(|s| {
                    Tensor::new(
                        gpu.ctx.clone(),
                        UnifiedTensor::new(&gpu.ctx, &values[rows(s)], &[seq, vocab]),
                    )
                })
                .collect()
        } else {
            vec![Tensor::new(
                gpu.ctx.clone(),
                UnifiedTensor::new(&gpu.ctx, &values, &[sequences, seq, vocab]),
            )]
        };
        let n = if per_sequence { 1 } else { sequences };
        let scored: Vec<ScoredRows<'_>> = leaves
            .iter()
            .enumerate()
            .map(|(i, l)| ScoredRows {
                logits: l,
                target_ids: &targets[i * n * seq..(i + 1) * n * seq],
                loss_mask: &mask[i * n * seq..(i + 1) * n * seq],
                sequences: n,
            })
            .collect();

        let got_logp: Vec<(f64, usize)> = scored
            .iter()
            .flat_map(|s| {
                sequence_log_probs(
                    s.logits,
                    s.target_ids,
                    s.loss_mask,
                    s.sequences,
                    &gpu.engine,
                )
                .expect("log-probs")
            })
            .collect();
        for (s, &(lp, tokens)) in got_logp.iter().enumerate() {
            let expect_tokens = mask[s * seq..(s + 1) * seq]
                .iter()
                .filter(|&&m| m != 0)
                .count();
            assert_eq!(tokens, expect_tokens, "sequence {s} tokens");
            assert!(
                (lp - logp[s]).abs() < 1e-5,
                "sequence {s}: {lp} vs {}",
                logp[s]
            );
        }

        let out = dpo_loss(&scored, &reference, beta, &gpu.engine).expect("dpo loss");
        assert!(
            (out.value as f64 - loss).abs() < 1e-6,
            "loss {} vs {loss}",
            out.value
        );
        assert_eq!(out.pairs.len(), pairs);
        out.tensor.backward(&gpu.engine);
        let got: Vec<f32> = leaves
            .iter()
            .flat_map(|l| host(&gpu, l.grad.borrow().as_ref().expect("logits gradient")))
            .collect();
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g as f64 - w).abs() < 1e-6,
                "per_sequence {per_sequence}: grad[{i}] {g} vs {w}"
            );
            if mask[i / vocab] == 0 {
                assert_eq!(g, 0.0, "masked row {} got a gradient", i / vocab);
            }
        }
    }

    // Mismatched inputs are refused rather than misread.
    let leaf = Tensor::new(
        gpu.ctx.clone(),
        UnifiedTensor::new(&gpu.ctx, &values, &[sequences, seq, vocab]),
    );
    let all = ScoredRows {
        logits: &leaf,
        target_ids: &targets,
        loss_mask: &mask,
        sequences,
    };
    assert!(dpo_loss(&[all], &reference[..2], beta, &gpu.engine).is_none());
    let none = vec![0u8; mask.len()];
    let unmasked = ScoredRows {
        loss_mask: &none,
        ..all
    };
    assert!(dpo_loss(&[unmasked], &reference, beta, &gpu.engine).is_none());
}

// ---------------------------------------------------------------------------
// Parity with PyTorch
// ---------------------------------------------------------------------------

/// The case's loss, log-probabilities and every LoRA gradient against the
/// PyTorch reference, with the policy run as one batch or one sequence per
/// forward pass.
fn check_pytorch_case(case_name: &str, per_sequence: bool) {
    let Some(dir) = std::env::var_os("FERROTENSOR_DPO_FIXTURE") else {
        eprintln!(
            "FERROTENSOR_DPO_FIXTURE is not set; skipping DPO parity [{case_name}] \
             (run scripts/gen_dpo_fixture.py DIR and set it to DIR)"
        );
        return;
    };
    let dir = PathBuf::from(dir);
    let meta: Meta = serde_json::from_str(
        &std::fs::read_to_string(dir.join("cases.json")).expect("read DPO cases.json"),
    )
    .unwrap_or_else(|e| panic!("parse DPO cases.json: {e}"));
    let expected = read_tensors(&dir.join("expected.safetensors"));
    let case = meta
        .cases
        .get(case_name)
        .unwrap_or_else(|| panic!("fixture has no case {case_name}"));

    let gpu = gpu();
    let model = parity_model(&gpu, meta.lora.lora_rank, meta.lora.lora_alpha);
    let rows: Vec<&[u32]> = case.rows.iter().map(Vec::as_slice).collect();
    let targets = case.targets.concat();
    let masks = case.masks.concat();
    let seq = rows[0].len();

    let reference: Vec<f64> = reference_log_probs(&model, &gpu.engine, &rows, &targets, &masks)
        .expect("reference log-probs")
        .into_iter()
        .map(|(lp, _)| lp)
        .collect();
    let logits: Vec<Tensor> = if per_sequence {
        rows.iter()
            .map(|r| model.forward(&gpu.engine, r, 0, None, None))
            .collect()
    } else {
        assert!(model.supports_batching());
        vec![model.forward_batch(&gpu.engine, &rows)]
    };
    let n = if per_sequence { 1 } else { rows.len() };
    let scored: Vec<ScoredRows<'_>> = logits
        .iter()
        .enumerate()
        .map(|(i, l)| ScoredRows {
            logits: l,
            target_ids: &targets[i * n * seq..(i + 1) * n * seq],
            loss_mask: &masks[i * n * seq..(i + 1) * n * seq],
            sequences: n,
        })
        .collect();
    let loss = dpo_loss(&scored, &reference, case.beta, &gpu.engine).expect("dpo loss");
    loss.tensor.backward(&gpu.engine);

    let label = format!(
        "{case_name}{}",
        if per_sequence {
            ", one pass per sequence"
        } else {
            ""
        }
    );
    let mut text = format!(
        "\nDPO parity [{label}] vs PyTorch, beta {}: the loss passes if |a-e| <= {PARITY_ATOL:e}; \
         a gradient passes if |a-e| <= {PARITY_ATOL:e} + {GRAD_RTOL:e}*|e|\n\
         {:<40} {:>12} {:>12} {:>12} {:>8}\n",
        case.beta, "tensor", "max|e|", "max abs", "max rel", "ratio"
    );
    let mut failing = Vec::new();
    let loss_err = (loss.value as f64 - case.loss).abs();
    text += &format!(
        "{:<40} {:>12.4e} {:>12.3e}\n",
        "loss",
        case.loss.abs(),
        loss_err
    );
    if loss_err.is_nan() || loss_err > PARITY_ATOL {
        failing.push("loss".to_string());
    }
    let policy: Vec<f64> = loss
        .pairs
        .iter()
        .flat_map(|p| [p.policy_chosen, p.policy_rejected])
        .collect();
    for (what, got, want) in [
        ("reference log-prob", &reference, &case.reference_log_probs),
        ("policy log-prob", &policy, &case.policy_log_probs),
    ] {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            text += &format!(
                "{:<40} {:>12.4e} {:>12.3e}\n",
                format!("{what} {i}"),
                w.abs(),
                err
            );
            if err.is_nan() || err > LOG_PROB_ATOL {
                failing.push(format!("{what} {i}"));
            }
        }
    }

    let prefix = format!("grad.{case_name}.");
    let expected_count = expected.keys().filter(|k| k.starts_with(&prefix)).count();
    let mut compared = 0;
    let mut worst = (String::new(), 0.0f64);
    for (name, tensor) in lora_tensors(&model) {
        let (shape, want) = expected
            .get(&format!("{prefix}{name}"))
            .unwrap_or_else(|| panic!("fixture has no gradient for {name}"));
        let grad = tensor
            .grad
            .borrow()
            .clone()
            .unwrap_or_else(|| panic!("no gradient for {name}"));
        assert_eq!(&grad.shape, shape, "gradient shape of {name}");
        let d = gradient_diff(&host(&gpu, &grad), want);
        text += &format!(
            "{name:<40} {:>12.4e} {:>12.3e} {:>12.3e} {:>8.3}\n",
            d.scale, d.abs, d.rel, d.ratio
        );
        if d.ratio.is_nan() || d.ratio > 1.0 {
            failing.push(name.clone());
        }
        if d.abs >= worst.1 {
            worst = (name, d.abs);
        }
        compared += 1;
    }
    text += &format!(
        "loss max abs {loss_err:.3e}; worst gradient {} max abs {:.3e}; {compared} gradients compared\n",
        worst.0, worst.1
    );
    let _ = std::io::stderr().write_all(text.as_bytes());
    assert_eq!(
        compared, expected_count,
        "not every expected gradient was compared"
    );
    assert!(
        failing.is_empty(),
        "[{label}] out of tolerance: {failing:?}{text}"
    );
}

/// One pair, beta 0.1: loss and the gradients of all 30 LoRA tensors.
#[test]
#[ignore = "needs a GPU"]
fn dpo_matches_pytorch_for_one_pair() {
    check_pytorch_case("pair", false);
}

/// Two pairs with different response lengths, beta 0.5, in one batch.
#[test]
#[ignore = "needs a GPU"]
fn dpo_matches_pytorch_for_two_pairs() {
    check_pytorch_case("batch", false);
}

/// The two pairs with one forward pass per sequence (the path for models
/// that cannot batch, such as LFM2 with convolution layers).
#[test]
#[ignore = "needs a GPU"]
fn dpo_matches_pytorch_one_sequence_per_pass() {
    check_pytorch_case("batch", true);
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn tiny_pairs() -> Vec<PreferencePair> {
    let prompt = [5u32, 12, 19, 26];
    let chosen = [33u32, 40, 47, 54, 61];
    let rejected = [8u32, 8, 8, 8, 8];
    let other_prompt = [6u32, 13, 20];
    vec![
        PreferencePair::from_tokens(&prompt, &chosen, &rejected, 64).expect("pair"),
        PreferencePair::from_tokens(&other_prompt, &chosen[..3], &rejected[..4], 64).expect("pair"),
    ]
}

/// Fresh adapters (B = 0) make the policy equal the reference, so the first
/// loss is ln 2 with a zero margin; training then ranks the chosen response
/// above the rejected one.
#[test]
#[ignore = "needs a GPU"]
fn train_dpo_starts_at_log_2_and_learns_the_preference() {
    let gpu = gpu();
    let dir = TempDir::new("dpo");
    common::write_tiny_llama(dir.path());
    let model = common::build_model(&gpu, dir.path());
    let (_, params) = common::sorted_lora_params(&model);
    let mut trainer = common::trainer(&gpu, model, 1e-2);
    trainer.grad_accum_steps = 1;
    let pairs = tiny_pairs();

    let first = trainer
        .train_dpo(&pairs, DEFAULT_BETA, &params)
        .expect("train step");
    assert_eq!(first.len(), pairs.len());
    let s = first[0];
    assert!(
        (s.loss - std::f32::consts::LN_2).abs() < 1e-5,
        "first loss {} should be ln 2",
        s.loss
    );
    assert!(s.margin.abs() < 1e-5, "first margin {}", s.margin);
    assert_eq!(s.tokens, pairs[0].supervised_tokens());
    assert!(first
        .iter()
        .all(|s| s.optimizer_step && s.grad_norm.is_some()));

    let mut last = first;
    for _ in 0..30 {
        last = trainer
            .train_dpo(&pairs, DEFAULT_BETA, &params)
            .expect("train step");
    }
    for (i, s) in last.iter().enumerate() {
        assert!(s.loss.is_finite(), "pair {i}: loss {}", s.loss);
        assert!(
            s.margin > 0.0,
            "pair {i}: margin {} after training",
            s.margin
        );
        assert!(s.loss < std::f32::consts::LN_2, "pair {i}: loss {}", s.loss);
        assert!(s.chosen_reward > s.rejected_reward);
    }
}

/// Accumulation counts pairs: with a window of two, three pairs step the
/// optimizer once (after the second) and leave one pair pending.
#[test]
#[ignore = "needs a GPU"]
fn train_dpo_steps_at_accumulation_boundaries() {
    let gpu = gpu();
    let dir = TempDir::new("dpo-accum");
    common::write_tiny_llama(dir.path());
    let model = common::build_model(&gpu, dir.path());
    let (_, params) = common::sorted_lora_params(&model);
    let mut trainer = common::trainer(&gpu, model, 1e-2);
    trainer.grad_accum_steps = 2;
    let mut pairs = tiny_pairs();
    pairs.push(pairs[0].clone());

    let stats = trainer
        .train_dpo(&pairs, DEFAULT_BETA, &params)
        .expect("train");
    let steps: Vec<bool> = stats.iter().map(|s| s.optimizer_step).collect();
    assert_eq!(steps, [false, true, false]);
    assert_eq!(trainer.step_in_accum, 1);
    assert!(params.iter().any(|p| p.grad.borrow().is_some()));
}

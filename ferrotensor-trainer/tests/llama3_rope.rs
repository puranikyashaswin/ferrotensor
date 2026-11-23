//! A Llama 3 checkpoint (`rope_type: "llama3"` scaling, grouped-query
//! attention, tied embeddings, RoPE base 5e5, head size 16) against Hugging
//! Face transformers.
//!
//! `scripts/gen_llama3_fixture.py` builds a tiny random `LlamaForCausalLM`,
//! saves it, and records its logits and the gradients of a cross-entropy
//! loss with respect to every decoder-block linear weight. These tests load
//! the saved checkpoint through [`Lfm2ModelBuilder`] and compare.
//!
//! They need a GPU, so they are `#[ignore]`d by default. Without
//! `FERROTENSOR_LLAMA3_FIXTURE` they print a message and skip; CI sets it:
//!
//! ```text
//! python scripts/gen_llama3_fixture.py /tmp/llama3-fixture
//! FERROTENSOR_LLAMA3_FIXTURE=/tmp/llama3-fixture \
//!   cargo test --release -p ferrotensor-trainer --test llama3_rope -- --ignored --test-threads=1
//! ```
//!
//! One more test compares the logits of a real checkpoint, such as
//! Llama-3.2-1B, with transformers' on a few fixed prompts. It reads
//! `FERROTENSOR_LLAMA3_MODEL` (the checkpoint directory) and
//! `FERROTENSOR_LLAMA3_REAL` (the output directory of
//! `scripts/gen_llama3_real_logits.py`), and skips without them.

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use common::{gpu, host, Gpu};
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_graph::Linear;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::masked_cross_entropy;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Logits agree with transformers within this absolute error, as in the
/// Llama and Qwen2 parity tests (`model_correctness.rs`, `qwen2_family.rs`):
/// both sides compute in f32, so they differ only in summation order and in
/// `exp`, `sin` and `cos`, about 1e-5 on logits of order 1, while unscaled
/// RoPE moves them by more than 0.05 (checked below).
const REFERENCE_TOL: f32 = 1e-3;

/// Gradients pass when `|actual - expected| <= ATOL + RTOL * |expected|`,
/// the rule and values of `gradient_parity.rs`.
const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-3;

const LINEARS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[derive(Deserialize)]
struct Case {
    ids: Vec<u32>,
    targets: Vec<u32>,
    loss: f32,
}

struct Fixture {
    dir: PathBuf,
    case: Case,
    /// `(shape, values)` by name: `logits` and `grad.<weight name>`.
    expected: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Fixture {
    /// `None`, with a message, when `FERROTENSOR_LLAMA3_FIXTURE` is unset, so the
    /// suite still runs on a machine without PyTorch. A path that is set but
    /// wrong fails, which keeps CI strict.
    fn load() -> Option<Self> {
        let Some(dir) = std::env::var_os("FERROTENSOR_LLAMA3_FIXTURE").map(PathBuf::from) else {
            eprintln!(
                "skipping Llama 3 parity; set FERROTENSOR_LLAMA3_FIXTURE to the output of scripts/gen_llama3_fixture.py"
            );
            return None;
        };
        let case = std::fs::read_to_string(dir.join("cases.json")).expect("read cases.json");
        let bytes = std::fs::read(dir.join("expected.safetensors")).expect("read expected");
        let file = safetensors::SafeTensors::deserialize(&bytes).expect("parse expected");
        let expected = file
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name} is not f32");
                let values = view
                    .data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes(*c))
                    .collect();
                (name, (view.shape().to_vec(), values))
            })
            .collect();
        Some(Self {
            dir,
            case: serde_json::from_str(&case).expect("parse cases.json"),
            expected,
        })
    }

    fn expected(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.expected
            .get(name)
            .unwrap_or_else(|| panic!("fixture has no {name}"))
    }

    fn build(&self, gpu: &Gpu, lora: Option<LoRaConfig>, f16_weights: bool) -> Lfm2Model {
        let dir = self.dir.join("model");
        let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
        let mut weights = WeightsContext::new();
        weights.load_model_dir(&dir).expect("weights");
        let mut builder =
            Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
                .with_f16_weights(f16_weights);
        if let Some(lora) = lora {
            builder = builder.with_lora(lora);
        }
        builder.build().expect("build")
    }
}

/// Every decoder-block linear layer with its weight name.
fn linears(model: &Lfm2Model) -> Vec<(String, &Linear)> {
    let mut out = Vec::new();
    for (l, layer) in model.layers.iter().enumerate() {
        let Lfm2LayerOp::Attention(attn) = &layer.op else {
            panic!("layer {l} should be attention");
        };
        let mlp = &layer.mlp;
        let layer_linears = [
            &attn.q_proj,
            &attn.k_proj,
            &attn.v_proj,
            &attn.o_proj,
            &mlp.gate_proj,
            &mlp.up_proj,
            &mlp.down_proj,
        ];
        for (name, linear) in LINEARS.iter().zip(layer_linears) {
            out.push((format!("model.layers.{l}.{name}.weight"), linear));
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert!(actual.iter().all(|v| v.is_finite()), "{what}: non-finite");
    let diff = max_abs_diff(actual, expected);
    assert!(diff <= tol, "{what}: max abs diff {diff} > {tol}");
}

#[test]
#[ignore = "needs a GPU; skipped without FERROTENSOR_LLAMA3_FIXTURE"]
fn logits_match_transformers() {
    let Some(fixture) = Fixture::load() else {
        return;
    };
    let gpu = gpu();
    let mut model = fixture.build(&gpu, None, false);
    for (name, linear) in linears(&model) {
        assert!(linear.bias.is_none(), "Llama has no bias on {name}");
    }
    for layer in &model.layers {
        let Lfm2LayerOp::Attention(attn) = &layer.op else {
            panic!("every layer should be attention");
        };
        // The table has one entry per rotation pair of the 16-wide head.
        let table = attn.rope_inv_freq.as_ref().expect("scaled RoPE table");
        assert_eq!(table.shape, vec![8]);
    }
    // Tied: no lm_head.weight in the file, so the head is embed_tokens^T.
    assert_eq!(model.lm_head.weight.data.shape, vec![64, 96]);

    let (shape, expected) = fixture.expected("logits");
    let logits = model.forward(&gpu.engine, &fixture.case.ids, 0, None, None);
    assert_eq!(&logits.data.shape, shape);
    let actual = host(&gpu, &logits);
    assert_close(&actual, expected, REFERENCE_TOL, "logits vs transformers");

    // The scaling matters, or the comparison above would not prove it is
    // applied: with plain RoPE of the same base the logits move.
    for layer in &mut model.layers {
        if let Lfm2LayerOp::Attention(attn) = &mut layer.op {
            attn.rope_inv_freq = None;
        }
    }
    let unscaled = host(
        &gpu,
        &model.forward(&gpu.engine, &fixture.case.ids, 0, None, None),
    );
    assert!(
        max_abs_diff(&unscaled, expected) > 0.05,
        "dropping the scaling barely changes the logits"
    );
}

#[test]
#[ignore = "needs a GPU; skipped without FERROTENSOR_LLAMA3_FIXTURE"]
fn kv_cache_and_batched_forward_match_transformers() {
    let Some(fixture) = Fixture::load() else {
        return;
    };
    let gpu = gpu();
    let model = fixture.build(&gpu, None, false);
    let ids = &fixture.case.ids;
    let (_, expected) = fixture.expected("logits");

    let mut cache = model.new_kv_cache(ids.len()).expect("all-attention model");
    let prompt = 5;
    let prefill = model.forward(&gpu.engine, &ids[..prompt], 0, Some(&mut cache), None);
    let mut decoded = host(&gpu, &prefill);
    for pos in prompt..ids.len() {
        let step = model.forward(&gpu.engine, &ids[pos..=pos], pos, Some(&mut cache), None);
        decoded.extend(host(&gpu, &step));
    }
    assert_close(&decoded, expected, REFERENCE_TOL, "cached decoding");

    let batched = model.forward_batch(&gpu.engine, &[ids.as_slice(), ids.as_slice()]);
    let batched = host(&gpu, &batched);
    let (first, second) = batched.split_at(expected.len());
    assert_close(first, expected, REFERENCE_TOL, "batch row 0");
    assert_close(second, expected, REFERENCE_TOL, "batch row 1");
}

#[test]
#[ignore = "needs a GPU; skipped without FERROTENSOR_LLAMA3_FIXTURE"]
fn f16_base_weights_stay_close_to_transformers() {
    let Some(fixture) = Fixture::load() else {
        return;
    };
    let gpu = gpu();
    let model = fixture.build(&gpu, None, true);
    let actual = host(
        &gpu,
        &model.forward(&gpu.engine, &fixture.case.ids, 0, None, None),
    );
    // The bound of `f16_base_weights_stay_close_to_f32` in
    // model_correctness.rs; norms stay f32.
    let (_, expected) = fixture.expected("logits");
    assert_close(&actual, expected, 1e-2, "f16 logits vs transformers");
}

/// With B = 0 a LoRA adapter leaves the forward unchanged, and the gradient
/// of B is `scale * A^T (x^T dY)`, where `x^T dY` is the gradient of the
/// base weight in Ferrotensor's `[in, out]` layout: transformers' weight
/// gradient (`[out, in]`) gives the expected value of every B.
#[test]
#[ignore = "needs a GPU; skipped without FERROTENSOR_LLAMA3_FIXTURE"]
fn lora_gradients_match_transformers() {
    let Some(fixture) = Fixture::load() else {
        return;
    };
    let gpu = gpu();
    let lora = LoRaConfig {
        rank: 4,
        alpha: 8.0,
        target_modules: LINEARS
            .iter()
            .map(|m| m.rsplit('.').next().expect("module name").to_string())
            .collect(),
    };
    let scale = lora.alpha / lora.rank as f32;
    let model = fixture.build(&gpu, Some(lora), false);
    assert_eq!(model.lora_params().len(), 2 * 2 * LINEARS.len());

    let ids = &fixture.case.ids;
    let logits = model.forward(&gpu.engine, ids, 0, None, None);
    let mask = vec![1u8; ids.len()];
    let loss = masked_cross_entropy(&logits, &fixture.case.targets, &mask, &gpu.engine)
        .expect("supervised positions");
    assert!(
        (loss.value - fixture.case.loss).abs() <= ATOL,
        "loss {} vs transformers {}",
        loss.value,
        fixture.case.loss
    );
    loss.tensor.backward(&gpu.engine);

    for (name, linear) in linears(&model) {
        let adapter = linear.lora.as_ref().expect("adapter");
        let [in_dim, rank] = adapter.a.data.shape[..] else {
            panic!("{name}: A should be 2-D");
        };
        let out_dim = adapter.b.data.shape[1];
        let a = host(&gpu, &adapter.a);
        let (shape, weight_grad) = fixture.expected(&format!("grad.{name}"));
        assert_eq!(shape, &[out_dim, in_dim], "{name}: gradient shape");
        let expected: Vec<f32> = (0..rank)
            .flat_map(|r| {
                let a = &a;
                (0..out_dim).map(move |o| {
                    let dot: f32 = (0..in_dim)
                        .map(|i| a[i * rank + r] * weight_grad[o * in_dim + i])
                        .sum();
                    scale * dot
                })
            })
            .collect();
        let grad = adapter
            .b
            .grad
            .borrow()
            .clone()
            .unwrap_or_else(|| panic!("{name}: B has no gradient"));
        let actual = pollster::block_on(grad.to_vec(&gpu.ctx));
        assert!(
            expected.iter().any(|v| v.abs() > 1e-3),
            "{name}: expected gradient is too small to test"
        );
        for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= ATOL + RTOL * want.abs(),
                "{name}: grad B[{index}] {got} vs {want}"
            );
        }
    }
}

/// A real checkpoint against transformers on 8 fixed prompts, with f32 base
/// weights. It prints the largest and the mean absolute difference of the
/// logits before it asserts, so a failing run still reports them.
#[test]
#[ignore = "needs a GPU, a Llama 3 checkpoint and its reference logits"]
fn real_checkpoint_logits_match_transformers() {
    let (Some(model_dir), Some(reference)) = (
        std::env::var_os("FERROTENSOR_LLAMA3_MODEL").map(PathBuf::from),
        std::env::var_os("FERROTENSOR_LLAMA3_REAL").map(PathBuf::from),
    ) else {
        eprintln!(
            "skipping the real-checkpoint check; set FERROTENSOR_LLAMA3_MODEL and FERROTENSOR_LLAMA3_REAL (the output of scripts/gen_llama3_real_logits.py)"
        );
        return;
    };
    #[derive(Deserialize)]
    struct Prompts {
        ids: Vec<Vec<u32>>,
    }
    let prompts: Prompts = serde_json::from_str(
        &std::fs::read_to_string(reference.join("cases.json")).expect("read cases.json"),
    )
    .expect("parse cases.json");
    let bytes = std::fs::read(reference.join("logits.safetensors")).expect("read logits");
    let file = safetensors::SafeTensors::deserialize(&bytes).expect("parse logits");

    let gpu = gpu();
    let config = ModelLoader::load_config(model_dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&model_dir).expect("weights");
    let model = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .build()
        .expect("build");

    let (mut worst, mut sum, mut count) = (0f32, 0f64, 0usize);
    for (n, ids) in prompts.ids.iter().enumerate() {
        let view = file
            .tensor(&format!("logits.{n}"))
            .expect("reference logits");
        let expected: Vec<f32> = view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let actual = host(&gpu, &model.forward(&gpu.engine, ids, 0, None, None));
        assert_eq!(actual.len(), expected.len(), "prompt {n}: logits shape");
        assert!(
            actual.iter().all(|v| v.is_finite()),
            "prompt {n}: non-finite"
        );
        let diff = max_abs_diff(&actual, &expected);
        let mean = actual
            .iter()
            .zip(&expected)
            .map(|(a, e)| f64::from((a - e).abs()))
            .sum::<f64>()
            / actual.len() as f64;
        eprintln!(
            "prompt {n}: {} tokens, max abs diff {diff:e}, mean abs diff {mean:e}",
            ids.len()
        );
        worst = worst.max(diff);
        sum += mean;
        count += 1;
    }
    eprintln!(
        "LLAMA3_REAL_LOGITS max_abs_diff={worst:e} mean_abs_diff={:e} prompts={count}",
        sum / count as f64
    );
    assert!(
        worst <= REFERENCE_TOL,
        "max abs diff {worst} > {REFERENCE_TOL}"
    );
}

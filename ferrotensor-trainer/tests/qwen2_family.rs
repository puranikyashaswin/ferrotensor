//! A Qwen2 checkpoint (biased query/key/value projections, tied embeddings,
//! RoPE base 1e6, head size 16) against Hugging Face transformers.
//!
//! `scripts/gen_qwen2_fixture.py` builds a tiny random `Qwen2ForCausalLM`,
//! saves it, and records its logits and the gradients of a cross-entropy
//! loss with respect to every decoder-block linear weight. These tests load
//! the saved checkpoint through [`Lfm2ModelBuilder`] and compare.
//!
//! They need a GPU, so they are `#[ignore]`d by default. Without
//! `FERROTENSOR_QWEN2_FIXTURE` they print a message and skip; CI sets it:
//!
//! ```text
//! python scripts/gen_qwen2_fixture.py /tmp/qwen2-fixture
//! FERROTENSOR_QWEN2_FIXTURE=/tmp/qwen2-fixture \
//!   cargo test --release -p ferrotensor-trainer --test qwen2_family -- --ignored --test-threads=1
//! ```

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
/// Llama parity tests (`model_correctness.rs`): both sides compute in f32,
/// so they differ only in summation order and in `exp`, `sin` and `cos`,
/// about 1e-5 on logits of order 1, while a dropped bias moves them by more
/// than 0.05 (checked below).
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
    /// `None`, with a message, when `FERROTENSOR_QWEN2_FIXTURE` is unset, so the
    /// suite still runs on a machine without PyTorch. A path that is set but
    /// wrong fails, which keeps CI strict.
    fn load() -> Option<Self> {
        let Some(dir) = std::env::var_os("FERROTENSOR_QWEN2_FIXTURE").map(PathBuf::from) else {
            eprintln!(
                "skipping Qwen2 parity; set FERROTENSOR_QWEN2_FIXTURE to the output of scripts/gen_qwen2_fixture.py"
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
#[ignore = "needs a GPU; skipped without FERROTENSOR_QWEN2_FIXTURE"]
fn logits_match_transformers() {
    let Some(fixture) = Fixture::load() else {
        return;
    };
    let gpu = gpu();
    let mut model = fixture.build(&gpu, None, false);
    for (name, linear) in linears(&model) {
        let biased = ["q_proj", "k_proj", "v_proj"]
            .iter()
            .any(|p| name.contains(p));
        assert_eq!(linear.bias.is_some(), biased, "bias of {name}");
    }
    // Tied: no lm_head.weight in the file, so the head is embed_tokens^T.
    assert_eq!(model.lm_head.weight.data.shape, vec![64, 96]);

    let (shape, expected) = fixture.expected("logits");
    let logits = model.forward(&gpu.engine, &fixture.case.ids, 0, None, None);
    assert_eq!(&logits.data.shape, shape);
    let actual = host(&gpu, &logits);
    assert_close(&actual, expected, REFERENCE_TOL, "logits vs transformers");

    // The biases matter, or the comparison above would not prove they load.
    for layer in &mut model.layers {
        if let Lfm2LayerOp::Attention(attn) = &mut layer.op {
            for proj in [&mut attn.q_proj, &mut attn.k_proj, &mut attn.v_proj] {
                proj.bias = None;
            }
        }
    }
    let unbiased = host(
        &gpu,
        &model.forward(&gpu.engine, &fixture.case.ids, 0, None, None),
    );
    assert!(
        max_abs_diff(&unbiased, expected) > 0.05,
        "dropping the biases barely changes the logits"
    );
}

#[test]
#[ignore = "needs a GPU; skipped without FERROTENSOR_QWEN2_FIXTURE"]
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
#[ignore = "needs a GPU; skipped without FERROTENSOR_QWEN2_FIXTURE"]
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
    // model_correctness.rs; biases and norms stay f32.
    let (_, expected) = fixture.expected("logits");
    assert_close(&actual, expected, 1e-2, "f16 logits vs transformers");
}

/// With B = 0 a LoRA adapter leaves the forward unchanged, and the gradient
/// of B is `scale * A^T (x^T dY)`, where `x^T dY` is the gradient of the
/// base weight in Ferrotensor's `[in, out]` layout: transformers' weight
/// gradient (`[out, in]`) gives the expected value of every B.
#[test]
#[ignore = "needs a GPU; skipped without FERROTENSOR_QWEN2_FIXTURE"]
fn lora_gradients_match_transformers_with_frozen_biases() {
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
        if let Some(bias) = &linear.bias {
            assert!(!bias.requires_grad, "{name}: bias trains");
            assert!(bias.grad.borrow().is_none(), "{name}: bias got a gradient");
        }
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
        let actual = pollster::block_on(grad.to_vec(&gpu.ctx)).expect("GPU readback");
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

//! Gradient parity with PyTorch autograd for LoRA training.
//!
//! `scripts/gen_gradient_fixtures.py` builds the tiny Llama-style model of
//! `ferrotensor-graph/tests/common` in plain PyTorch (the LoRA math written
//! out, no PEFT), with non-zero LoRA factors on every linear layer, and
//! stores its logits, loss and the autograd gradient of every trainable
//! tensor in `ferrotensor-graph/tests/fixtures/gradient_parity`. These tests
//! run the same batches through [`Lfm2Model`], [`masked_cross_entropy`] /
//! [`masked_cross_entropy_batch`] and backward on the GPU, and compare.
//!
//! `model_correctness.rs` in ferrotensor-graph checks that the fixture's
//! weights are the Rust fixture's and that its logits equal the CPU
//! reference there, so the PyTorch side is tied to an independent forward.
//!
//! These need a GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test --release -p ferrotensor-trainer --test gradient_parity -- --ignored --test-threads=1
//! ```

use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_graph::Linear;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::{masked_cross_entropy, masked_cross_entropy_batch};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// An element passes when `|actual - expected| <= ATOL + RTOL * |expected|`
/// (the `torch.allclose` rule).
///
/// Both sides compute in f32 from the same weights. Recomputing the PyTorch
/// side in f64 moves its gradients by at most 7e-7 and its loss by 2e-7, so
/// that is the size of f32 rounding here; the GPU adds different summation
/// orders (blocked matmuls, online softmax in the attention kernels) and its
/// own `exp`, which are of the same order. The gradients are of order 0.1
/// to 1, so 1e-4 absolute plus 1e-3 relative leaves about two orders of
/// magnitude of headroom, while a real backward mistake (a missing LoRA
/// scale, a transposed product, a dropped residual or norm term, the wrong
/// loss normalisation) changes gradients by 1e-2 or more.
const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-3;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../ferrotensor-graph/tests/fixtures/gradient_parity")
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Meta {
    lora_rank: usize,
    lora_alpha: f32,
    cases: HashMap<String, Case>,
}

#[derive(Deserialize)]
struct Case {
    rows: Vec<Vec<u32>>,
    targets: Vec<Vec<u32>>,
    masks: Vec<Vec<u8>>,
    trainable_base: bool,
    loss: f32,
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

struct Fixture {
    meta: Meta,
    adapters: Tensors,
    expected: Tensors,
}

impl Fixture {
    fn load() -> Self {
        let dir = fixture_dir();
        let meta = std::fs::read_to_string(dir.join("cases.json")).expect("read cases.json");
        Self {
            meta: serde_json::from_str(&meta).expect("parse cases.json"),
            adapters: read_tensors(&dir.join("adapters.safetensors")),
            expected: read_tensors(&dir.join("expected.safetensors")),
        }
    }

    fn expected(&self, key: &str) -> &(Vec<usize>, Vec<f32>) {
        self.expected
            .get(key)
            .unwrap_or_else(|| panic!("fixture has no {key}"))
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct Gpu {
    ctx: Arc<WgpuContext>,
    engine: Arc<ComputeEngine>,
}

fn gpu() -> Gpu {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = Arc::new(ComputeEngine::new(&ctx));
    Gpu { ctx, engine }
}

fn host(gpu: &Gpu, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&gpu.ctx))
}

/// The fixture checkpoint with an adapter on every linear layer and the LM
/// head; base weights frozen unless `trainable_base`.
fn build(gpu: &Gpu, meta: &Meta, trainable_base: bool) -> Lfm2Model {
    let dir = fixture_dir();
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(&dir).expect("weights");
    let lora = LoRaConfig {
        rank: meta.lora_rank,
        alpha: meta.lora_alpha,
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
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_lora(lora)
        .with_trainable_base(trainable_base)
        .build()
        .expect("build")
}

/// Every linear layer with the Hugging Face name prefix of its weight.
fn linears(model: &Lfm2Model) -> Vec<(String, &Linear)> {
    let mut out = Vec::new();
    for (l, layer) in model.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        let Lfm2LayerOp::Attention(attn) = &layer.op else {
            panic!("layer {l} should be attention");
        };
        out.push((format!("{p}.self_attn.q_proj"), &attn.q_proj));
        out.push((format!("{p}.self_attn.k_proj"), &attn.k_proj));
        out.push((format!("{p}.self_attn.v_proj"), &attn.v_proj));
        out.push((format!("{p}.self_attn.o_proj"), &attn.o_proj));
        out.push((format!("{p}.mlp.gate_proj"), &layer.mlp.gate_proj));
        out.push((format!("{p}.mlp.up_proj"), &layer.mlp.up_proj));
        out.push((format!("{p}.mlp.down_proj"), &layer.mlp.down_proj));
    }
    out.push(("lm_head".to_string(), &model.lm_head));
    out
}

/// Every tensor that can hold a gradient, under the fixture's names:
/// linear weights (`[in, out]`), LoRA factors, norm scales, embedding.
fn named_tensors(model: &Lfm2Model) -> Vec<(String, Tensor)> {
    let mut out = vec![(
        "model.embed_tokens.weight".to_string(),
        model.embed_tokens.clone(),
    )];
    for (l, layer) in model.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        out.push((
            format!("{p}.input_layernorm.weight"),
            layer.operator_norm.weight.clone(),
        ));
        out.push((
            format!("{p}.post_attention_layernorm.weight"),
            layer.ffn_norm.weight.clone(),
        ));
    }
    out.push(("model.norm.weight".to_string(), model.norm_f.weight.clone()));
    for (prefix, linear) in linears(model) {
        out.push((format!("{prefix}.weight"), linear.weight.clone()));
        let lora = linear
            .lora
            .as_ref()
            .unwrap_or_else(|| panic!("{prefix} has no adapter"));
        out.push((format!("{prefix}.lora_a"), lora.a.clone()));
        out.push((format!("{prefix}.lora_b"), lora.b.clone()));
    }
    out
}

/// Replace the fresh adapters (B = 0) with the fixture's non-zero factors.
fn load_adapters(gpu: &Gpu, model: &mut Lfm2Model, adapters: &Tensors) {
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
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

struct Diff {
    name: String,
    max_abs: f32,
    max_rel: f32,
    /// Largest `|a - e| / (ATOL + RTOL * |e|)`; the check passes at <= 1.
    worst: f32,
    expected_max: f32,
}

impl Diff {
    /// Every element within tolerance (false for NaN).
    fn passes(&self) -> bool {
        self.worst <= 1.0
    }
}

fn diff(name: &str, actual: &[f32], expected: &[f32]) -> Diff {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    let mut d = Diff {
        name: name.to_string(),
        max_abs: 0.0,
        max_rel: 0.0,
        worst: 0.0,
        expected_max: 0.0,
    };
    for (&a, &e) in actual.iter().zip(expected) {
        let err = (a - e).abs();
        if !err.is_finite() {
            d.worst = f32::INFINITY;
        }
        d.max_abs = d.max_abs.max(err);
        // Relative error is only meaningful away from zero.
        if e.abs() > 1e-3 {
            d.max_rel = d.max_rel.max(err / e.abs());
        }
        d.worst = d.worst.max(err / (ATOL + RTOL * e.abs()));
        d.expected_max = d.expected_max.max(e.abs());
    }
    d
}

/// Print every diff (to the real stderr, so it shows in CI logs even for
/// passing tests) and fail listing every tensor out of tolerance.
fn report(case: &str, diffs: &[Diff]) {
    let mut text = format!(
        "\ngradient parity [{case}] vs PyTorch, pass if |a-e| <= {ATOL:e} + {RTOL:e}*|e|\n\
         {:<52} {:>10} {:>10} {:>10} {:>8}\n",
        "tensor", "max|e|", "max abs", "max rel", "worst"
    );
    for d in diffs {
        text += &format!(
            "{:<52} {:>10.3e} {:>10.3e} {:>10.3e} {:>8.4}{}\n",
            d.name,
            d.expected_max,
            d.max_abs,
            d.max_rel,
            d.worst,
            if d.passes() { "" } else { "  FAIL" }
        );
    }
    let grads: Vec<&Diff> = diffs
        .iter()
        .filter(|d| d.name != "loss" && d.name != "logits")
        .collect();
    if let Some(w) = grads.iter().max_by(|a, b| a.worst.total_cmp(&b.worst)) {
        text += &format!(
            "worst gradient: {} (max abs {:.3e}, max rel {:.3e}); {} gradients compared\n",
            w.name,
            w.max_abs,
            w.max_rel,
            grads.len()
        );
    }
    let _ = std::io::stderr().write_all(text.as_bytes());
    let failing: Vec<&str> = diffs
        .iter()
        .filter(|d| !d.passes())
        .map(|d| d.name.as_str())
        .collect();
    assert!(
        failing.is_empty(),
        "[{case}] out of tolerance: {failing:?}{text}"
    );
}

/// Forward, loss and backward for one fixture case, compared with PyTorch.
fn check_case(case_name: &str) {
    let fixture = Fixture::load();
    let case = fixture
        .meta
        .cases
        .get(case_name)
        .unwrap_or_else(|| panic!("fixture has no case {case_name}"));
    let gpu = gpu();
    let mut model = build(&gpu, &fixture.meta, case.trainable_base);
    load_adapters(&gpu, &mut model, &fixture.adapters);

    let (logits, loss, value) = if let [row] = &case.rows[..] {
        let logits = model.forward(&gpu.engine, row, 0, None, None);
        let loss = masked_cross_entropy(&logits, &case.targets[0], &case.masks[0], &gpu.engine)
            .expect("supervised positions");
        (logits, loss.tensor, loss.value)
    } else {
        assert!(model.supports_batching());
        let rows: Vec<&[u32]> = case.rows.iter().map(Vec::as_slice).collect();
        let logits = model.forward_batch(&gpu.engine, &rows);
        let loss = masked_cross_entropy_batch(
            &logits,
            &case.targets.concat(),
            &case.masks.concat(),
            rows.len(),
            &gpu.engine,
        )
        .expect("supervised positions");
        let value = loss.per_sequence.iter().map(|&(l, _)| l).sum::<f32>();
        (logits, loss.tensor, value)
    };
    loss.backward(&gpu.engine);

    let mut diffs = vec![diff("loss", &[value], &[case.loss])];
    let (shape, expected_logits) = fixture.expected(&format!("logits.{case_name}"));
    assert_eq!(&logits.data.shape, shape, "logits shape");
    diffs.push(diff("logits", &host(&gpu, &logits.data), expected_logits));

    let prefix = format!("grad.{case_name}.");
    let expected_count = fixture
        .expected
        .keys()
        .filter(|k| k.starts_with(&prefix))
        .count();
    for (name, tensor) in named_tensors(&model) {
        let grad = tensor.grad.borrow().clone();
        match (fixture.expected.get(&format!("{prefix}{name}")), grad) {
            (Some((shape, expected)), Some(grad)) => {
                assert_eq!(&grad.shape, shape, "{case_name}: gradient shape of {name}");
                diffs.push(diff(&name, &host(&gpu, &grad), expected));
            }
            (Some(_), None) => panic!("{case_name}: no gradient for {name}"),
            // Frozen base weights, and the embedding (its lookup runs on
            // the host, so it never gets a gradient; PyTorch keeps it
            // frozen too).
            (None, Some(_)) => panic!("{case_name}: {name} should be frozen but got a gradient"),
            (None, None) => {}
        }
    }
    assert_eq!(
        diffs.len() - 2,
        expected_count,
        "{case_name}: not every expected gradient was compared"
    );
    report(case_name, &diffs);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// One sequence through `forward` and `masked_cross_entropy`, every
/// position supervised: loss and the gradients of all 30 LoRA tensors.
#[test]
#[ignore = "needs a GPU"]
fn lora_gradients_match_pytorch_for_one_sequence() {
    check_case("single");
}

/// Two sequences through `forward_batch` and `masked_cross_entropy_batch`
/// (the batched training path): the loss is the sum of per-sequence means.
#[test]
#[ignore = "needs a GPU"]
fn lora_gradients_match_pytorch_for_a_batch() {
    check_case("batch");
}

/// The batch with ignored target positions, a different number in each
/// sequence, so each sequence's mean has its own normaliser.
#[test]
#[ignore = "needs a GPU"]
fn lora_gradients_match_pytorch_with_a_loss_mask() {
    check_case("masked");
}

/// The masked batch with trainable base weights: the gradients of every
/// linear weight and norm scale as well as the adapters.
#[test]
#[ignore = "needs a GPU"]
fn all_gradients_match_pytorch_with_a_trainable_base() {
    check_case("full");
}

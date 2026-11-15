//! `merge_adapter` on the tiny Llama-style model: what is copied, what is
//! changed and by how much, what is refused, and (with a GPU) that the
//! merged model computes the logits of base plus adapter.
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test merge_adapter
//! cargo test -p ferrotensor-trainer --test merge_adapter -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

mod common;

use common::*;
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{LoraCheckpoint, LoraMetadata};
use ferrotensor_trainer::merge::{merge_adapter, MERGED_WEIGHTS_FILE};
use ferrotensor_trainer::peft::PEFT_TARGET_MODULES;
use std::collections::HashMap;
use std::path::Path;

const RANK: usize = 2;
const ALPHA: f32 = 6.0;
const SCALE: f32 = ALPHA / RANK as f32;

fn metadata() -> LoraMetadata {
    LoraMetadata {
        rank: RANK,
        alpha: ALPHA,
        target_modules: Vec::new(),
        trained_modules: None,
    }
}

/// An adapter with random A and B on `(module, out, in)`.
fn adapter(modules: &[(String, usize, usize)]) -> LoraCheckpoint {
    let mut tensors = HashMap::new();
    for (seed, (module, out, inp)) in modules.iter().enumerate() {
        let seed = 100 + 2 * seed as u64;
        tensors.insert(
            format!("{module}.lora_A.weight"),
            (vec![*inp, RANK], pseudo_random(inp * RANK, seed, 0.5)),
        );
        tensors.insert(
            format!("{module}.lora_B.weight"),
            (vec![RANK, *out], pseudo_random(RANK * out, seed + 1, 0.5)),
        );
    }
    LoraCheckpoint {
        tensors,
        metadata: Some(metadata()),
    }
}

/// `SCALE * (A B)^T` as `[out, in]`, summed in the order of the formula.
fn expected_delta(ckpt: &LoraCheckpoint, module: &str) -> Vec<f32> {
    let (a_shape, a) = &ckpt.tensors[&format!("{module}.lora_A.weight")];
    let (b_shape, b) = &ckpt.tensors[&format!("{module}.lora_B.weight")];
    let (inp, out) = (a_shape[0], b_shape[1]);
    let mut delta = vec![0.0; out * inp];
    for o in 0..out {
        for i in 0..inp {
            let dot: f32 = (0..RANK).map(|r| a[i * RANK + r] * b[r * out + o]).sum();
            delta[o * inp + i] = SCALE * dot;
        }
    }
    delta
}

fn weights(dir: &Path) -> WeightsContext {
    let mut w = WeightsContext::new();
    w.load_model_dir(dir).expect("load weights");
    w
}

fn raw(w: &WeightsContext, name: &str) -> Vec<u8> {
    let mut out = Vec::new();
    w.read_bytes(name, |bytes, _| out = bytes.to_vec())
        .expect("read tensor");
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn merge_changes_adapted_weights_by_the_delta_and_copies_the_rest() {
    let base_dir = TempDir::new("merge-base");
    write_tiny_llama(base_dir.path());
    let out = TempDir::new("merge-out");
    let kv = HIDDEN / HEADS * KV_HEADS;
    let mut modules = Vec::new();
    for l in 0..LAYERS {
        modules.push((format!("model.layers.{l}.self_attn.q_proj"), HIDDEN, HIDDEN));
        modules.push((format!("model.layers.{l}.self_attn.v_proj"), kv, HIDDEN));
        // The name adapters had before Llama blocks used Hugging Face names.
        modules.push((
            format!("model.layers.{l}.feed_forward.down_proj"),
            HIDDEN,
            FFN,
        ));
    }
    // tie_word_embeddings is true in the tiny model.
    modules.push(("lm_head".to_string(), VOCAB, HIDDEN));
    let ckpt = adapter(&modules);

    let report = merge_adapter(base_dir.path(), &ckpt, out.path()).expect("merge");
    let base = weights(base_dir.path());
    let merged = weights(out.path());
    let base_count = base.tensor_names().count();
    assert_eq!(report.weights, out.path().join(MERGED_WEIGHTS_FILE));
    assert_eq!(report.merged, 3 * LAYERS + 1);
    assert_eq!(report.copied, base_count - 3 * LAYERS);
    assert_eq!(merged.tensor_names().count(), base_count + 1);

    let adapted: HashMap<String, String> = modules
        .iter()
        .map(|(m, _, _)| {
            let target = m.replace(".feed_forward.", ".mlp.");
            let source = if m == "lm_head" {
                "model.embed_tokens.weight".to_string()
            } else {
                format!("{target}.weight")
            };
            (format!("{target}.weight"), source)
        })
        .collect();
    for name in base.tensor_names() {
        if !adapted.contains_key(name) {
            assert_eq!(raw(&merged, name), raw(&base, name), "{name} is not copied");
        }
    }
    for (module, _, _) in &modules {
        let target = format!("{}.weight", module.replace(".feed_forward.", ".mlp."));
        let (before, shape) = base.read_f32_tensor(&adapted[&target]).unwrap();
        let (after, merged_shape) = merged.read_f32_tensor(&target).unwrap();
        assert_eq!(merged_shape, shape, "{target}");
        let diff: Vec<f32> = after.iter().zip(&before).map(|(a, b)| a - b).collect();
        // f32 sums of two products of values below 1, added to weights
        // below 1: rounding stays far below 1e-5.
        let err = max_abs_diff(&diff, &expected_delta(&ckpt, module));
        assert!(err < 1e-5, "{target}: off by {err}");
    }

    let config = |dir: &Path| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap()
    };
    let mut expected_config = config(base_dir.path());
    expected_config["tie_word_embeddings"] = serde_json::Value::Bool(false);
    assert_eq!(config(out.path()), expected_config);
    ModelLoader::load_config(out.path().join("config.json")).expect("the config loads");
}

#[test]
fn f16_weights_stay_f16() {
    let base_dir = TempDir::new("merge-f16");
    std::fs::write(base_dir.path().join("config.json"), "{}").unwrap();
    let f16_bytes = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect()
    };
    let w = f16_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let norm = f16_bytes(&[0.5, 0.25]);
    let views = HashMap::from([
        (
            "w.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2, 3], &w).unwrap(),
        ),
        (
            "norm.weight".to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![2], &norm).unwrap(),
        ),
    ]);
    std::fs::write(
        base_dir.path().join("model.safetensors"),
        safetensors::tensor::serialize(views, &None).unwrap(),
    )
    .unwrap();
    // A B = [[1, 2], [3, 4], [4, 6]] and scale 0.5, so by hand
    // W' = W + 0.5 * (A B)^T = [[1.5, 3.5, 5], [5, 7, 9]], exact in f16.
    let ckpt = LoraCheckpoint {
        tensors: HashMap::from([
            (
                "w.lora_A.weight".to_string(),
                (vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            ),
            (
                "w.lora_B.weight".to_string(),
                (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ),
        ]),
        metadata: Some(LoraMetadata {
            rank: 2,
            alpha: 1.0,
            ..metadata()
        }),
    };
    let out = TempDir::new("merge-f16-out");
    let report = merge_adapter(base_dir.path(), &ckpt, out.path()).expect("merge");
    assert_eq!((report.merged, report.copied), (1, 1));

    let merged = weights(out.path());
    let meta = merged.get_tensor_metadata("w.weight").unwrap();
    assert_eq!(meta.dtype, safetensors::Dtype::F16);
    assert_eq!(
        raw(&merged, "w.weight"),
        f16_bytes(&[1.5, 3.5, 5.0, 5.0, 7.0, 9.0])
    );
    assert_eq!(raw(&merged, "norm.weight"), norm);
}

#[test]
fn mismatches_and_quantized_bases_are_refused_before_writing() {
    let base_dir = TempDir::new("merge-refuse");
    write_tiny_llama(base_dir.path());
    let tmp = TempDir::new("merge-refuse-out");
    let out = tmp.path().join("merged");
    let err = |ckpt: &LoraCheckpoint, model: &Path, out: &Path| {
        merge_adapter(model, ckpt, out).unwrap_err().to_string()
    };
    let q = |out: usize, inp: usize| {
        adapter(&[("model.layers.0.self_attn.q_proj".to_string(), out, inp)])
    };

    let e = err(&q(HIDDEN, HIDDEN), base_dir.path(), base_dir.path());
    assert!(e.contains("is not empty"), "{e}");

    let e = err(&q(HIDDEN, FFN), base_dir.path(), &out);
    assert!(
        e.contains("model.layers.0.self_attn.q_proj.lora_A.weight"),
        "{e}"
    );
    assert!(e.contains(&format!("[{HIDDEN}, {HIDDEN}]")), "{e}");

    let missing = adapter(&[(
        "model.layers.0.self_attn.x_proj".to_string(),
        HIDDEN,
        HIDDEN,
    )]);
    let e = err(&missing, base_dir.path(), &out);
    assert!(
        e.contains("no tensor model.layers.0.self_attn.x_proj.weight"),
        "{e}"
    );

    let mut wrong_rank = q(HIDDEN, HIDDEN);
    wrong_rank.metadata = Some(LoraMetadata {
        rank: 4,
        ..metadata()
    });
    let e = err(&wrong_rank, base_dir.path(), &out);
    assert!(e.contains("metadata says 4"), "{e}");

    let mut trained = q(HIDDEN, HIDDEN);
    trained.metadata = Some(LoraMetadata::trained_weights("attn"));
    let e = err(&trained, base_dir.path(), &out);
    assert!(e.contains("not a LoRA adapter"), "{e}");

    let config_path = base_dir.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    config["quantization_config"] = serde_json::json!({ "load_in_4bit": true });
    std::fs::write(&config_path, config.to_string()).unwrap();
    let e = err(&q(HIDDEN, HIDDEN), base_dir.path(), &out);
    assert!(e.contains("quantized"), "{e}");

    assert!(!out.exists(), "nothing is written on error");
}

fn build(gpu: &Gpu, dir: &Path, lora: Option<LoRaConfig>) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let w = weights(dir);
    let builder = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &w, config);
    match lora {
        Some(lora) => builder.with_lora(lora),
        None => builder,
    }
    .build()
    .expect("build")
}

#[test]
#[ignore = "needs a GPU"]
fn merged_model_gives_the_logits_of_base_plus_adapter() {
    let base_dir = TempDir::new("merge-gpu");
    write_tiny_llama(base_dir.path());
    let gpu = gpu();
    let lora = LoRaConfig {
        rank: RANK,
        alpha: ALPHA,
        target_modules: PEFT_TARGET_MODULES.iter().map(|t| t.to_string()).collect(),
    };
    let model = build(&gpu, base_dir.path(), Some(lora));
    let (_, params) = sorted_lora_params(&model);
    let mut t = trainer(&gpu, model, 1e-2);
    let seq = pattern_sequence();
    for _ in 0..5 {
        t.train_step(&seq[..seq.len() - 1], &seq[1..], &params)
            .expect("step");
    }
    let mut tensors = HashMap::new();
    for (name, tensor) in t.model.named_lora_params() {
        let values = host(&gpu, &tensor);
        if name.contains("lora_B") {
            assert!(values.iter().any(|v| *v != 0.0), "{name} is still zero");
        }
        tensors.insert(name, (tensor.data.shape.clone(), values));
    }
    let ckpt = LoraCheckpoint {
        tensors,
        metadata: Some(metadata()),
    };

    let ids: Vec<u32> = (0..12).map(|i| ((i * 5 + 2) % VOCAB) as u32).collect();
    let logits = |model: &Lfm2Model| host(&gpu, &model.forward(&gpu.engine, &ids, 0, None, None));
    let with_adapter = logits(&t.model);
    let base = logits(&build(&gpu, base_dir.path(), None));

    let out = TempDir::new("merge-gpu-out");
    merge_adapter(base_dir.path(), &ckpt, out.path()).expect("merge");
    let merged = logits(&build(&gpu, out.path(), None));

    // Everything is f32: x (W + s A B) and x W + s (x A) B differ only in
    // the order of f32 operations, about 1e-7 relative per op over a few
    // hundred terms on logits of order 1.
    let err = max_abs_diff(&merged, &with_adapter);
    let effect = max_abs_diff(&base, &with_adapter);
    assert!(err < 1e-4, "merged vs base + adapter: {err}");
    assert!(
        effect > 1e-2,
        "the adapter changes the logits by only {effect}"
    );
}

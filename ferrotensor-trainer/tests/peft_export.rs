//! Hugging Face names for Llama models, and LoRA adapters exported to and
//! imported from the PEFT format, on a tiny Llama-style model.
//!
//! These need a GPU, so they are `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test peft_export -- --ignored --test-threads=1
//! ```
//!
//! `scripts/check_peft_export.py` checks the other half: the same exported
//! adapter loaded by PyTorch + PEFT gives the same logits.

mod common;

use common::*;
use ferrotensor_graph::lfm::{legacy_llama_name, Lfm2Model};
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata,
};
use ferrotensor_trainer::peft::{
    export_peft_adapter, load_peft_adapter, save_peft_adapter, PEFT_CONFIG_FILE,
    PEFT_TARGET_MODULES, PEFT_WEIGHTS_FILE,
};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

const RANK: usize = 4;
const ALPHA: f32 = 8.0;

fn lora(targets: &[&str]) -> LoRaConfig {
    LoRaConfig {
        rank: RANK,
        alpha: ALPHA,
        target_modules: targets.iter().map(|t| t.to_string()).collect(),
    }
}

fn build(gpu: &Gpu, dir: &Path, lora: LoRaConfig) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_lora(lora)
        .build()
        .expect("build")
}

/// A model with adapters on `targets`, trained a few steps so that every
/// `lora_B` is non-zero.
fn trained(gpu: &Gpu, dir: &Path, targets: &[&str]) -> ferrotensor_trainer::Trainer {
    let model = build(gpu, dir, lora(targets));
    let (_, params) = sorted_lora_params(&model);
    let mut t = trainer(gpu, model, 1e-2);
    let seq = pattern_sequence();
    for _ in 0..5 {
        t.train_step(&seq[..seq.len() - 1], &seq[1..], &params)
            .expect("step");
    }
    for (name, tensor) in t.model.named_lora_params() {
        if name.contains("lora_B") {
            let max = host(gpu, &tensor)
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(max > 0.0, "{name} is still zero after training");
        }
    }
    t
}

/// Token ids the model was not trained on.
fn probe_ids() -> Vec<u32> {
    (0..12).map(|i| ((i * 5 + 2) % VOCAB) as u32).collect()
}

fn logits(gpu: &Gpu, model: &Lfm2Model) -> Vec<f32> {
    host(
        gpu,
        &model.forward(&gpu.engine, &probe_ids(), 0, None, None),
    )
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
#[ignore = "needs a GPU"]
fn llama_models_use_hugging_face_parameter_names() {
    let dir = TempDir::new("hf-names");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build(&gpu, dir.path(), lora(&["q_proj", "gate_proj"]));

    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir.path()).expect("weights");
    // Every base weight is named exactly as in model.safetensors (lm_head is
    // tied to the embedding, so the file has no lm_head.weight).
    for name in model.named_params().keys() {
        if name.contains("lora_") || name == "lm_head.weight" {
            continue;
        }
        assert!(
            weights.get_tensor_info(name).is_some(),
            "{name} is not a Hugging Face name of the checkpoint"
        );
    }
    let lora_names = model.named_lora_params();
    assert!(lora_names.contains_key("model.layers.1.mlp.gate_proj.lora_A.weight"));
    assert!(lora_names.contains_key("model.layers.1.self_attn.q_proj.lora_B.weight"));
    assert!(lora_names.keys().all(|n| !n.contains("feed_forward")));
}

#[test]
#[ignore = "needs a GPU"]
fn peft_export_then_import_gives_identical_logits() {
    let dir = TempDir::new("peft-round-trip");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let t = trained(&gpu, dir.path(), &PEFT_TARGET_MODULES);
    let expected = logits(&gpu, &t.model);

    let peft_dir = dir.path().join("peft");
    let export = pollster::block_on(save_peft_adapter(&gpu.ctx, &t.model, "tiny", &peft_dir))
        .expect("export");
    assert_eq!(export.tensors, 2 * (LAYERS * 7 + 1));

    let ckpt = load_peft_adapter(&peft_dir).expect("import");
    let meta = ckpt.metadata.clone().expect("metadata");
    assert_eq!((meta.rank, meta.alpha), (RANK, ALPHA));
    let fresh = build(
        &gpu,
        dir.path(),
        LoRaConfig {
            rank: meta.rank,
            alpha: meta.alpha,
            target_modules: meta.target_modules.clone(),
        },
    );
    let untrained = logits(&gpu, &fresh);
    let n = apply_checkpoint(&gpu.ctx, &ckpt, &fresh.named_lora_params()).expect("apply");
    assert_eq!(n, export.tensors);
    let restored = logits(&gpu, &fresh);

    assert!(
        max_abs_diff(&untrained, &expected) > 1e-3,
        "the adapter should change the logits"
    );
    // Transposing twice is exact, so the logits are bit for bit the same.
    assert_eq!(bits(&restored), bits(&expected));
}

#[test]
#[ignore = "needs a GPU"]
fn peft_export_has_exactly_peft_keys_shapes_and_config() {
    let dir = TempDir::new("peft-keys");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let t = trained(&gpu, dir.path(), &PEFT_TARGET_MODULES);
    let peft_dir = dir.path().join("peft");
    pollster::block_on(save_peft_adapter(
        &gpu.ctx,
        &t.model,
        "tiny-llama",
        &peft_dir,
    ))
    .expect("export");

    // What PEFT's get_peft_model_state_dict gives for LlamaForCausalLM.
    let kv = HIDDEN / HEADS * KV_HEADS;
    let mut expected: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let modules = [
        ("self_attn", "q_proj", HIDDEN, HIDDEN),
        ("self_attn", "k_proj", HIDDEN, kv),
        ("self_attn", "v_proj", HIDDEN, kv),
        ("self_attn", "o_proj", HIDDEN, HIDDEN),
        ("mlp", "gate_proj", HIDDEN, FFN),
        ("mlp", "up_proj", HIDDEN, FFN),
        ("mlp", "down_proj", FFN, HIDDEN),
    ];
    for l in 0..LAYERS {
        for (group, module, fan_in, fan_out) in modules {
            let p = format!("base_model.model.model.layers.{l}.{group}.{module}");
            expected.insert(format!("{p}.lora_A.weight"), vec![RANK, fan_in]);
            expected.insert(format!("{p}.lora_B.weight"), vec![fan_out, RANK]);
        }
    }
    expected.insert(
        "base_model.model.lm_head.lora_A.weight".into(),
        vec![RANK, HIDDEN],
    );
    expected.insert(
        "base_model.model.lm_head.lora_B.weight".into(),
        vec![VOCAB, RANK],
    );

    let bytes = std::fs::read(peft_dir.join(PEFT_WEIGHTS_FILE)).expect("weights file");
    let (_, header) = safetensors::tensor::SafeTensors::read_metadata(&bytes).expect("header");
    assert_eq!(
        header
            .metadata()
            .as_ref()
            .and_then(|m| m.get("format"))
            .map(String::as_str),
        Some("pt")
    );
    let file = safetensors::tensor::SafeTensors::deserialize(&bytes).expect("safetensors");
    let found: BTreeMap<String, Vec<usize>> = file
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(view.dtype(), safetensors::Dtype::F32, "{name}");
            (name, view.shape().to_vec())
        })
        .collect();
    assert_eq!(found, expected);

    // Values are the Ferrotensor tensors transposed: PEFT lora_A[r][i] is
    // Ferrotensor a[i][r].
    let a = host(
        &gpu,
        &t.model.named_lora_params()["model.layers.1.mlp.down_proj.lora_A.weight"],
    );
    let peft_a = load_lora_checkpoint(peft_dir.join(PEFT_WEIGHTS_FILE))
        .expect("load")
        .tensors["base_model.model.model.layers.1.mlp.down_proj.lora_A.weight"]
        .1
        .clone();
    for r in 0..RANK {
        for i in 0..FFN {
            assert_eq!(peft_a[r * FFN + i], a[i * RANK + r]);
        }
    }

    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(peft_dir.join(PEFT_CONFIG_FILE)).expect("config"),
    )
    .expect("json");
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!(config["task_type"], "CAUSAL_LM");
    assert_eq!(config["base_model_name_or_path"], "tiny-llama");
    assert_eq!(config["r"], RANK);
    assert_eq!(config["lora_alpha"], 8);
    assert_eq!(config["bias"], "none");
    assert_eq!(config["use_dora"], false);
    assert_eq!(config["use_rslora"], false);
    assert_eq!(config["fan_in_fan_out"], false);
    let mut targets: Vec<&str> = PEFT_TARGET_MODULES.to_vec();
    targets.sort();
    assert_eq!(config["target_modules"], serde_json::json!(targets));
}

#[test]
#[ignore = "needs a GPU"]
fn adapter_with_old_llama_names_still_loads_with_identical_logits() {
    let dir = TempDir::new("old-names");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let targets = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"];
    let t = trained(&gpu, dir.path(), &targets);
    let expected = logits(&gpu, &t.model);

    // The same tensors under the names earlier releases wrote.
    let named = t.model.named_lora_params();
    let old: HashMap<String, _> = named
        .iter()
        .map(|(k, v)| (legacy_llama_name(k).unwrap_or_else(|| k.clone()), v.clone()))
        .collect();
    assert!(old.keys().any(|k| k.contains(".feed_forward.")));
    assert!(old.keys().all(|k| !k.contains(".mlp.")));
    let meta = LoraMetadata {
        rank: RANK,
        alpha: ALPHA,
        target_modules: targets.iter().map(|t| t.to_string()).collect(),
    };
    let path = dir.path().join("old.safetensors");
    pollster::block_on(save_lora_checkpoint(&gpu.ctx, &path, &old, Some(&meta))).expect("save");

    let ckpt = load_lora_checkpoint(&path).expect("load");
    let fresh = build(&gpu, dir.path(), lora(&targets));
    let n = apply_checkpoint(&gpu.ctx, &ckpt, &fresh.named_lora_params()).expect("apply");
    assert_eq!(n, named.len());
    assert_eq!(bits(&logits(&gpu, &fresh)), bits(&expected));

    // An old-name file exports to PEFT and imports back to the same logits.
    let config = ModelLoader::load_config(dir.path().join("config.json")).expect("config");
    let peft_dir = dir.path().join("peft");
    export_peft_adapter(&ckpt, &config, "tiny", &peft_dir).expect("export old names");
    let again = build(&gpu, dir.path(), lora(&targets));
    let imported = load_peft_adapter(&peft_dir).expect("import");
    apply_checkpoint(&gpu.ctx, &imported, &again.named_lora_params()).expect("apply peft");
    assert_eq!(bits(&logits(&gpu, &again)), bits(&expected));

    // A tensor the model does not have is an error, not a silent skip.
    let mut extra = ckpt.clone();
    extra.tensors.insert(
        "model.layers.0.self_attn.k_proj.lora_A.weight".into(),
        (vec![HIDDEN, RANK], vec![0.0; HIDDEN * RANK]),
    );
    let err = apply_checkpoint(&gpu.ctx, &extra, &fresh.named_lora_params()).expect_err("extra");
    assert!(err.contains("k_proj"), "{err}");

    // So is a file with both the old and the new name of one tensor.
    let mut both = ckpt.clone();
    let (old_name, value) = both
        .tensors
        .iter()
        .find(|(k, _)| k.contains(".feed_forward."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .expect("an old name");
    let new_name = old_name.replacen(".feed_forward.", ".mlp.", 1);
    both.tensors.insert(new_name, value);
    let err = apply_checkpoint(&gpu.ctx, &both, &fresh.named_lora_params()).expect_err("both");
    assert!(err.contains("old name"), "{err}");
}

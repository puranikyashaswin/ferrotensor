//! LoRA adapters in the Hugging Face PEFT format.
//!
//! A PEFT adapter is a directory with `adapter_model.safetensors` and
//! `adapter_config.json`; `peft.PeftModel.from_pretrained(base, dir)` loads
//! it onto the same base model in PyTorch. [`export_peft_adapter`] and
//! [`save_peft_adapter`] write one from a Ferrotensor adapter of a
//! Llama-style model; [`load_peft_adapter`] reads one back as a
//! [`LoraCheckpoint`] for [`apply_checkpoint`](crate::checkpoint::apply_checkpoint).
//!
//! The two formats differ in layout, not in values:
//!
//! | | Ferrotensor | PEFT |
//! |---|---|---|
//! | key | `model.layers.0.self_attn.q_proj.lora_A.weight` | `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` |
//! | `lora_A` | `[in, rank]` | `[rank, in]` |
//! | `lora_B` | `[rank, out]` | `[out, rank]` |
//! | scale | `alpha / rank` | `lora_alpha / r` |
//!
//! Only adapters PEFT can represent on a Llama model are accepted: the
//! projections `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
//! `up_proj`, `down_proj` and `lm_head`, with one rank and alpha for all of
//! them, on every layer. Anything else is an error, never a silent skip.

use crate::checkpoint::{LoraCheckpoint, LoraMetadata};
use ferrotensor_backend::WgpuContext;
use ferrotensor_graph::lfm::{llama_name_from_legacy, Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::Architecture;
use ferrotensor_graph::Linear;
use ferrotensor_loader::Lfm2Config;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// File name of the adapter weights inside a PEFT adapter directory.
pub const PEFT_WEIGHTS_FILE: &str = "adapter_model.safetensors";
/// File name of the adapter settings inside a PEFT adapter directory.
pub const PEFT_CONFIG_FILE: &str = "adapter_config.json";
/// Modules whose LoRA adapters can be exported to PEFT.
pub const PEFT_TARGET_MODULES: [&str; 8] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
];

/// Prefix PEFT puts in front of the base model's module path.
const PEFT_PREFIX: &str = "base_model.model.";

/// What [`export_peft_adapter`] wrote.
#[derive(Clone, Debug, PartialEq)]
pub struct PeftExport {
    /// Adapter rank (`r` in `adapter_config.json`).
    pub rank: usize,
    /// Scale numerator (`lora_alpha`).
    pub alpha: f32,
    /// Adapted modules (`target_modules`), sorted.
    pub target_modules: Vec<String>,
    /// Number of tensors in `adapter_model.safetensors`.
    pub tensors: usize,
}

/// Whether `path` is a PEFT adapter directory (it holds `adapter_config.json`).
pub fn is_peft_adapter_dir(path: &Path) -> bool {
    path.is_dir() && path.join(PEFT_CONFIG_FILE).is_file()
}

/// One LoRA tensor of a Llama model, parsed from its Ferrotensor name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LoraSlot {
    /// Block index, `None` for `lm_head`.
    layer: Option<usize>,
    /// Index into [`PEFT_TARGET_MODULES`].
    module: usize,
    /// `false` for `lora_A`, `true` for `lora_B`.
    is_b: bool,
}

impl LoraSlot {
    /// Parse a Hugging Face Llama module path plus `.lora_{A,B}.weight`.
    fn parse(name: &str) -> Option<Self> {
        let (path, is_b) = if let Some(p) = name.strip_suffix(".lora_A.weight") {
            (p, false)
        } else {
            (name.strip_suffix(".lora_B.weight")?, true)
        };
        if path == "lm_head" {
            return Some(Self {
                layer: None,
                module: PEFT_TARGET_MODULES.len() - 1,
                is_b,
            });
        }
        let rest = path.strip_prefix("model.layers.")?;
        let (layer, rest) = rest.split_once('.')?;
        if layer.is_empty() || !layer.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let layer = layer.parse().ok()?;
        let (group, module) = rest.split_once('.')?;
        let module = PEFT_TARGET_MODULES[..7].iter().position(|m| *m == module)?;
        let expected_group = if module < 4 { "self_attn" } else { "mlp" };
        (group == expected_group).then_some(Self {
            layer: Some(layer),
            module,
            is_b,
        })
    }

    /// The Hugging Face module path (`model.layers.0.self_attn.q_proj`).
    fn module_path(&self) -> String {
        let module = PEFT_TARGET_MODULES[self.module];
        match self.layer {
            None => module.to_string(),
            Some(l) => {
                let group = if self.module < 4 { "self_attn" } else { "mlp" };
                format!("model.layers.{l}.{group}.{module}")
            }
        }
    }

    fn suffix(&self) -> &'static str {
        if self.is_b {
            "lora_B.weight"
        } else {
            "lora_A.weight"
        }
    }

    /// Ferrotensor name (Hugging Face names for Llama models).
    fn ferrotensor_name(&self) -> String {
        format!("{}.{}", self.module_path(), self.suffix())
    }

    /// Key in `adapter_model.safetensors`.
    fn peft_name(&self) -> String {
        format!("{PEFT_PREFIX}{}.{}", self.module_path(), self.suffix())
    }
}

/// Row-major `[rows, cols]` to `[cols, rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Check that `tensors` (Ferrotensor names, old or new, and layout) form an
/// adapter PEFT can represent, and return them keyed by slot plus the rank.
fn collect_slots(
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    n_layers: usize,
) -> Result<(BTreeMap<LoraSlot, (&Vec<usize>, &Vec<f32>)>, usize)> {
    if tensors.is_empty() {
        return Err("the adapter has no LoRA tensors".into());
    }
    let mut slots = BTreeMap::new();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    for name in names {
        let hf_name = llama_name_from_legacy(name).unwrap_or_else(|| name.clone());
        let slot = LoraSlot::parse(&hf_name).ok_or_else(|| {
            format!(
                "tensor {name} cannot be written as a PEFT adapter: only LoRA on {} is supported",
                PEFT_TARGET_MODULES.join(", ")
            )
        })?;
        if slot.layer.is_some_and(|l| l >= n_layers) {
            return Err(format!("tensor {name}: the model has only {n_layers} layers").into());
        }
        let (shape, values) = &tensors[name];
        if shape.len() != 2 || shape.iter().product::<usize>() != values.len() {
            return Err(format!("tensor {name} has shape {shape:?}, expected 2-D").into());
        }
        if slots.insert(slot, (shape, values)).is_some() {
            return Err(format!(
                "the adapter has {} under both its old and its new name",
                slot.ferrotensor_name()
            )
            .into());
        }
    }

    let mut rank = None;
    for (slot, (shape, _)) in &slots {
        let other = LoraSlot {
            is_b: !slot.is_b,
            ..*slot
        };
        if !slots.contains_key(&other) {
            return Err(format!(
                "{} has no matching {}",
                slot.ferrotensor_name(),
                other.suffix()
            )
            .into());
        }
        // A is [in, rank], B is [rank, out].
        let r = if slot.is_b { shape[0] } else { shape[1] };
        match rank {
            None => rank = Some(r),
            Some(r0) if r0 != r => {
                return Err(format!(
                    "{} has rank {r}, other adapters rank {r0}; PEFT export needs one rank",
                    slot.ferrotensor_name()
                )
                .into())
            }
            Some(_) => {}
        }
    }

    // PEFT applies `target_modules` to every layer, so a module must carry
    // adapters on all of them.
    for (module, module_name) in PEFT_TARGET_MODULES.iter().enumerate().take(7) {
        let layers = slots
            .keys()
            .filter(|s| s.module == module && !s.is_b)
            .count();
        if layers != 0 && layers != n_layers {
            return Err(format!(
                "{module_name} has adapters on {layers} of {n_layers} layers; PEFT export needs all of them"
            )
            .into());
        }
    }
    Ok((slots, rank.unwrap_or(0)))
}

/// `alpha` as JSON: an integer when it is one (PEFT declares `lora_alpha`
/// as an int), else a float.
fn alpha_json(alpha: f32) -> serde_json::Value {
    if alpha.fract() == 0.0 && alpha.abs() < 1e9 {
        serde_json::json!(alpha as i64)
    } else {
        serde_json::json!(alpha)
    }
}

fn write_peft(
    tensors: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    architecture: Architecture,
    n_layers: usize,
    alpha: f32,
    base_model: &str,
    dir: &Path,
) -> Result<PeftExport> {
    if architecture != Architecture::Llama {
        return Err(format!(
            "PEFT export supports Llama-style models; this is an {architecture:?} model"
        )
        .into());
    }
    let (slots, rank) = collect_slots(tensors, n_layers)?;

    let mut data = Vec::with_capacity(slots.len());
    for (slot, (shape, values)) in &slots {
        let (rows, cols) = (shape[0], shape[1]);
        let t = transpose(values, rows, cols);
        let bytes: Vec<u8> = t.iter().flat_map(|v| v.to_le_bytes()).collect();
        data.push((slot.peft_name(), vec![cols, rows], bytes));
    }
    let mut views = HashMap::new();
    for (name, shape, bytes) in &data {
        views.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)?,
        );
    }
    let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
    let weights = safetensors::tensor::serialize(views, &Some(metadata))?;

    let targets: BTreeSet<&str> = slots
        .keys()
        .map(|s| PEFT_TARGET_MODULES[s.module])
        .collect();
    let target_modules: Vec<String> = targets.iter().map(|s| s.to_string()).collect();
    let config = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "revision": null,
        "r": rank,
        "lora_alpha": alpha_json(alpha),
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "bias": "none",
        "fan_in_fan_out": false,
        "use_rslora": false,
        "use_dora": false,
        "init_lora_weights": true,
        "inference_mode": true,
        "modules_to_save": null,
        "layers_to_transform": null,
        "layers_pattern": null,
        "rank_pattern": {},
        "alpha_pattern": {},
    });

    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    for (file, bytes) in [
        (PEFT_WEIGHTS_FILE, weights),
        (
            PEFT_CONFIG_FILE,
            serde_json::to_string_pretty(&config)?.into_bytes(),
        ),
    ] {
        let path = dir.join(file);
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(PeftExport {
        rank,
        alpha,
        target_modules,
        tensors: slots.len(),
    })
}

/// Write `checkpoint` (a Ferrotensor adapter file, read with
/// [`load_lora_checkpoint`](crate::checkpoint::load_lora_checkpoint)) for
/// the model described by `config` as a PEFT adapter directory `dir`.
///
/// `base_model` goes into `base_model_name_or_path` (a Hub id or the path
/// of the base model). Adapters written with the old Llama-block names load
/// too. Fails, writing nothing, when the model is not Llama-style, the file
/// has no LoRA metadata, or a tensor is not one PEFT can represent.
pub fn export_peft_adapter(
    checkpoint: &LoraCheckpoint,
    config: &Lfm2Config,
    base_model: &str,
    dir: &Path,
) -> Result<PeftExport> {
    let meta = checkpoint.metadata.as_ref().ok_or(
        "the adapter file has no LoRA metadata (rank/alpha); it was not written by Ferrotensor",
    )?;
    write_peft_checked(
        checkpoint,
        Architecture::detect(config),
        config.num_hidden_layers,
        meta,
        base_model,
        dir,
    )
}

fn write_peft_checked(
    checkpoint: &LoraCheckpoint,
    architecture: Architecture,
    n_layers: usize,
    meta: &LoraMetadata,
    base_model: &str,
    dir: &Path,
) -> Result<PeftExport> {
    // Validate before anything is written.
    if architecture == Architecture::Llama {
        let (_, rank) = collect_slots(&checkpoint.tensors, n_layers)?;
        if rank != meta.rank {
            return Err(format!(
                "the adapter tensors have rank {rank}, its metadata says {}",
                meta.rank
            )
            .into());
        }
    }
    write_peft(
        &checkpoint.tensors,
        architecture,
        n_layers,
        meta.alpha,
        base_model,
        dir,
    )
}

/// Every linear layer of `model` that carries a LoRA adapter.
fn adapters(model: &Lfm2Model) -> Vec<&Linear> {
    let mut out = Vec::new();
    for layer in &model.layers {
        match &layer.op {
            Lfm2LayerOp::Attention(a) => {
                out.extend([&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj]);
            }
            Lfm2LayerOp::Conv(c) => out.extend([&c.in_proj, &c.out_proj]),
        }
        out.extend([
            &layer.mlp.gate_proj,
            &layer.mlp.up_proj,
            &layer.mlp.down_proj,
        ]);
    }
    out.push(&model.lm_head);
    out.retain(|l| l.lora.is_some());
    out
}

/// Write the LoRA adapters of `model` as a PEFT adapter directory `dir`
/// (see [`export_peft_adapter`]); rank and alpha are read from the model.
pub async fn save_peft_adapter(
    ctx: &Arc<WgpuContext>,
    model: &Lfm2Model,
    base_model: &str,
    dir: &Path,
) -> Result<PeftExport> {
    let mut settings = adapters(model)
        .into_iter()
        .filter_map(|l| l.lora.as_ref().map(|a| (a.rank, a.alpha.to_bits())));
    let Some((rank, alpha)) = settings.next() else {
        return Err("the model has no LoRA adapters".into());
    };
    if settings.any(|s| s != (rank, alpha)) {
        return Err(
            "the model's adapters differ in rank or alpha; PEFT export needs one of each".into(),
        );
    }
    let mut tensors = HashMap::new();
    for (name, tensor) in model.named_lora_params() {
        let values = tensor.data.to_vec(ctx).await;
        tensors.insert(name, (tensor.data.shape.clone(), values));
    }
    let checkpoint = LoraCheckpoint {
        tensors,
        metadata: None,
    };
    let meta = LoraMetadata {
        rank,
        alpha: f32::from_bits(alpha),
        target_modules: Vec::new(),
    };
    write_peft_checked(
        &checkpoint,
        model.architecture(),
        model.layers.len(),
        &meta,
        base_model,
        dir,
    )
}

/// The settings of `adapter_config.json` that Ferrotensor reads or rejects.
#[derive(serde::Deserialize)]
struct PeftConfig {
    peft_type: Option<String>,
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    bias: Option<String>,
    #[serde(default)]
    fan_in_fan_out: bool,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    use_dora: bool,
    #[serde(default)]
    lora_bias: bool,
    #[serde(default)]
    modules_to_save: Option<Vec<String>>,
    #[serde(default)]
    layers_to_transform: Option<serde_json::Value>,
    #[serde(default)]
    rank_pattern: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    alpha_pattern: Option<HashMap<String, serde_json::Value>>,
}

/// Read the PEFT adapter directory `dir` (as written by PEFT's
/// `save_pretrained` or by [`export_peft_adapter`]) as a [`LoraCheckpoint`]
/// with Ferrotensor names and layout, and metadata (rank, alpha, targets)
/// for building the model.
///
/// Fails on settings Ferrotensor cannot reproduce (DoRA, LoRA bias,
/// `fan_in_fan_out`, per-module ranks, `modules_to_save`, a subset of
/// layers) and on tensors other than LoRA on the Llama projections and
/// `lm_head`. With `use_rslora` the scale `lora_alpha / sqrt(r)` is kept
/// by storing `alpha = lora_alpha * sqrt(r)`.
pub fn load_peft_adapter(dir: &Path) -> Result<LoraCheckpoint> {
    let config_path = dir.join(PEFT_CONFIG_FILE);
    let text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let config: PeftConfig =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", config_path.display()))?;
    let unsupported = |what: &str| -> Box<dyn std::error::Error> {
        format!("{}: {what} is not supported", config_path.display()).into()
    };
    if config.peft_type.as_deref() != Some("LORA") {
        return Err(unsupported(&format!(
            "peft_type {:?} (only LORA)",
            config.peft_type
        )));
    }
    if config.bias.as_deref().is_some_and(|b| b != "none") || config.lora_bias {
        return Err(unsupported("a LoRA bias"));
    }
    if config.use_dora {
        return Err(unsupported("DoRA (use_dora)"));
    }
    if config.fan_in_fan_out {
        return Err(unsupported("fan_in_fan_out"));
    }
    if config
        .modules_to_save
        .as_ref()
        .is_some_and(|m| !m.is_empty())
    {
        return Err(unsupported("modules_to_save"));
    }
    if config
        .layers_to_transform
        .as_ref()
        .is_some_and(|v| !v.is_null())
    {
        return Err(unsupported("layers_to_transform"));
    }
    if config.rank_pattern.as_ref().is_some_and(|m| !m.is_empty())
        || config.alpha_pattern.as_ref().is_some_and(|m| !m.is_empty())
    {
        return Err(unsupported("rank_pattern / alpha_pattern"));
    }

    let weights_path = dir.join(PEFT_WEIGHTS_FILE);
    if !weights_path.is_file() {
        return Err(format!(
            "{} not found (only safetensors PEFT adapters are supported)",
            weights_path.display()
        )
        .into());
    }
    let file = crate::checkpoint::load_lora_checkpoint(&weights_path)?;
    let mut tensors = HashMap::new();
    let mut targets = BTreeSet::new();
    let mut names: Vec<&String> = file.tensors.keys().collect();
    names.sort();
    for name in names {
        let slot = name
            .strip_prefix(PEFT_PREFIX)
            .and_then(LoraSlot::parse)
            .ok_or_else(|| {
                format!(
                    "{}: tensor {name} is not LoRA on one of {}",
                    weights_path.display(),
                    PEFT_TARGET_MODULES.join(", ")
                )
            })?;
        let (shape, values) = &file.tensors[name];
        let [rows, cols] = shape[..] else {
            return Err(format!("tensor {name} has shape {shape:?}, expected 2-D").into());
        };
        // PEFT: A is [r, in], B is [out, r].
        let r = if slot.is_b { cols } else { rows };
        if r != config.r {
            return Err(format!(
                "tensor {name} has rank {r}, adapter_config.json says r = {}",
                config.r
            )
            .into());
        }
        targets.insert(PEFT_TARGET_MODULES[slot.module]);
        tensors.insert(
            slot.ferrotensor_name(),
            (vec![cols, rows], transpose(values, rows, cols)),
        );
    }
    if tensors.is_empty() {
        return Err(format!("{} has no tensors", weights_path.display()).into());
    }
    let alpha = if config.use_rslora {
        config.lora_alpha * (config.r as f32).sqrt()
    } else {
        config.lora_alpha
    };
    Ok(LoraCheckpoint {
        tensors,
        metadata: Some(LoraMetadata {
            rank: config.r,
            alpha,
            target_modules: targets.into_iter().map(str::to_string).collect(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYERS: usize = 2;

    /// A synthetic adapter: q_proj and gate_proj on every layer plus
    /// lm_head, hidden 3, rank 2, with distinct values per entry.
    fn checkpoint(old_names: bool) -> LoraCheckpoint {
        let mut tensors = HashMap::new();
        let mut seed = 0.0;
        let mut add = |name: String, shape: Vec<usize>| {
            let n = shape.iter().product::<usize>();
            let values = (0..n).map(|i| seed + i as f32).collect();
            seed += 100.0;
            tensors.insert(name, (shape, values));
        };
        let mlp = if old_names { "feed_forward" } else { "mlp" };
        for l in 0..LAYERS {
            add(
                format!("model.layers.{l}.self_attn.q_proj.lora_A.weight"),
                vec![3, 2],
            );
            add(
                format!("model.layers.{l}.self_attn.q_proj.lora_B.weight"),
                vec![2, 3],
            );
            add(
                format!("model.layers.{l}.{mlp}.gate_proj.lora_A.weight"),
                vec![3, 2],
            );
            add(
                format!("model.layers.{l}.{mlp}.gate_proj.lora_B.weight"),
                vec![2, 5],
            );
        }
        add("lm_head.lora_A.weight".into(), vec![3, 2]);
        add("lm_head.lora_B.weight".into(), vec![2, 7]);
        LoraCheckpoint {
            tensors,
            metadata: Some(LoraMetadata {
                rank: 2,
                alpha: 16.0,
                target_modules: vec!["q_proj".into(), "gate_proj".into(), "lm_head".into()],
            }),
        }
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("ferrotensor_peft_{}_{tag}", std::process::id()))
    }

    fn export(ckpt: &LoraCheckpoint, arch: Architecture, dir: &Path) -> Result<PeftExport> {
        let meta = ckpt.metadata.clone().unwrap();
        write_peft_checked(ckpt, arch, LAYERS, &meta, "tiny", dir)
    }

    #[test]
    fn export_writes_peft_keys_shapes_and_config_and_reads_back() {
        let dir = temp_dir("round_trip");
        let ckpt = checkpoint(false);
        let export = export(&ckpt, Architecture::Llama, &dir).expect("export");
        assert_eq!(export.rank, 2);
        assert_eq!(export.target_modules, ["gate_proj", "lm_head", "q_proj"]);
        assert_eq!(export.tensors, ckpt.len());

        let file = crate::checkpoint::load_lora_checkpoint(dir.join(PEFT_WEIGHTS_FILE)).unwrap();
        let shape = |k: &str| file.tensors[k].0.clone();
        assert_eq!(
            shape("base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"),
            [2, 3]
        );
        assert_eq!(
            shape("base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight"),
            [5, 2]
        );
        assert_eq!(shape("base_model.model.lm_head.lora_B.weight"), [7, 2]);
        // lora_A [3, 2] = [[a, b], [c, d], [e, f]] is stored as its transpose.
        let a = &ckpt.tensors["model.layers.0.self_attn.q_proj.lora_A.weight"].1;
        let peft_a =
            &file.tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].1;
        assert_eq!(peft_a, &vec![a[0], a[2], a[4], a[1], a[3], a[5]]);

        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join(PEFT_CONFIG_FILE)).unwrap())
                .unwrap();
        assert_eq!(config["peft_type"], "LORA");
        assert_eq!(config["task_type"], "CAUSAL_LM");
        assert_eq!(config["r"], 2);
        assert_eq!(config["lora_alpha"], 16);
        assert_eq!(config["bias"], "none");

        let back = load_peft_adapter(&dir).expect("import");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(back.tensors, ckpt.tensors);
        let meta = back.metadata.unwrap();
        assert_eq!((meta.rank, meta.alpha), (2, 16.0));
        assert_eq!(meta.target_modules, ["gate_proj", "lm_head", "q_proj"]);
    }

    #[test]
    fn old_names_export_like_new_names() {
        let (a, b) = (temp_dir("old_a"), temp_dir("old_b"));
        export(&checkpoint(true), Architecture::Llama, &a).expect("old names");
        export(&checkpoint(false), Architecture::Llama, &b).expect("new names");
        let read = |d: &Path| std::fs::read(d.join(PEFT_WEIGHTS_FILE)).unwrap();
        let same = read(&a) == read(&b);
        let _ = std::fs::remove_dir_all(&a);
        let _ = std::fs::remove_dir_all(&b);
        assert!(same);
    }

    #[test]
    fn what_peft_cannot_represent_is_an_error() {
        let dir = temp_dir("errors");
        let err = |ckpt: &LoraCheckpoint, arch| export(ckpt, arch, &dir).unwrap_err().to_string();

        let e = err(&checkpoint(false), Architecture::Lfm2);
        assert!(e.contains("Llama-style"), "{e}");

        let mut conv = checkpoint(false);
        conv.tensors.insert(
            "model.layers.0.conv.in_proj.lora_A.weight".into(),
            (vec![3, 2], vec![0.0; 6]),
        );
        assert!(err(&conv, Architecture::Llama).contains("conv.in_proj"));

        let mut one_layer = checkpoint(false);
        one_layer
            .tensors
            .retain(|k, _| !k.starts_with("model.layers.1.mlp"));
        assert!(err(&one_layer, Architecture::Llama).contains("1 of 2 layers"));

        let mut no_b = checkpoint(false);
        no_b.tensors.remove("lm_head.lora_B.weight");
        assert!(err(&no_b, Architecture::Llama).contains("no matching"));

        let mut both = checkpoint(false);
        both.tensors.insert(
            "model.layers.0.feed_forward.gate_proj.lora_A.weight".into(),
            (vec![3, 2], vec![0.0; 6]),
        );
        assert!(err(&both, Architecture::Llama).contains("old and its new name"));

        assert!(!dir.exists(), "nothing is written on error");
    }

    #[test]
    fn unsupported_peft_settings_are_rejected() {
        let dir = temp_dir("settings");
        export(&checkpoint(false), Architecture::Llama, &dir).expect("export");
        let path = dir.join(PEFT_CONFIG_FILE);
        let original: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for (key, value) in [
            ("use_dora", serde_json::json!(true)),
            ("bias", serde_json::json!("all")),
            ("fan_in_fan_out", serde_json::json!(true)),
            ("modules_to_save", serde_json::json!(["score"])),
            ("layers_to_transform", serde_json::json!([0])),
            ("rank_pattern", serde_json::json!({"q_proj": 4})),
            ("peft_type", serde_json::json!("IA3")),
        ] {
            let mut config = original.clone();
            config[key] = value;
            std::fs::write(&path, config.to_string()).unwrap();
            assert!(load_peft_adapter(&dir).is_err(), "{key} should be rejected");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

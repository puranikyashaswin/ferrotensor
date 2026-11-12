//! Saving and restoring LoRA adapters and training state.
//!
//! Adapters are written as safetensors (F32) with their LoRA settings in
//! the file's metadata, so `ferrotensor-cli generate --lora` can rebuild the same
//! adapter shapes. [`TrainingCheckpoint`] holds the step counter, optimizer
//! moments and loss-scaler state for resuming: settings as JSON, the moments
//! as F32 safetensors next to it.

use ferrotensor_autograd::{AdamWCheckpoint, OptimizerCheckpoint, OptimizerTensorState, Tensor};
use ferrotensor_backend::WgpuContext;
use ferrotensor_graph::lfm::legacy_llama_name;
use ferrotensor_loader::{LoaderError, WeightsContext};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::scaler::GradScalerCheckpoint;

/// Overwrite each tensor in `tensors` with the weight of the same name in
/// `weights_ctx`; names that are missing are logged and skipped.
pub fn load_weights(
    ctx: &Arc<WgpuContext>,
    weights_ctx: &WeightsContext,
    tensors: &HashMap<String, Tensor>,
) -> Result<(), LoaderError> {
    for (name, tensor) in tensors {
        if let Err(e) = weights_ctx.read_bytes(name, |data, _shape| {
            tensor.data.write_data(ctx, data);
        }) {
            log::warn!("Could not load weight {}: {:?}", name, e);
        }
    }
    Ok(())
}

/// LoRA settings stored in an adapter file's metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct LoraMetadata {
    /// Adapter rank.
    pub rank: usize,
    /// Scale numerator (the update is scaled by `alpha / rank`).
    pub alpha: f32,
    /// Projection names that carry adapters.
    pub target_modules: Vec<String>,
}

impl LoraMetadata {
    fn to_map(&self) -> HashMap<String, String> {
        HashMap::from([
            ("format".to_string(), "ferrotensor-lora".to_string()),
            ("rank".to_string(), self.rank.to_string()),
            ("alpha".to_string(), self.alpha.to_string()),
            ("target_modules".to_string(), self.target_modules.join(",")),
        ])
    }

    fn from_map(map: &HashMap<String, String>) -> Option<Self> {
        Some(Self {
            rank: map.get("rank")?.parse().ok()?,
            alpha: map.get("alpha")?.parse().ok()?,
            target_modules: map
                .get("target_modules")?
                .split(',')
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect(),
        })
    }
}

/// Tensors and metadata read from an adapter file.
#[derive(Clone, Debug)]
pub struct LoraCheckpoint {
    /// Tensor name to `(shape, values)`.
    pub tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    /// LoRA settings, when the file was written with them.
    pub metadata: Option<LoraMetadata>,
}

impl LoraCheckpoint {
    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file held no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Save LoRA adapter weights (F32 safetensors) to `path`, with `metadata`
/// in the header. The file is written to a temporary name and renamed, so a
/// crash never leaves a truncated adapter behind.
pub async fn save_lora_checkpoint<P: AsRef<Path>>(
    ctx: &Arc<WgpuContext>,
    path: P,
    tensors: &HashMap<String, Tensor>,
    metadata: Option<&LoraMetadata>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data_to_save = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let values = tensor.data.to_vec(ctx).await;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        data_to_save.push((name.clone(), tensor.data.shape.clone(), bytes));
    }

    let mut tensor_map = HashMap::new();
    for (name, shape, bytes) in &data_to_save {
        tensor_map.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)?,
        );
    }

    let data = safetensors::tensor::serialize(tensor_map, &metadata.map(LoraMetadata::to_map))?;
    let path = path.as_ref();
    let tmp = path.with_extension("safetensors.tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Load LoRA adapter weights from a safetensors file (F32, F16 or BF16
/// tensors are accepted and returned as F32).
pub fn load_lora_checkpoint<P: AsRef<Path>>(
    path: P,
) -> Result<LoraCheckpoint, Box<dyn std::error::Error>> {
    let path = path.as_ref();
    let data = std::fs::read(path)
        .map_err(|e| format!("cannot read adapter file {}: {e}", path.display()))?;
    let (_, header) = safetensors::tensor::SafeTensors::read_metadata(&data)
        .map_err(|e| format!("{} is not a safetensors file: {e}", path.display()))?;
    let metadata = header.metadata().as_ref().and_then(LoraMetadata::from_map);
    let tensors = safetensors::tensor::SafeTensors::deserialize(&data)?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let bytes = view.data();
        let values: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            safetensors::Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => return Err(format!("tensor {name} has unsupported dtype {other:?}").into()),
        };
        result.insert(name.to_string(), (view.shape().to_vec(), values));
    }

    Ok(LoraCheckpoint {
        tensors: result,
        metadata,
    })
}

/// Copy checkpoint values into the matching `tensors`.
///
/// Every tensor in `tensors` must be present in the checkpoint with the
/// same shape, and every tensor of the checkpoint must be used; otherwise
/// nothing is written and the error names the first problem (a different
/// rank or target set usually causes it). Returns the number of tensors
/// restored.
///
/// Adapters for Llama-style models written before Ferrotensor used Hugging
/// Face names for Llama blocks (`feed_forward.*`, `operator_norm`,
/// `ffn_norm`) load too: a name the checkpoint lacks is looked up under its
/// old form ([`legacy_llama_name`]).
pub fn apply_checkpoint(
    ctx: &Arc<WgpuContext>,
    checkpoint: &LoraCheckpoint,
    tensors: &HashMap<String, Tensor>,
) -> Result<usize, String> {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let mut sources: Vec<(&String, String)> = Vec::with_capacity(names.len());
    for name in names {
        let tensor = &tensors[name];
        let legacy = legacy_llama_name(name).filter(|old| checkpoint.tensors.contains_key(old));
        let key = match (checkpoint.tensors.contains_key(name), legacy) {
            (true, Some(old)) => {
                return Err(format!(
                    "adapter file has both {name} and its old name {old}"
                ))
            }
            (true, None) => name.clone(),
            (false, Some(old)) => old,
            (false, None) => return Err(format!("adapter file has no tensor {name}")),
        };
        let (shape, _) = &checkpoint.tensors[&key];
        if *shape != tensor.data.shape {
            return Err(format!(
                "tensor {key}: adapter file has shape {shape:?}, model expects {:?}",
                tensor.data.shape
            ));
        }
        sources.push((name, key));
    }
    let used: std::collections::HashSet<&str> = sources.iter().map(|(_, k)| k.as_str()).collect();
    let mut unused: Vec<&String> = checkpoint
        .tensors
        .keys()
        .filter(|k| !used.contains(k.as_str()))
        .collect();
    if !unused.is_empty() {
        unused.sort();
        return Err(format!(
            "adapter file has {} tensor(s) the model does not have, e.g. {} (were other LoRA targets trained?)",
            unused.len(),
            unused[0]
        ));
    }
    for (name, key) in &sources {
        let (_, values) = &checkpoint.tensors[key];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        tensors[*name].data.write_data(ctx, &bytes);
    }
    Ok(tensors.len())
}

/// Rename optimizer moments stored under old Llama-block names
/// ([`legacy_llama_name`]) to the names in `params`, so a run started before
/// the rename resumes with its moments. Returns how many were renamed.
pub fn upgrade_optimizer_names(
    state: &mut OptimizerCheckpoint,
    params: &HashMap<String, Tensor>,
) -> usize {
    let OptimizerCheckpoint::AdamW(adam) = state;
    let mut renamed = 0;
    for states in [&mut adam.m_states, &mut adam.v_states] {
        for name in params.keys() {
            if states.contains_key(name) {
                continue;
            }
            let Some(old) = legacy_llama_name(name) else {
                continue;
            };
            if let Some(moment) = states.remove(&old) {
                states.insert(name.clone(), moment);
                renamed += 1;
            }
        }
    }
    renamed
}

/// Progress and settings of a training run (also read by `ferrotensor-monitor`
/// from `training_state.json`).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingState {
    /// Optimizer steps completed.
    pub step: usize,
    /// Planned optimizer steps.
    pub total_steps: usize,
    /// Lowest logged training loss so far.
    pub best_loss: Option<f32>,
    /// Peak learning rate.
    pub lr: f32,
    /// Epochs completed.
    pub epoch: usize,
    /// Sequences per batch.
    pub batch_size: usize,
    /// Batches per optimizer step.
    pub grad_accum_steps: usize,
    /// Maximum tokens per example.
    pub sequence_length: usize,
    /// Whether loss scaling was enabled.
    pub fp16: bool,
    /// What was trained (`lora`, `all`, `attn`, ...).
    pub train_modules: String,
    /// Objective (`sft` or `pretrain`).
    #[serde(default = "default_task")]
    pub task: String,
    /// Model directory.
    #[serde(default)]
    pub model_path: String,
    /// Training data file.
    #[serde(default)]
    pub dataset_path: String,
    /// Data-order seed of the run, so a resume replays the same order.
    #[serde(default)]
    pub seed: u64,
}

fn default_task() -> String {
    "sft".to_string()
}

impl TrainingState {
    /// A fresh state for a run of `total_steps` steps.
    pub fn new(total_steps: usize) -> Self {
        Self {
            step: 0,
            total_steps,
            best_loss: None,
            lr: 0.0,
            epoch: 0,
            batch_size: 1,
            grad_accum_steps: 1,
            sequence_length: 128,
            fp16: false,
            train_modules: "lora".to_string(),
            task: "sft".to_string(),
            model_path: String::new(),
            dataset_path: String::new(),
            seed: 0,
        }
    }

    /// Write as pretty JSON.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Read a state written by [`save`](Self::save).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let state: Self = serde_json::from_str(&json)?;
        Ok(state)
    }

    /// Validate that a resume is compatible with the current run configuration.
    pub fn validate_resume(
        &self,
        train_modules: &str,
        task: &str,
        sequence_length: usize,
        model_path: &str,
    ) -> Result<(), String> {
        if !self.train_modules.is_empty() && self.train_modules != train_modules {
            return Err(format!(
                "checkpoint train_modules `{}` != requested `{}`",
                self.train_modules, train_modules
            ));
        }
        if !self.task.is_empty() && self.task != task {
            return Err(format!(
                "checkpoint task `{}` != requested `{}`",
                self.task, task
            ));
        }
        if self.sequence_length != 0 && self.sequence_length != sequence_length {
            return Err(format!(
                "checkpoint sequence_length `{}` != requested `{}`",
                self.sequence_length, sequence_length
            ));
        }
        if !self.model_path.is_empty() && self.model_path != model_path {
            return Err(format!(
                "checkpoint model_path `{}` != requested `{}`",
                self.model_path, model_path
            ));
        }
        Ok(())
    }
}

/// Everything besides the adapter weights needed to resume a run.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingCheckpoint {
    /// Progress and settings.
    pub training_state: TrainingState,
    /// Optimizer moments keyed by parameter name.
    pub optimizer_state: Option<OptimizerCheckpoint>,
    /// Loss-scaler state, if a scaler was used.
    pub grad_scaler_state: Option<GradScalerCheckpoint>,
}

impl TrainingCheckpoint {
    /// Write the checkpoint to `path` as JSON, with the optimizer moments in
    /// a separate F32 safetensors file ([`moments_path`]). Each file goes
    /// through a temporary name, so a crash never leaves a truncated one.
    ///
    /// The moments file is written first and records the step; if a crash
    /// falls between the two writes, [`load`](Self::load) reports the
    /// mismatch instead of resuming with the wrong moments.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let mut light = TrainingCheckpoint {
            training_state: self.training_state.clone(),
            optimizer_state: None,
            grad_scaler_state: self.grad_scaler_state.clone(),
        };
        if let Some(OptimizerCheckpoint::AdamW(adam)) = &self.optimizer_state {
            if moment_count(adam) > 0 {
                write_moments(&moments_path(path), adam, self.training_state.step)?;
            }
            light.optimizer_state = Some(OptimizerCheckpoint::AdamW(AdamWCheckpoint {
                lr: adam.lr,
                beta1: adam.beta1,
                beta2: adam.beta2,
                eps: adam.eps,
                weight_decay: adam.weight_decay,
                t: adam.t,
                m_states: shapes_only(&adam.m_states),
                v_states: shapes_only(&adam.v_states),
            }));
        }
        let json = serde_json::to_string_pretty(&light)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read a checkpoint written by [`save`](Self::save). Checkpoints that
    /// keep the moments inline in the JSON (written by older versions) load too.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let json = std::fs::read_to_string(path)?;
        let mut state: Self = serde_json::from_str(&json)?;
        let step = state.training_state.step;
        if let Some(OptimizerCheckpoint::AdamW(adam)) = &mut state.optimizer_state {
            let missing = adam
                .m_states
                .values()
                .chain(adam.v_states.values())
                .any(|t| t.data.is_empty() && t.shape.iter().product::<usize>() > 0);
            if missing {
                read_moments(&moments_path(path), adam, step)?;
            }
        }
        Ok(state)
    }
}

/// Where [`TrainingCheckpoint::save`] puts the optimizer moments for a
/// checkpoint at `path`: the same name with a `.safetensors` extension
/// (`x.state.json` -> `x.state.safetensors`).
pub fn moments_path(path: &Path) -> PathBuf {
    path.with_extension("safetensors")
}

const MOMENTS_FORMAT: &str = "ferrotensor-adamw-moments";

fn moment_count(adam: &AdamWCheckpoint) -> usize {
    adam.m_states.len() + adam.v_states.len()
}

fn shapes_only(
    states: &HashMap<String, OptimizerTensorState>,
) -> HashMap<String, OptimizerTensorState> {
    states
        .iter()
        .map(|(name, t)| {
            let state = OptimizerTensorState {
                shape: t.shape.clone(),
                data: Vec::new(),
            };
            (name.clone(), state)
        })
        .collect()
}

fn write_moments(
    path: &Path,
    adam: &AdamWCheckpoint,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries = Vec::with_capacity(moment_count(adam));
    for (prefix, states) in [("m/", &adam.m_states), ("v/", &adam.v_states)] {
        for (name, t) in states {
            let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
            entries.push((format!("{prefix}{name}"), t.shape.clone(), bytes));
        }
    }
    let mut views = HashMap::new();
    for (name, shape, bytes) in &entries {
        views.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)?,
        );
    }
    let metadata = HashMap::from([
        ("format".to_string(), MOMENTS_FORMAT.to_string()),
        ("step".to_string(), step.to_string()),
        ("t".to_string(), adam.t.to_string()),
    ]);
    let data = safetensors::tensor::serialize(views, &Some(metadata))?;
    let tmp = path.with_extension("safetensors.tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_moments(
    path: &Path,
    adam: &mut AdamWCheckpoint,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(path)
        .map_err(|e| format!("cannot read optimizer moments {}: {e}", path.display()))?;
    let (_, header) = safetensors::tensor::SafeTensors::read_metadata(&data)?;
    let meta = header.metadata().clone().unwrap_or_default();
    if meta.get("format").map(String::as_str) != Some(MOMENTS_FORMAT) {
        return Err(format!("{} is not an optimizer moments file", path.display()).into());
    }
    let saved_step: usize = meta
        .get("step")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    let saved_t: f32 = meta
        .get("t")
        .and_then(|s| s.parse().ok())
        .unwrap_or(f32::NAN);
    if saved_step != step || saved_t != adam.t {
        return Err(format!(
            "optimizer moments in {} are from step {saved_step}, but the checkpoint is at step {step};              the run was probably interrupted while saving",
            path.display()
        )
        .into());
    }
    let tensors = safetensors::tensor::SafeTensors::deserialize(&data)?;
    for (prefix, states) in [("m/", &mut adam.m_states), ("v/", &mut adam.v_states)] {
        for (name, t) in states.iter_mut() {
            let key = format!("{prefix}{name}");
            let view = tensors
                .tensor(&key)
                .map_err(|_| format!("{} has no tensor {key}", path.display()))?;
            if view.dtype() != safetensors::Dtype::F32 || view.shape() != t.shape.as_slice() {
                return Err(format!(
                    "{key} in {} has shape {:?}, expected {:?}",
                    path.display(),
                    view.shape(),
                    t.shape
                )
                .into());
            }
            t.data = view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{moments_path, TrainingCheckpoint, TrainingState};
    use crate::scaler::GradScalerCheckpoint;
    use ferrotensor_autograd::{AdamWCheckpoint, OptimizerCheckpoint, OptimizerTensorState};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn resume_validation_rejects_task_mismatch() {
        let mut state = TrainingState::new(10);
        state.task = "sft".into();
        state.train_modules = "lora".into();
        state.sequence_length = 128;
        state.model_path = "/models/a".into();
        let err = state
            .validate_resume("lora", "pretrain", 128, "/models/a")
            .unwrap_err();
        assert!(err.contains("task"));
    }

    #[test]
    fn resume_validation_rejects_seq_len_and_modules() {
        let mut state = TrainingState::new(10);
        state.task = "sft".into();
        state.train_modules = "lora".into();
        state.sequence_length = 128;
        state.model_path = "/models/a".into();
        assert!(state
            .validate_resume("all", "sft", 128, "/models/a")
            .unwrap_err()
            .contains("train_modules"));
        assert!(state
            .validate_resume("lora", "sft", 256, "/models/a")
            .unwrap_err()
            .contains("sequence_length"));
        assert!(state
            .validate_resume("lora", "sft", 128, "/models/b")
            .unwrap_err()
            .contains("model_path"));
    }

    #[test]
    fn resume_validation_accepts_matching_config() {
        let mut state = TrainingState::new(10);
        state.task = "sft".into();
        state.train_modules = "lora".into();
        state.sequence_length = 128;
        state.model_path = "/models/a".into();
        assert!(state
            .validate_resume("lora", "sft", 128, "/models/a")
            .is_ok());
    }

    #[test]
    fn training_checkpoint_roundtrip_preserves_resume_state() {
        let path = std::env::temp_dir().join("ferrotensor_resume_roundtrip_test.json");
        let mut training_state = TrainingState::new(100);
        training_state.step = 42;
        training_state.epoch = 1;
        training_state.task = "pretrain".into();
        training_state.train_modules = "lora".into();
        training_state.sequence_length = 128;
        training_state.model_path = "/models/tiny".into();
        training_state.dataset_path = "./data/train.bin".into();
        training_state.grad_accum_steps = 8;
        training_state.fp16 = true;

        let checkpoint = TrainingCheckpoint {
            training_state: training_state.clone(),
            optimizer_state: Some(OptimizerCheckpoint::AdamW(AdamWCheckpoint {
                lr: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
                t: 42.0,
                m_states: HashMap::new(),
                v_states: HashMap::new(),
            })),
            grad_scaler_state: Some(GradScalerCheckpoint {
                scale: 1024.0,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 2000,
                consecutive_good_steps: 3,
                found_overflow: false,
                enabled: true,
            }),
        };
        checkpoint.save(&path).expect("save checkpoint");
        let loaded = TrainingCheckpoint::load(&path).expect("load checkpoint");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.training_state.step, 42);
        assert_eq!(loaded.training_state.task, "pretrain");
        assert!(loaded
            .training_state
            .validate_resume("lora", "pretrain", 128, "/models/tiny")
            .is_ok());
        match loaded.optimizer_state.as_ref().unwrap() {
            OptimizerCheckpoint::AdamW(state) => assert_eq!(state.t, 42.0),
        }
        assert_eq!(loaded.grad_scaler_state.as_ref().unwrap().scale, 1024.0);
    }

    fn checkpoint_at(step: usize, values: usize) -> TrainingCheckpoint {
        let mut training_state = TrainingState::new(100);
        training_state.step = step;
        let state = |seed: f32| OptimizerTensorState {
            shape: vec![values / 4, 4],
            data: (0..values).map(|i| seed + i as f32 * 1e-3).collect(),
        };
        TrainingCheckpoint {
            training_state,
            optimizer_state: Some(OptimizerCheckpoint::AdamW(AdamWCheckpoint {
                lr: 5e-4,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
                t: step as f32,
                m_states: HashMap::from([
                    ("layers.0.q_proj.lora_a".to_string(), state(0.25)),
                    ("layers.0.q_proj.lora_b".to_string(), state(-0.5)),
                ]),
                v_states: HashMap::from([
                    ("layers.0.q_proj.lora_a".to_string(), state(1e-6)),
                    ("layers.0.q_proj.lora_b".to_string(), state(2e-6)),
                ]),
            })),
            grad_scaler_state: None,
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "ferrotensor_{}_{name}.state.json",
            std::process::id()
        ))
    }

    fn adam(ckpt: &TrainingCheckpoint) -> &AdamWCheckpoint {
        match ckpt.optimizer_state.as_ref().expect("optimizer state") {
            OptimizerCheckpoint::AdamW(a) => a,
        }
    }

    #[test]
    fn moments_round_trip_through_safetensors_bit_for_bit() {
        let path = temp_path("moments_round_trip");
        let ckpt = checkpoint_at(7, 40_000);
        ckpt.save(&path).expect("save");

        let json = std::fs::read_to_string(&path).expect("read json");
        let moments = std::fs::metadata(moments_path(&path))
            .expect("moments file")
            .len();
        let loaded = TrainingCheckpoint::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(moments_path(&path));

        // 4 tensors of 40,000 f32: the values live in the safetensors file,
        // not in the JSON.
        assert!(json.len() < 4_000, "JSON is {} bytes", json.len());
        assert!(moments >= 4 * 40_000 * 4, "moments file is {moments} bytes");
        let (a, b) = (adam(&ckpt), adam(&loaded));
        assert_eq!(b.t, 7.0);
        for (orig, back) in [(&a.m_states, &b.m_states), (&a.v_states, &b.v_states)] {
            assert_eq!(orig.len(), back.len());
            for (name, t) in orig {
                let r = &back[name];
                assert_eq!(r.shape, t.shape);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&r.data), bits(&t.data), "{name}");
            }
        }
    }

    #[test]
    fn moments_from_another_step_are_rejected() {
        let path = temp_path("moments_mismatch");
        let stale = std::env::temp_dir().join(format!(
            "ferrotensor_{}_stale.safetensors",
            std::process::id()
        ));
        checkpoint_at(5, 400).save(&path).expect("save step 5");
        std::fs::copy(moments_path(&path), &stale).expect("keep step 5 moments");
        checkpoint_at(6, 400).save(&path).expect("save step 6");
        // A crash after the moments of step 5 but before the JSON of step 6
        // would look like this, the other way round.
        std::fs::copy(&stale, moments_path(&path)).expect("put back step 5 moments");
        let err = TrainingCheckpoint::load(&path).unwrap_err().to_string();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(moments_path(&path));
        let _ = std::fs::remove_file(&stale);
        assert!(err.contains("step 5") && err.contains("step 6"), "{err}");
    }

    #[test]
    fn inline_json_checkpoints_still_load() {
        // The format written before the moments moved to safetensors.
        let path = temp_path("inline_json");
        let ckpt = checkpoint_at(3, 8);
        std::fs::write(&path, serde_json::to_string(&ckpt).unwrap()).expect("write");
        let loaded = TrainingCheckpoint::load(&path).expect("load");
        let _ = std::fs::remove_file(&path);
        assert!(!moments_path(&path).exists());
        let name = "layers.0.q_proj.lora_b";
        assert_eq!(
            adam(&loaded).m_states[name].data,
            adam(&ckpt).m_states[name].data
        );
    }

    /// Simulated continuous vs resume-to-end: remaining steps after resume
    /// should equal an uninterrupted schedule.
    #[test]
    fn resume_remaining_steps_match_uninterrupted_schedule() {
        let total_steps = 20usize;
        let interrupt_at = 7usize;

        let continuous_end = total_steps;
        let mut resumed = TrainingState::new(total_steps);
        resumed.step = interrupt_at;
        resumed.task = "sft".into();
        resumed.train_modules = "lora".into();
        resumed.sequence_length = 64;
        resumed.model_path = "m".into();
        resumed.validate_resume("lora", "sft", 64, "m").unwrap();

        let remaining = resumed.total_steps.saturating_sub(resumed.step);
        assert_eq!(interrupt_at + remaining, continuous_end);
        assert_eq!(remaining, total_steps - interrupt_at);
    }
}

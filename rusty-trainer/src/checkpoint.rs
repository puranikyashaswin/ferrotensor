//! Saving and restoring LoRA adapters and training state.
//!
//! Adapters are written as safetensors (F32) with their LoRA settings in
//! the file's metadata, so `rusty-cli generate --lora` can rebuild the same
//! adapter shapes. [`TrainingCheckpoint`] (JSON) holds the step counter,
//! optimizer moments and loss-scaler state for resuming.

use rusty_autograd::{OptimizerCheckpoint, Tensor};
use rusty_backend::WgpuContext;
use rusty_loader::{LoaderError, WeightsContext};
use std::collections::HashMap;
use std::path::Path;
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
            ("format".to_string(), "rusty-lora".to_string()),
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
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            safetensors::Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            safetensors::Dtype::BF16 => bytes
                .chunks_exact(2)
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
/// same shape; otherwise nothing is written and the error names the first
/// problem (a different rank or target set usually causes it). Returns the
/// number of tensors restored.
pub fn apply_checkpoint(
    ctx: &Arc<WgpuContext>,
    checkpoint: &LoraCheckpoint,
    tensors: &HashMap<String, Tensor>,
) -> Result<usize, String> {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    for name in &names {
        let tensor = &tensors[*name];
        match checkpoint.tensors.get(*name) {
            None => return Err(format!("adapter file has no tensor {name}")),
            Some((shape, _)) if *shape != tensor.data.shape => {
                return Err(format!(
                    "tensor {name}: adapter file has shape {shape:?}, model expects {:?}",
                    tensor.data.shape
                ))
            }
            Some(_) => {}
        }
    }
    for name in names {
        if let Some((_, values)) = checkpoint.tensors.get(name) {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            tensors[name].data.write_data(ctx, &bytes);
        }
    }
    Ok(tensors.len())
}

/// Progress and settings of a training run (also read by `rusty-monitor`
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
    /// Write as JSON (via a temporary file, so a crash never leaves a
    /// truncated checkpoint).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let json = serde_json::to_string_pretty(self)?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Read a checkpoint written by [`save`](Self::save).
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let state: Self = serde_json::from_str(&json)?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::{TrainingCheckpoint, TrainingState};
    use crate::scaler::GradScalerCheckpoint;
    use rusty_autograd::{AdamWCheckpoint, OptimizerCheckpoint};
    use std::collections::HashMap;

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
        let path = std::env::temp_dir().join("rusty_resume_roundtrip_test.json");
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

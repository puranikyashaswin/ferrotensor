use bytemuck;
use rusty_autograd::{OptimizerCheckpoint, Tensor};
use rusty_backend::WgpuContext;
use rusty_loader::{LoaderError, WeightsContext};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::scaler::GradScalerCheckpoint;

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

/// Save LoRA adapter weights to a .safetensors file
pub async fn save_lora_checkpoint<P: AsRef<Path>>(
    ctx: &Arc<WgpuContext>,
    path: P,
    tensors: &HashMap<String, Tensor>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut data_to_save = HashMap::new();

    for (name, tensor) in tensors {
        let f32_data = tensor.data.to_vec(ctx).await;
        let byte_data: Vec<u8> = bytemuck::cast_slice(&f32_data).to_vec();
        data_to_save.insert(name.clone(), (tensor.data.shape.clone(), byte_data));
    }

    // Use safetensors::serialize to create the buffer
    let mut tensor_map = HashMap::new();
    for (name, (shape, data)) in &data_to_save {
        tensor_map.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), data)?,
        );
    }

    let data = safetensors::tensor::serialize(tensor_map, &None)?;
    std::fs::write(path, data)?;

    Ok(())
}

/// Load LoRA adapter weights from a .safetensors file
/// Returns a HashMap of weight name to (shape, data) that can be used to update tensors
pub fn load_lora_checkpoint<P: AsRef<Path>>(
    path: P,
) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let tensors = safetensors::tensor::SafeTensors::deserialize(&data)?;

    let mut result = HashMap::new();
    for (name, tensor_view) in tensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let bytes = tensor_view.data();
        let f32_data: Vec<f32> = bytemuck::cast_slice(bytes).to_vec();
        result.insert(name.to_string(), (shape, f32_data));
    }

    Ok(result)
}

/// Apply loaded checkpoint data to existing tensors
pub fn apply_checkpoint(
    ctx: &Arc<WgpuContext>,
    checkpoint_data: &HashMap<String, (Vec<usize>, Vec<f32>)>,
    tensors: &HashMap<String, Tensor>,
) {
    for (name, tensor) in tensors {
        if let Some((shape, data)) = checkpoint_data.get(name) {
            if tensor.data.shape == *shape {
                let bytes: &[u8] = bytemuck::cast_slice(data);
                tensor.data.write_data(ctx, bytes);
            } else {
                log::warn!(
                    "Shape mismatch for {}: expected {:?}, got {:?}",
                    name,
                    tensor.data.shape,
                    shape
                );
            }
        }
    }
}

/// Training checkpoint containing step count and scheduler state
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingState {
    pub step: usize,
    pub total_steps: usize,
    pub best_loss: Option<f32>,
    pub lr: f32,
    pub epoch: usize,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub sequence_length: usize,
    pub fp16: bool,
    pub train_modules: String,
    #[serde(default = "default_task")]
    pub task: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub dataset_path: String,
}

fn default_task() -> String {
    "sft".to_string()
}

impl TrainingState {
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
        }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingCheckpoint {
    pub training_state: TrainingState,
    pub optimizer_state: Option<OptimizerCheckpoint>,
    pub grad_scaler_state: Option<GradScalerCheckpoint>,
}

impl TrainingCheckpoint {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

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

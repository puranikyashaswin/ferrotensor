//! Model configuration parsing

use crate::HubResult;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Model configuration from config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Hidden size / embedding dimension
    #[serde(default)]
    pub hidden_size: usize,

    /// Number of attention heads
    #[serde(default)]
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA)
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Number of transformer layers
    #[serde(default)]
    pub num_hidden_layers: usize,

    /// Intermediate size (MLP)
    #[serde(default)]
    pub intermediate_size: usize,

    /// Vocabulary size
    #[serde(default)]
    pub vocab_size: usize,

    /// Maximum context length
    #[serde(default)]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,

    /// RoPE theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Tie word embeddings
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Model type string
    #[serde(default)]
    pub model_type: String,
}

fn default_rms_eps() -> f32 {
    1e-5
}
fn default_rope_theta() -> f32 {
    10000.0
}

impl ModelConfig {
    /// Load config from a config.json file.
    pub fn from_file(path: &Path) -> HubResult<Self> {
        let content = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }

    /// Load config from a model directory.
    pub fn from_model_dir(model_dir: &Path) -> HubResult<Self> {
        Self::from_file(&model_dir.join("config.json"))
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get number of KV heads (for GQA).
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Check if using Grouped Query Attention.
    pub fn uses_gqa(&self) -> bool {
        self.kv_heads() != self.num_attention_heads
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec![],
            hidden_size: 4096,
            num_attention_heads: 32,
            num_key_value_heads: None,
            num_hidden_layers: 32,
            intermediate_size: 11008,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            model_type: "llama".to_string(),
        }
    }
}

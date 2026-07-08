//! Rusty Hub - HuggingFace Model Hub Integration
//!
//! Download and load pre-trained models from HuggingFace with a single line of code.
//!
//! # Example
//! ```rust,ignore
//! use rusty_hub::RustyHub;
//!
//! // Point to local model directory (downloaded from HuggingFace)
//! let model_path = hub.get_model_path("meta-llama/Llama-3.2-1B");
//! let config = ModelConfig::from_model_dir(&model_path)?;
//! ```

use std::fs;
use std::path::PathBuf;

pub mod config;
pub mod models;

pub use config::ModelConfig;
pub use models::ModelType;

/// Errors that can occur when using RustyHub
#[derive(Debug)]
pub enum HubError {
    Io(std::io::Error),
    Json(serde_json::Error),
    ModelNotFound(String),
    UnsupportedModel(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::ModelNotFound(m) => write!(f, "Model not found: {}", m),
            Self::UnsupportedModel(m) => write!(f, "Unsupported model: {}", m),
        }
    }
}

impl std::error::Error for HubError {}

impl From<std::io::Error> for HubError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for HubError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub type HubResult<T> = Result<T, HubError>;

/// Main entry point for loading models from HuggingFace Hub.
///
/// Currently supports loading from local directories.
/// Download models using `huggingface-cli download` or the Python `huggingface_hub` library.
///
/// # Example
/// ```rust,ignore
/// let hub = RustyHub::new();
/// let config = hub.load_config("meta-llama/Llama-3.2-1B")?;
/// println!("Model has {} layers", config.num_hidden_layers);
/// ```
pub struct RustyHub {
    cache_dir: PathBuf,
}

impl RustyHub {
    /// Create a new RustyHub instance with default cache directory.
    ///
    /// Default cache: `~/.cache/huggingface/hub/models--{org}--{model}`
    pub fn new() -> Self {
        let cache_dir = dirs_cache_dir().join("huggingface").join("hub");

        Self { cache_dir }
    }

    /// Create a RustyHub with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the path to a cached model.
    ///
    /// Compatible with HuggingFace cache format.
    pub fn get_model_path(&self, model_id: &str) -> PathBuf {
        // HuggingFace cache format: models--{org}--{model}
        let cache_name = format!("models--{}", model_id.replace("/", "--"));
        self.cache_dir.join(cache_name).join("snapshots")
    }

    /// Find the latest snapshot of a model.
    pub fn find_latest_snapshot(&self, model_id: &str) -> HubResult<PathBuf> {
        let snapshots_dir = self.get_model_path(model_id);

        if !snapshots_dir.exists() {
            return Err(HubError::ModelNotFound(format!(
                "Model {} not found. Download with: huggingface-cli download {}",
                model_id, model_id
            )));
        }

        // Find first snapshot directory
        let entries = fs::read_dir(&snapshots_dir)?;
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                return Ok(entry.path());
            }
        }

        Err(HubError::ModelNotFound(format!(
            "No snapshots found for {}",
            model_id
        )))
    }

    /// Load model configuration.
    pub fn load_config(&self, model_id: &str) -> HubResult<ModelConfig> {
        let snapshot = self.find_latest_snapshot(model_id)?;
        ModelConfig::from_file(&snapshot.join("config.json"))
    }

    /// Check if a model is cached locally.
    pub fn is_cached(&self, model_id: &str) -> bool {
        self.get_model_path(model_id).exists()
    }

    /// List all cached models.
    pub fn list_cached(&self) -> Vec<String> {
        if !self.cache_dir.exists() {
            return vec![];
        }

        fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .filter(|name| name.starts_with("models--"))
                    .map(|name| name.trim_start_matches("models--").replace("--", "/"))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get safetensors file paths for a model.
    pub fn get_weight_files(&self, model_id: &str) -> HubResult<Vec<PathBuf>> {
        let snapshot = self.find_latest_snapshot(model_id)?;

        let mut files = Vec::new();
        for entry in fs::read_dir(&snapshot)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "safetensors") {
                files.push(path);
            }
        }

        // Sort to ensure consistent order
        files.sort();
        Ok(files)
    }
}

impl Default for RustyHub {
    fn default() -> Self {
        Self::new()
    }
}

/// Get cache directory (fallback if dirs crate not available)
fn dirs_cache_dir() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".cache"))
        .unwrap_or_else(|_| PathBuf::from(".cache"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_creation() {
        let hub = RustyHub::new();
        assert!(hub.cache_dir.to_string_lossy().contains("huggingface"));
    }

    #[test]
    fn test_model_path() {
        let hub = RustyHub::new();
        let path = hub.get_model_path("meta-llama/Llama-3.2-1B");
        assert!(path
            .to_string_lossy()
            .contains("models--meta-llama--Llama-3.2-1B"));
    }
}

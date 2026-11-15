//! Ferrotensor Hub - HuggingFace Model Hub Integration
//!
//! Find and load models already downloaded to the Hugging Face cache.
//!
//! # Example
//! ```no_run
//! use ferrotensor_hub::{FerrotensorHub, ModelConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Resolve a model already downloaded to the Hugging Face cache
//! let hub = FerrotensorHub::new();
//! let model_path = hub.get_model_path("HuggingFaceTB/SmolLM2-135M");
//! let config = ModelConfig::from_model_dir(&model_path)?;
//! println!("{} layers", config.num_hidden_layers);
//! # Ok(())
//! # }
//! ```

use std::fs;
use std::path::{Path, PathBuf};

pub mod config;
pub mod models;

pub use config::ModelConfig;
pub use models::ModelType;

/// Errors that can occur when using FerrotensorHub
#[derive(Debug)]
pub enum HubError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// Failure parsing a JSON file.
    Json(serde_json::Error),
    /// The model directory or file was not found.
    ModelNotFound(String),
    /// The model type is not supported.
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

/// Result type for hub operations.
pub type HubResult<T> = Result<T, HubError>;

/// Main entry point for loading models from HuggingFace Hub.
///
/// Currently supports loading from local directories.
/// Download models with `hf download` or the Python `huggingface_hub` library.
///
/// # Example
/// ```no_run
/// # use ferrotensor_hub::FerrotensorHub;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let hub = FerrotensorHub::new();
/// let config = hub.load_config("HuggingFaceTB/SmolLM2-135M")?;
/// println!("Model has {} layers", config.num_hidden_layers);
/// # Ok(())
/// # }
/// ```
pub struct FerrotensorHub {
    cache_dir: PathBuf,
}

impl FerrotensorHub {
    /// Create a FerrotensorHub on the Hugging Face cache directory: `HF_HUB_CACHE`,
    /// else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
    /// `~/.cache/huggingface/hub` (`~` is `HOME`, or `USERPROFILE`). Empty
    /// variables count as unset.
    pub fn new() -> Self {
        Self {
            cache_dir: cache_dir_from(|name| std::env::var(name).ok()),
        }
    }

    /// Create a FerrotensorHub with a custom cache directory.
    pub fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// The directory that holds the `models--{org}--{model}` folders.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get the path to a cached model.
    ///
    /// Compatible with HuggingFace cache format.
    pub fn get_model_path(&self, model_id: &str) -> PathBuf {
        self.model_folder(model_id).join("snapshots")
    }

    /// `models--{org}--{model}` in the cache.
    fn model_folder(&self, model_id: &str) -> PathBuf {
        self.cache_dir
            .join(format!("models--{}", model_id.replace('/', "--")))
    }

    /// Find the snapshot of a model that Hugging Face's `main` branch points
    /// to: `snapshots/<hash>` for the hash in `refs/main`, when that directory
    /// exists. Otherwise the snapshot directory with the smallest name, so
    /// the result does not depend on directory order (a snapshot's age cannot
    /// be read from its name).
    pub fn find_latest_snapshot(&self, model_id: &str) -> HubResult<PathBuf> {
        let snapshots_dir = self.get_model_path(model_id);
        let not_found = || {
            HubError::ModelNotFound(format!(
                "Model {model_id} not found. Download with: hf download {model_id} \
                 (older installs: huggingface-cli download {model_id})"
            ))
        };

        if !snapshots_dir.exists() {
            return Err(not_found());
        }

        let refs_main = self.model_folder(model_id).join("refs").join("main");
        match fs::read_to_string(&refs_main) {
            Ok(hash) => {
                let hash = hash.trim();
                if !hash.is_empty() && snapshots_dir.join(hash).is_dir() {
                    return Ok(snapshots_dir.join(hash));
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }

        let mut snapshots = Vec::new();
        for entry in fs::read_dir(&snapshots_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                snapshots.push(entry.path());
            }
        }
        snapshots.into_iter().min().ok_or_else(not_found)
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
            if path.extension().is_some_and(|e| e == "safetensors") {
                files.push(path);
            }
        }

        // Sort to ensure consistent order
        files.sort();
        Ok(files)
    }
}

impl Default for FerrotensorHub {
    fn default() -> Self {
        Self::new()
    }
}

/// The Hugging Face hub cache directory, in the order Hugging Face documents;
/// `var` looks up an environment variable.
fn cache_dir_from(var: impl Fn(&str) -> Option<String>) -> PathBuf {
    let var = |name: &str| var(name).filter(|value| !value.is_empty());
    if let Some(hub) = var("HF_HUB_CACHE") {
        return PathBuf::from(hub);
    }
    if let Some(home) = var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let cache = match var("XDG_CACHE_HOME") {
        Some(xdg) => PathBuf::from(xdg),
        None => var("HOME")
            .or_else(|| var("USERPROFILE"))
            .map_or_else(PathBuf::new, PathBuf::from)
            .join(".cache"),
    };
    cache.join("huggingface").join("hub")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    /// The cache directory for the variables in `vars`.
    fn cache_for(vars: &[(&str, &str)]) -> PathBuf {
        cache_dir_from(|name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        })
    }

    #[test]
    fn model_path_follows_the_cache_layout() {
        let hub = FerrotensorHub::with_cache_dir(PathBuf::from("/cache"));
        assert_eq!(
            hub.get_model_path("meta-llama/Llama-3.2-1B"),
            PathBuf::from("/cache/models--meta-llama--Llama-3.2-1B/snapshots")
        );
    }

    #[test]
    fn hf_hub_cache_is_the_hub_directory_itself() {
        let all = [
            ("HF_HUB_CACHE", "/hub"),
            ("HF_HOME", "/hf"),
            ("XDG_CACHE_HOME", "/xdg"),
            ("HOME", "/home/u"),
        ];
        assert_eq!(cache_for(&all), PathBuf::from("/hub"));
    }

    #[test]
    fn hf_home_comes_before_xdg_and_home() {
        let vars = [
            ("HF_HOME", "/hf"),
            ("XDG_CACHE_HOME", "/xdg"),
            ("HOME", "/home/u"),
        ];
        assert_eq!(cache_for(&vars), PathBuf::from("/hf/hub"));
    }

    #[test]
    fn xdg_cache_home_comes_before_home() {
        let vars = [("XDG_CACHE_HOME", "/xdg"), ("HOME", "/home/u")];
        assert_eq!(cache_for(&vars), PathBuf::from("/xdg/huggingface/hub"));
    }

    #[test]
    fn home_and_userprofile_are_the_last_resort() {
        let home = [("HOME", "/home/u"), ("USERPROFILE", "/users/u")];
        assert_eq!(
            cache_for(&home),
            PathBuf::from("/home/u/.cache/huggingface/hub")
        );
        assert_eq!(
            cache_for(&[("USERPROFILE", "/users/u")]),
            PathBuf::from("/users/u/.cache/huggingface/hub")
        );
        assert_eq!(cache_for(&[]), PathBuf::from(".cache/huggingface/hub"));
    }

    #[test]
    fn empty_variables_count_as_unset() {
        let vars = [
            ("HF_HUB_CACHE", ""),
            ("HF_HOME", ""),
            ("XDG_CACHE_HOME", ""),
            ("HOME", "/home/u"),
        ];
        assert_eq!(
            cache_for(&vars),
            PathBuf::from("/home/u/.cache/huggingface/hub")
        );
    }

    /// A cache directory under the system temp dir, removed on drop.
    struct TempCache(PathBuf);

    impl TempCache {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("ferrotensor-hub-{tag}-{}", std::process::id()));
            fs::create_dir_all(&dir).expect("create temp cache");
            Self(dir)
        }

        /// Adds `snapshots/<hash>` for `org/name`.
        fn snapshot(&self, hash: &str) {
            fs::create_dir_all(self.snapshot_path(hash)).expect("create snapshot");
        }

        fn snapshot_path(&self, hash: &str) -> PathBuf {
            self.0.join("models--org--name/snapshots").join(hash)
        }

        fn refs_main(&self, contents: &str) {
            let refs = self.0.join("models--org--name/refs");
            fs::create_dir_all(&refs).expect("create refs");
            fs::write(refs.join("main"), contents).expect("write refs/main");
        }

        fn latest(&self) -> Option<PathBuf> {
            FerrotensorHub::with_cache_dir(self.0.clone())
                .find_latest_snapshot("org/name")
                .ok()
        }
    }

    impl Drop for TempCache {
        fn drop(&mut self) {
            if let Err(e) = fs::remove_dir_all(&self.0) {
                log::warn!("cannot remove {}: {e}", self.0.display());
            }
        }
    }

    #[test]
    fn refs_main_picks_the_snapshot() {
        let cache = TempCache::new("refs-main");
        cache.snapshot("aaa");
        cache.snapshot("bbb");
        cache.refs_main("bbb\n");
        assert_eq!(cache.latest(), Some(cache.snapshot_path("bbb")));
    }

    #[test]
    fn without_refs_main_the_smallest_snapshot_name_wins() {
        let cache = TempCache::new("no-refs");
        for hash in ["ccc", "aaa", "bbb"] {
            cache.snapshot(hash);
        }
        assert_eq!(cache.latest(), Some(cache.snapshot_path("aaa")));
    }

    #[test]
    fn a_refs_main_without_a_snapshot_falls_back() {
        let cache = TempCache::new("stale-refs");
        cache.snapshot("bbb");
        cache.snapshot("ccc");
        cache.refs_main("zzz");
        assert_eq!(cache.latest(), Some(cache.snapshot_path("bbb")));
        cache.refs_main("");
        assert_eq!(cache.latest(), Some(cache.snapshot_path("bbb")));
    }

    #[test]
    fn a_model_without_snapshots_is_not_found() {
        let cache = TempCache::new("empty");
        fs::create_dir_all(cache.0.join("models--org--name/snapshots")).expect("create dir");
        let hub = FerrotensorHub::with_cache_dir(cache.0.clone());
        let err = hub
            .find_latest_snapshot("org/name")
            .err()
            .map(|e| e.to_string());
        assert_eq!(
            err.as_deref(),
            Some(
                "Model not found: Model org/name not found. Download with: hf download org/name \
                 (older installs: huggingface-cli download org/name)"
            )
        );
    }
}

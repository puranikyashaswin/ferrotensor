//! Model loading for Ferrotensor: Hugging Face `config.json` parsing
//! ([`Lfm2Config`]), memory-mapped safetensors weights (single file or
//! sharded, F32/F16/BF16 read as F32) through [`WeightsContext`], and a
//! [`Tokenizer`] / [`Sampler`] pair for text in and out.
//! [`WeightsContext::upload_quantized`] uploads a `.weight` / `.scales` /
//! `.biases` triple as one quantized GPU tensor, but the model builder in
//! `ferrotensor-graph` rejects pre-quantized checkpoints.

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::de::Error as _;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Component, Path};
use std::sync::Arc;
use thiserror::Error;

mod release;
pub mod tokenizer;
pub use tokenizer::{argmax, Sampler, Tokenizer};

use ferrotensor_backend::UnifiedTensor;

/// Errors returned while loading model configs, weights and tensors.
#[derive(Error, Debug)]
pub enum LoaderError {
    /// Underlying I/O failure.
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    /// Failure parsing a safetensors file.
    #[error("Safetensors Error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// Failure parsing a JSON file.
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
    /// A requested tensor name is not present in any loaded file.
    #[error("Weight not found: {0}")]
    MissingWeight(String),
    /// The tensor uses a dtype this loader cannot convert to f32.
    #[error("Unsupported dtype for tensor {name}: {dtype}")]
    UnsupportedDtype {
        /// Tensor name.
        name: String,
        /// Dtype name as reported by safetensors.
        dtype: String,
    },
    /// The tensor byte length does not match its shape and dtype.
    #[error("Invalid tensor byte length for {name}: expected {expected} bytes, got {actual}")]
    InvalidTensorLength {
        /// Tensor name.
        name: String,
        /// Byte length implied by shape and dtype.
        expected: usize,
        /// Byte length found in the file.
        actual: usize,
    },
}

/// Quantization settings from the model config.
#[derive(Debug, Deserialize, Clone)]
pub struct QuantizationConfig {
    /// Number of weights that share one scale and bias.
    pub group_size: usize,
    /// Bits per quantized weight.
    pub bits: usize,
    /// Quantization scheme name, e.g. `"affine"`.
    pub mode: String,
}

/// Model hyperparameters read from `config.json` (Llama and LFM2 layouts).
#[derive(Debug, Deserialize, Clone)]
pub struct Lfm2Config {
    /// Width of the residual stream.
    pub hidden_size: usize,
    /// Width of the feed-forward (MLP) hidden layer.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of tokens in the vocabulary.
    pub vocab_size: usize,
    /// Epsilon added inside RMSNorm; defaults to 1e-5.
    #[serde(default = "default_norm_eps", alias = "rms_norm_eps")]
    pub norm_eps: f32,
    /// Hidden width used by LFM2 blocks, if set.
    #[serde(default)]
    pub block_dim: Option<usize>,
    /// Feed-forward width used by LFM2 blocks, if set.
    #[serde(default)]
    pub block_ff_dim: Option<usize>,
    /// Per-layer block type names (LFM2); empty when not given.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Number of key/value heads for grouped-query attention, if set.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    /// RoPE base frequency, if set.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Quantization settings, present for quantized checkpoints.
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
    /// Beginning-of-sequence token id, if set.
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    /// End-of-sequence token ids: `eos_token_id` may be one id or a list.
    #[serde(default, rename = "eos_token_id", deserialize_with = "one_or_more_ids")]
    pub eos_token_ids: Vec<u32>,
    /// Attention dropout probability, if set.
    #[serde(default)]
    pub attention_dropout: Option<f32>,
    /// Hugging Face `model_type`, e.g. `"llama"` or `"lfm2"`.
    #[serde(default)]
    pub model_type: Option<String>,
    /// Whether `lm_head` shares the token-embedding matrix.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
    /// Per-head dimension when it is not `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Longest sequence, in tokens, the model was trained on, if set.
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// MLP activation, e.g. `"silu"`.
    #[serde(default)]
    pub hidden_act: Option<String>,
    /// RoPE scaling (`transformers` 4 layout); `null` means none.
    #[serde(default)]
    pub rope_scaling: Option<RopeConfig>,
    /// RoPE type and base (`transformers` 5 layout).
    #[serde(default)]
    pub rope_parameters: Option<RopeConfig>,
    /// Attention window of sliding-window layers, if set.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Whether `sliding_window` applies (Qwen2 sets it to `false`).
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
}

/// A `rope_scaling` or `rope_parameters` object of `config.json`.
#[derive(Debug, Deserialize, Clone)]
pub struct RopeConfig {
    /// `"default"` for unscaled RoPE, else a scaling scheme such as `"llama3"`.
    #[serde(default)]
    pub rope_type: Option<String>,
    /// The older name of `rope_type`; some configs carry both.
    #[serde(default, rename = "type")]
    pub legacy_type: Option<String>,
    /// RoPE base frequency, if given here.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Scaling factor of a scaled RoPE, if any.
    #[serde(default)]
    pub factor: Option<f64>,
    /// Llama 3 scaling: the wavelength ratio below which a frequency is
    /// divided by `factor`.
    #[serde(default)]
    pub low_freq_factor: Option<f64>,
    /// Llama 3 scaling: the wavelength ratio above which a frequency is kept.
    #[serde(default)]
    pub high_freq_factor: Option<f64>,
    /// Llama 3 scaling: the context length the model was first trained on.
    #[serde(default)]
    pub original_max_position_embeddings: Option<f64>,
}

impl RopeConfig {
    /// `rope_type`, or else the older `type`.
    pub fn kind(&self) -> Option<&str> {
        self.rope_type.as_deref().or(self.legacy_type.as_deref())
    }
}

fn default_norm_eps() -> f32 {
    1e-5
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMoreIds {
    One(u32),
    More(Vec<u32>),
}

fn one_or_more_ids<'de, D: serde::Deserializer<'de>>(de: D) -> Result<Vec<u32>, D::Error> {
    Ok(match Option::<OneOrMoreIds>::deserialize(de)? {
        None => Vec::new(),
        Some(OneOrMoreIds::One(id)) => vec![id],
        Some(OneOrMoreIds::More(ids)) => ids,
    })
}

/// Largest `config.json` or shard index file the loader will read.
const MAX_JSON_FILE_BYTES: u64 = 64 * 1024 * 1024;
/// Upper bounds for config sizes; real models are far below all of them.
const MAX_HIDDEN_SIZE: usize = 1 << 20;
const MAX_INTERMEDIATE_SIZE: usize = 1 << 22;
const MAX_VOCAB_SIZE: usize = 1 << 24;
const MAX_ATTENTION_HEADS: usize = 1 << 16;
const MAX_HIDDEN_LAYERS: usize = 1 << 14;

/// Opens `path` for reading after checking that it is a regular file (so a
/// directory or a named pipe cannot hang or confuse the loader) and, when
/// `max_len` is set, not larger than that. Errors name the path.
pub(crate) fn open_regular_file(path: &Path, max_len: Option<u64>) -> Result<File, LoaderError> {
    let with_path = |e: std::io::Error| {
        LoaderError::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {}", path.display(), e),
        ))
    };
    let meta = std::fs::metadata(path).map_err(with_path)?;
    if !meta.is_file() {
        return Err(LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{}: not a regular file", path.display()),
        )));
    }
    if let Some(max) = max_len {
        if meta.len() > max {
            return Err(LoaderError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "{}: file is {} bytes, more than the {} byte limit",
                    path.display(),
                    meta.len(),
                    max
                ),
            )));
        }
    }
    File::open(path).map_err(with_path)
}

/// A JSON error whose message starts with the file it came from.
fn json_error(path: &Path, detail: impl std::fmt::Display) -> LoaderError {
    LoaderError::Json(serde_json::Error::custom(format!(
        "{}: {}",
        path.display(),
        detail
    )))
}

/// Reads and parses a JSON file; errors name the file.
fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LoaderError> {
    let file = open_regular_file(path, Some(MAX_JSON_FILE_BYTES))?;
    serde_json::from_reader(BufReader::new(file)).map_err(|e| json_error(path, e))
}

fn check_range(
    path: &Path,
    field: &str,
    value: usize,
    min: usize,
    max: usize,
) -> Result<(), LoaderError> {
    if value < min || value > max {
        return Err(json_error(
            path,
            format!("invalid {field}: {value} (expected {min} to {max})"),
        ));
    }
    Ok(())
}

/// Whether `divisor` divides `n` exactly (false for a zero divisor).
fn divides(divisor: usize, n: usize) -> bool {
    n.checked_rem(divisor) == Some(0)
}

/// Rejects configs whose sizes are zero, absurdly large, or inconsistent.
fn validate_config(path: &Path, c: &Lfm2Config) -> Result<(), LoaderError> {
    check_range(path, "hidden_size", c.hidden_size, 1, MAX_HIDDEN_SIZE)?;
    check_range(
        path,
        "intermediate_size",
        c.intermediate_size,
        1,
        MAX_INTERMEDIATE_SIZE,
    )?;
    check_range(
        path,
        "num_attention_heads",
        c.num_attention_heads,
        1,
        MAX_ATTENTION_HEADS,
    )?;
    check_range(
        path,
        "num_hidden_layers",
        c.num_hidden_layers,
        1,
        MAX_HIDDEN_LAYERS,
    )?;
    check_range(path, "vocab_size", c.vocab_size, 1, MAX_VOCAB_SIZE)?;
    for (field, value, max) in [
        ("head_dim", c.head_dim, MAX_HIDDEN_SIZE),
        ("block_dim", c.block_dim, MAX_HIDDEN_SIZE),
        ("block_ff_dim", c.block_ff_dim, MAX_INTERMEDIATE_SIZE),
    ] {
        if let Some(v) = value {
            check_range(path, field, v, 1, max)?;
        }
    }
    if c.head_dim.is_none() && !divides(c.num_attention_heads, c.hidden_size) {
        return Err(json_error(
            path,
            format!(
                "hidden_size ({}) is not divisible by num_attention_heads ({}) and head_dim is not set",
                c.hidden_size, c.num_attention_heads
            ),
        ));
    }
    if let Some(kv) = c.num_key_value_heads {
        check_range(path, "num_key_value_heads", kv, 1, c.num_attention_heads)?;
        if !divides(kv, c.num_attention_heads) {
            return Err(json_error(
                path,
                format!(
                    "num_attention_heads ({}) is not a multiple of num_key_value_heads ({kv})",
                    c.num_attention_heads
                ),
            ));
        }
    }
    if let Some(q) = &c.quantization {
        check_range(
            path,
            "quantization.group_size",
            q.group_size,
            1,
            MAX_HIDDEN_SIZE,
        )?;
        check_range(path, "quantization.bits", q.bits, 1, 32)?;
    }
    Ok(())
}

// Backwards compat alias
/// Alias kept for older code that used the Llama-specific config name.
pub type LlamaConfig = Lfm2Config;

/// Reads model config files and raw safetensors files.
pub struct ModelLoader {
    // Map weight name -> (mmap_index, tensor_view) logic
    // For MVP, simplified to single file or manual lookup
}

impl ModelLoader {
    /// Reads and parses a `config.json` file, rejecting zero, absurdly large
    /// or inconsistent sizes. Error messages name the file and the field.
    pub fn load_config<P: AsRef<Path>>(path: P) -> Result<LlamaConfig, LoaderError> {
        let path = path.as_ref();
        let config: LlamaConfig = read_json(path)?;
        validate_config(path, &config)?;
        Ok(config)
    }

    // Helper to load a single safetensors file
    /// Memory-maps a single safetensors file.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<memmap2::Mmap, LoaderError> {
        let path = path.as_ref();
        let file = open_regular_file(path, None)?;
        // SAFETY: the file is opened read-only and the mapping is only read.
        // The one hazard is another process truncating the file while it is
        // mapped; `open_regular_file` has already ruled out directories and pipes.
        #[expect(unsafe_code, reason = "memory-mapping a file is unsafe in memmap2")]
        let mmap = unsafe { MmapOptions::new().map(&file) }.map_err(|e| {
            LoaderError::Io(std::io::Error::new(
                e.kind(),
                format!("{}: {}", path.display(), e),
            ))
        })?;
        Ok(mmap)
    }
}

// A wrapper to hold the loaded SafeTensors views
// Since SafeTensors borrows from Mmap, we need a struct that holds the Mmap owner
// and provides access. This is tricky with self-referential structs in Rust.
//
// Solution: "WeightsContext" that holds the Mmaps, and we pass `&WeightsContext`
// to a function that extracts views on demand.

/// Reinterprets little-endian bytes as `f32` values; the length must be a
/// multiple of 4.
fn le_bytes_to_f32(name: &str, bytes: &[u8]) -> Result<Vec<f32>, LoaderError> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(LoaderError::InvalidTensorLength {
            name: name.to_string(),
            expected: bytes.len().next_multiple_of(4),
            actual: bytes.len(),
        });
    }
    Ok(chunks.iter().map(|c| f32::from_le_bytes(*c)).collect())
}

/// Memory-mapped safetensors files and an index from tensor name to file.
pub struct WeightsContext {
    mmaps: HashMap<String, memmap2::Mmap>, // filename -> mmap
    // We can also parse the index.json if it exists to know which file has which weight
    weight_map: HashMap<String, String>, // weight_name -> filename
}

#[derive(Debug, Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

/// Shape and dtype of a stored tensor.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Stored element type.
    pub dtype: safetensors::Dtype,
}

impl Default for WeightsContext {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightsContext {
    /// Creates an empty context with no files loaded.
    pub fn new() -> Self {
        Self {
            mmaps: HashMap::new(),
            weight_map: HashMap::new(),
        }
    }

    /// Maps one safetensors file and indexes its tensor names.
    pub fn load_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), LoaderError> {
        let path = path.as_ref();
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .ok_or_else(|| {
                LoaderError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("{} is not a file path", path.display()),
                ))
            })?;
        let mmap = ModelLoader::load_safetensors(path)?;

        // Pre-scan to populate weight_map
        let tensors = SafeTensors::deserialize(&mmap)?;
        for name in tensors.names() {
            self.weight_map.insert(name.to_string(), filename.clone());
        }

        self.mmaps.insert(filename, mmap);
        Ok(())
    }

    /// Loads a model directory: sharded files via the index, else `model.safetensors`, else `consolidated.safetensors`.
    pub fn load_model_dir<P: AsRef<Path>>(&mut self, model_dir: P) -> Result<(), LoaderError> {
        let model_dir = model_dir.as_ref();
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index: SafetensorIndex = read_json(&index_path)?;
            let mut shard_files: Vec<String> = index.weight_map.values().cloned().collect();
            shard_files.sort();
            shard_files.dedup();
            if shard_files.is_empty() {
                return Err(json_error(&index_path, "weight_map lists no weights"));
            }

            // Shard names come from a file the user downloaded: they must stay
            // inside the model directory.
            for shard in &shard_files {
                let relative = Path::new(shard);
                let inside = relative.components().next().is_some()
                    && relative
                        .components()
                        .all(|c| matches!(c, Component::Normal(_)));
                if !inside {
                    return Err(json_error(
                        &index_path,
                        format!(
                            "shard path {shard:?} is not a relative path inside the model directory"
                        ),
                    ));
                }
            }

            for shard in shard_files {
                self.load_file(model_dir.join(&shard))?;
            }
            return Ok(());
        }

        let primary = model_dir.join("model.safetensors");
        if primary.exists() {
            return self.load_file(primary);
        }

        let consolidated = model_dir.join("consolidated.safetensors");
        if consolidated.exists() {
            return self.load_file(consolidated);
        }

        Err(LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no model.safetensors, consolidated.safetensors, or model.safetensors.index.json found",
        )))
    }

    /// Names of every tensor in the loaded files, in no particular order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }

    /// Returns the shape and dtype of a tensor, or `None` when no loaded file
    /// holds it; use it to probe for optional tensors.
    ///
    /// # Errors
    ///
    /// If the file that holds the tensor no longer parses.
    pub fn get_tensor_info(&self, name: &str) -> Result<Option<TensorMetadata>, LoaderError> {
        let Some(filename) = self.weight_map.get(name) else {
            return Ok(None);
        };
        let mmap = self
            .mmaps
            .get(filename)
            .ok_or_else(|| LoaderError::MissingWeight(name.to_string()))?;
        let st = SafeTensors::deserialize(mmap)?;
        let view = st.tensor(name)?;
        Ok(Some(TensorMetadata {
            shape: view.shape().to_vec(),
            dtype: view.dtype(),
        }))
    }

    /// Returns the shape and dtype of a tensor.
    ///
    /// # Errors
    ///
    /// [`LoaderError::MissingWeight`] when no loaded file holds it, or the
    /// parse error of the file that does.
    pub fn get_tensor_metadata(&self, name: &str) -> Result<TensorMetadata, LoaderError> {
        self.get_tensor_info(name)?
            .ok_or_else(|| LoaderError::MissingWeight(name.to_string()))
    }

    // We can't return TensorView because it borrows from local mmap variable inside function
    // We must pass a closure or return raw data copy.
    // For our usage: We want to upload to GPU immediately.
    /// Calls `callback` with the raw bytes and shape of a tensor.
    ///
    /// Afterwards the memory pages that hold only this tensor's bytes are
    /// released from the process (on Unix), so a model load does not keep
    /// the whole file resident; reading the tensor again faults them back
    /// in from the file.
    ///
    /// # Errors
    ///
    /// If no loaded file holds the tensor, the file no longer parses, or the
    /// kernel refuses to release the pages.
    pub fn read_bytes<F>(&self, name: &str, callback: F) -> Result<(), LoaderError>
    where
        F: FnOnce(&[u8], &[usize]),
    {
        if let Some(filename) = self.weight_map.get(name) {
            let mmap = self
                .mmaps
                .get(filename)
                .ok_or_else(|| LoaderError::MissingWeight(name.to_string()))?;
            let st = SafeTensors::deserialize(mmap)?;
            let view = st.tensor(name)?;
            callback(view.data(), view.shape());
            release::release(mmap, view.data()).map_err(|e| {
                LoaderError::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot release the pages of {name} ({filename}): {e}"),
                ))
            })
        } else {
            Err(LoaderError::MissingWeight(name.to_string()))
        }
    }

    /// Reads an F32, F16 or BF16 tensor as `f32` values plus its shape.
    pub fn read_f32_tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), LoaderError> {
        let meta = self.get_tensor_metadata(name)?;
        // Check the dtype before allocating: a large tensor of an unsupported
        // dtype must not reserve memory only to be rejected.
        if !matches!(
            meta.dtype,
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        ) {
            return Err(LoaderError::UnsupportedDtype {
                name: name.to_string(),
                dtype: format!("{:?}", meta.dtype),
            });
        }
        let numel: usize = meta.shape.iter().product();
        let mut out = Vec::with_capacity(numel);
        let mut actual_len = 0usize;

        self.read_bytes(name, |bytes, shape| {
            debug_assert_eq!(shape, meta.shape.as_slice());
            actual_len = bytes.len();
            match meta.dtype {
                safetensors::Dtype::F32 => {
                    if bytes.len() != numel * 4 {
                        out = Vec::new();
                        return;
                    }
                    out.extend(
                        bytes.as_chunks::<4>().0.iter().map(|chunk| {
                            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        }),
                    );
                }
                safetensors::Dtype::F16 => {
                    if bytes.len() != numel * 2 {
                        out = Vec::new();
                        return;
                    }
                    out.extend(bytes.as_chunks::<2>().0.iter().map(|chunk| {
                        half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                    }));
                }
                safetensors::Dtype::BF16 => {
                    if bytes.len() != numel * 2 {
                        out = Vec::new();
                        return;
                    }
                    out.extend(bytes.as_chunks::<2>().0.iter().map(|chunk| {
                        half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32()
                    }));
                }
                _ => out = vec![],
            }
        })?;

        let expected_len = match meta.dtype {
            safetensors::Dtype::F32 => numel * 4,
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => numel * 2,
            _ => {
                return Err(LoaderError::UnsupportedDtype {
                    name: name.to_string(),
                    dtype: format!("{:?}", meta.dtype),
                })
            }
        };

        if out.len() != numel {
            let actual = match meta.dtype {
                safetensors::Dtype::F32 => actual_len,
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => actual_len,
                _ => 0,
            };
            return Err(LoaderError::InvalidTensorLength {
                name: name.to_string(),
                expected: expected_len,
                actual,
            });
        }

        Ok((out, meta.shape))
    }

    /// Reads `<name>.weight`, `.scales` and `.biases` and uploads them as one quantized GPU tensor.
    pub fn upload_quantized(
        &self,
        name: &str,
        ctx: &Arc<ferrotensor_backend::WgpuContext>,
        engine: &ferrotensor_backend::ComputeEngine,
    ) -> Result<UnifiedTensor, LoaderError> {
        let weight_name = format!("{}.weight", name);
        let scales_name = format!("{}.scales", name);
        let biases_name = format!("{}.biases", name);

        if !self.weight_map.contains_key(&weight_name) {
            return Err(LoaderError::MissingWeight(weight_name));
        }

        // 1. Get Host Data
        let mut q_data: Vec<u8> = Vec::new();
        let mut q_shape: Vec<usize> = Vec::new();
        self.read_bytes(&weight_name, |data, shape| {
            q_data = data.to_vec();
            q_shape = shape.to_vec();
        })?;

        let mut s_data: Vec<u8> = Vec::new();
        self.read_bytes(&scales_name, |data, _| s_data = data.to_vec())?;

        let mut b_data: Vec<u8> = Vec::new();
        self.read_bytes(&biases_name, |data, _| b_data = data.to_vec())?;

        // 2. Upload to GPU buffers using UnifiedTensor (or raw create using backend's re-exported wgpu)
        // For simplicity, we'll convert to f32 slices where appropriate
        // Here we assume scales and biases are f32 already in safetensors
        let s_floats = le_bytes_to_f32(&scales_name, &s_data)?;
        let b_floats = le_bytes_to_f32(&biases_name, &b_data)?;

        let s_tensor = UnifiedTensor::new(ctx, &s_floats, &[s_floats.len()]);
        let b_tensor = UnifiedTensor::new(ctx, &b_floats, &[b_floats.len()]);

        // For quantized weights, we need a u8 buffer. The dequant kernel reads u32 (4 u8s packed).
        // For now, let's just wrap it in a storage buffer.
        let q_buf = ctx
            .device
            .create_buffer(&ferrotensor_backend::wgpu::BufferDescriptor {
                label: Some("Q Weight"),
                size: q_data.len() as u64,
                usage: ferrotensor_backend::wgpu::BufferUsages::STORAGE
                    | ferrotensor_backend::wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        ctx.queue.write_buffer(&q_buf, 0, &q_data);
        ctx.note_upload(q_data.len() as u64);
        // Dropped once the dequantize dispatch below is recorded.
        ctx.memory().count_dispatch_buffer(q_buf.size());

        // 3. Output Tensor (FP32)
        let out_tensor = UnifiedTensor::empty(ctx, &q_shape);

        // 4. Dispatch Dequant
        engine.affine_dequantize(
            ctx,
            &q_buf,
            s_tensor.buffer(),
            b_tensor.buffer(),
            &out_tensor,
        );

        Ok(out_tensor)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn le_bytes_to_f32_reads_whole_values() {
        let bytes: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(le_bytes_to_f32("s", &bytes).unwrap(), vec![1.5, -2.0]);
        assert!(le_bytes_to_f32("s", &[]).unwrap().is_empty());
    }

    /// A safetensors file holding one F32 tensor `a` of shape `[1]`.
    fn one_tensor_file() -> Vec<u8> {
        let header = br#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&1.5f32.to_le_bytes());
        bytes
    }

    #[test]
    fn a_probe_tells_an_absent_tensor_from_a_corrupt_file() {
        let dir = std::env::temp_dir().join(format!("ferrotensor-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("model.safetensors");
        let corrupt = dir.join("corrupt.safetensors");
        let full = one_tensor_file();
        std::fs::write(&good, &full).unwrap();
        // The header length still says the whole header, but the file ends
        // halfway through it.
        std::fs::write(&corrupt, &full[..20]).unwrap();

        let mut weights = WeightsContext::new();
        weights.load_file(&good).unwrap();
        let a = weights.get_tensor_info("a").unwrap().unwrap();
        assert_eq!((a.shape, a.dtype), (vec![1], safetensors::Dtype::F32));
        assert!(weights.get_tensor_info("a.bias").unwrap().is_none());
        assert!(matches!(
            weights.get_tensor_metadata("a.bias"),
            Err(LoaderError::MissingWeight(name)) if name == "a.bias"
        ));

        let load = WeightsContext::new().load_file(&corrupt);
        assert!(matches!(load, Err(LoaderError::Safetensors(_))), "{load:?}");
        // The indexed file is replaced by the corrupt one, as when a
        // checkpoint changes on disk after it was loaded.
        let corrupt_map = ModelLoader::load_safetensors(&corrupt).unwrap();
        weights
            .mmaps
            .insert("model.safetensors".to_string(), corrupt_map);
        let probe = weights.get_tensor_info("a");
        assert!(
            matches!(probe, Err(LoaderError::Safetensors(_))),
            "{probe:?}"
        );
        assert!(matches!(
            weights.get_tensor_metadata("a"),
            Err(LoaderError::Safetensors(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn eos_token_id_reads_one_id_a_list_or_nothing() {
        let sizes = r#""hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2,
            "num_hidden_layers": 1, "vocab_size": 32"#;
        for (eos, want) in [
            (r#", "eos_token_id": 2"#, vec![2]),
            (r#", "eos_token_id": [2, 7]"#, vec![2, 7]),
            (r#", "eos_token_id": null"#, vec![]),
            ("", vec![]),
        ] {
            let config: Lfm2Config = serde_json::from_str(&format!("{{{sizes}{eos}}}")).unwrap();
            assert_eq!(config.eos_token_ids, want, "{eos}");
        }
        let bad = format!(r#"{{{sizes}, "eos_token_id": "</s>"}}"#);
        assert!(serde_json::from_str::<Lfm2Config>(&bad).is_err());
    }

    #[test]
    fn le_bytes_to_f32_rejects_a_partial_value() {
        let err = le_bytes_to_f32("layer.scales", &[0u8; 6]).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidTensorLength { .. }));
        assert!(err.to_string().contains("layer.scales"), "{err}");
    }
}

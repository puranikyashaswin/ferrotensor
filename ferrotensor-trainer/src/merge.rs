//! Folding a LoRA adapter into the base weights.
//!
//! [`merge_adapter`] writes a model directory whose weights are those of the
//! base with every adapter added in: for a PyTorch `Linear.weight` `W`
//! (`[out, in]`) and an adapter with `lora_A` `[in, rank]` and `lora_B`
//! `[rank, out]` (the Ferrotensor layout, see [`peft`](crate::peft)), the
//! merged weight is `W + alpha / rank * (A B)^T`. The result loads as an
//! ordinary checkpoint, with no adapter.
//!
//! The merge streams: each base tensor is read from the memory-mapped
//! checkpoint, changed one row at a time if it carries an adapter, written
//! and released before the next, so the process holds about one row plus
//! the adapter, not the model.

use crate::checkpoint::LoraCheckpoint;
use crate::peft::transpose;
use ferrotensor_graph::lfm::llama_name_from_legacy;
use ferrotensor_loader::WeightsContext;
use safetensors::tensor::TensorInfo;
use safetensors::Dtype;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Name of the weights file [`merge_adapter`] writes.
pub const MERGED_WEIGHTS_FILE: &str = "model.safetensors";

/// Files copied from the base model directory when they exist.
const COPIED_FILES: [&str; 11] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "chat_template.json",
];

/// What [`merge_adapter`] wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeReport {
    /// The merged weights file.
    pub weights: PathBuf,
    /// Tensors with an adapter added.
    pub merged: usize,
    /// Tensors copied byte for byte.
    pub copied: usize,
}

/// The element types an adapter can be added to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Float {
    F32,
    F16,
    Bf16,
}

impl Float {
    fn of(dtype: Dtype) -> Option<Self> {
        match dtype {
            Dtype::F32 => Some(Self::F32),
            Dtype::F16 => Some(Self::F16),
            Dtype::BF16 => Some(Self::Bf16),
            _ => None,
        }
    }

    fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    fn decode(self, bytes: &[u8], values: &mut [f32]) {
        match self {
            Self::F32 => {
                for (v, c) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *v = f32::from_le_bytes(*c);
                }
            }
            Self::F16 => {
                for (v, c) in values.iter_mut().zip(bytes.as_chunks::<2>().0) {
                    *v = half::f16::from_le_bytes(*c).to_f32();
                }
            }
            Self::Bf16 => {
                for (v, c) in values.iter_mut().zip(bytes.as_chunks::<2>().0) {
                    *v = half::bf16::from_le_bytes(*c).to_f32();
                }
            }
        }
    }

    fn encode(self, values: &[f32], bytes: &mut Vec<u8>) {
        match self {
            Self::F32 => bytes.extend(values.iter().flat_map(|v| v.to_le_bytes())),
            Self::F16 => {
                bytes.extend(
                    values
                        .iter()
                        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes()),
                );
            }
            Self::Bf16 => {
                bytes.extend(
                    values
                        .iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes()),
                );
            }
        }
    }
}

/// One adapter, checked against the base weight it is added to.
struct LowRank<'a> {
    /// `lora_A` transposed to `[rank, in]`, so a row of the delta reads it
    /// contiguously.
    a_t: Vec<f32>,
    /// `lora_B`, `[rank, out]`.
    b: &'a [f32],
    in_dim: usize,
    out_dim: usize,
    scale: f32,
    /// Element type of the base weight, kept in the output.
    float: Float,
}

impl LowRank<'_> {
    /// Adds row `o` of `scale * (A B)^T` to `row` (`[in]`).
    fn add_to_row(&self, o: usize, row: &mut [f32]) {
        for (r, a_row) in self.a_t.chunks_exact(self.in_dim).enumerate() {
            let coeff = self.scale * self.b[r * self.out_dim + o];
            for (w, a) in row.iter_mut().zip(a_row) {
                *w += coeff * a;
            }
        }
    }

    /// Writes the base weight `bytes` (`[out, in]`) with the adapter added,
    /// one row at a time, computed in f32 and stored in the base's type.
    fn write_merged(&self, bytes: &[u8], out: &mut impl Write) -> std::io::Result<()> {
        let mut row = vec![0.0; self.in_dim];
        let mut encoded = Vec::with_capacity(self.in_dim * self.float.width());
        for (o, base_row) in bytes
            .chunks_exact(self.in_dim * self.float.width())
            .enumerate()
        {
            self.float.decode(base_row, &mut row);
            self.add_to_row(o, &mut row);
            encoded.clear();
            self.float.encode(&row, &mut encoded);
            out.write_all(&encoded)?;
        }
        Ok(())
    }
}

/// A tensor of the output file and where its bytes come from.
struct Entry<'a> {
    name: String,
    /// Base tensor read for it (the embeddings for an untied `lm_head`).
    source: String,
    dtype: Dtype,
    shape: Vec<usize>,
    low_rank: Option<LowRank<'a>>,
}

/// The adapters of `adapter` keyed by the output tensor they change, each
/// with the tensor of `weights` it is added to.
fn adapters_by_target<'a>(
    adapter: &'a LoraCheckpoint,
    weights: &WeightsContext,
) -> Result<HashMap<String, (String, LowRank<'a>)>> {
    let meta = adapter.metadata.as_ref().ok_or(
        "the adapter has no LoRA metadata (rank/alpha); it was not written by Ferrotensor",
    )?;
    if let Some(modules) = &meta.trained_modules {
        return Err(format!(
            "the file holds base weights trained with --train-modules {modules}, not a LoRA adapter"
        )
        .into());
    }
    if meta.rank == 0 {
        return Err("the adapter metadata says rank 0".into());
    }
    let scale = meta.alpha / meta.rank as f32;

    let mut names: Vec<&String> = adapter.tensors.keys().collect();
    names.sort();
    let mut out = HashMap::new();
    for name in names {
        if name.ends_with(".lora_B.weight") {
            let a_name = name.replace(".lora_B.weight", ".lora_A.weight");
            if !adapter.tensors.contains_key(&a_name) {
                return Err(format!("{name} has no matching {a_name}").into());
            }
            continue;
        }
        let module = name.strip_suffix(".lora_A.weight").ok_or_else(|| {
            format!("tensor {name} is not a LoRA tensor (.lora_A.weight or .lora_B.weight)")
        })?;
        let b_name = format!("{module}.lora_B.weight");
        let (b_shape, b) = adapter
            .tensors
            .get(&b_name)
            .ok_or_else(|| format!("{name} has no matching {b_name}"))?;
        let (a_shape, a) = &adapter.tensors[name];
        let (&[in_dim, rank], &[rank_b, out_dim]) = (a_shape.as_slice(), b_shape.as_slice()) else {
            return Err(format!(
                "{name} has shape {a_shape:?} and {b_name} {b_shape:?}; both must be 2-D"
            )
            .into());
        };
        if a.len() != in_dim * rank || b.len() != rank_b * out_dim {
            return Err(
                format!("{name} or {b_name} has fewer or more values than its shape").into(),
            );
        }
        if rank != meta.rank || rank_b != meta.rank {
            return Err(format!(
                "{name} has rank {rank} and {b_name} rank {rank_b}; the adapter metadata says {}",
                meta.rank
            )
            .into());
        }

        // Adapters saved before Llama blocks used Hugging Face names carry
        // the old names; a tied lm_head reads the embeddings.
        let target = [Some(module.to_string()), llama_name_from_legacy(module)]
            .into_iter()
            .flatten()
            .map(|m| format!("{m}.weight"))
            .find(|t| weights.get_tensor_info(t).is_some())
            .unwrap_or_else(|| format!("{module}.weight"));
        let source = if target == "lm_head.weight" && weights.get_tensor_info(&target).is_none() {
            "model.embed_tokens.weight".to_string()
        } else {
            target.clone()
        };
        if weights.get_tensor_info(&source).is_none() {
            return Err(format!("{name}: the base model has no tensor {target}").into());
        }
        let base = weights.get_tensor_metadata(&source)?;
        if base.shape != [out_dim, in_dim] {
            return Err(format!(
                "{name}: the adapter changes a [{out_dim}, {in_dim}] weight, but {source} is {:?}",
                base.shape
            )
            .into());
        }
        let float = Float::of(base.dtype).ok_or_else(|| {
            format!(
                "{source} is {:?}; only F32, F16 and BF16 weights can take an adapter",
                base.dtype
            )
        })?;
        let low_rank = LowRank {
            a_t: transpose(a, in_dim, rank),
            b,
            in_dim,
            out_dim,
            scale,
            float,
        };
        if out.insert(target.clone(), (source, low_rank)).is_some() {
            return Err(format!(
                "the adapter has two adapters for {target} (under its old and its new name)"
            )
            .into());
        }
    }
    if out.is_empty() {
        return Err("the adapter has no LoRA tensors".into());
    }
    Ok(out)
}

/// Write the base model in `model_dir` with `adapter` added to its weights
/// into `out_dir`, as `model.safetensors` (one file, also for a sharded
/// base) plus the base's `config.json`, generation config and tokenizer
/// files.
///
/// Each weight keeps its dtype; the sum is computed in f32. Tensors without
/// an adapter are copied byte for byte. An adapter on `lm_head` of a model
/// with tied embeddings writes a separate `lm_head.weight` and sets
/// `tie_word_embeddings` to false in the copied `config.json`, since the
/// embeddings themselves stay unchanged.
///
/// # Errors
///
/// Fails, before writing any weights, when `out_dir` exists and is not
/// empty (so it is never the model directory), when the base is quantized
/// (`quantization` or `quantization_config` in `config.json`, or
/// `.scales` tensors), when the adapter has no LoRA metadata or holds
/// trained base weights, or when an adapter tensor has no matching base
/// weight of the right shape and a float dtype; the message names the
/// tensor. I/O errors while writing are returned as they occur.
pub fn merge_adapter(
    model_dir: &Path,
    adapter: &LoraCheckpoint,
    out_dir: &Path,
) -> Result<MergeReport> {
    let not_empty = out_dir.exists()
        && std::fs::read_dir(out_dir)
            .map_err(|e| format!("cannot read {}: {e}", out_dir.display()))?
            .next()
            .is_some();
    if not_empty {
        return Err(format!(
            "{} is not empty; merge writes into a new or empty directory",
            out_dir.display()
        )
        .into());
    }
    let config_path = model_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("cannot read {}: {e}", config_path.display()))?;
    let mut config: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|e| format!("{}: {e}", config_path.display()))?;
    for key in ["quantization", "quantization_config"] {
        if config.get(key).is_some() {
            return Err(format!(
                "{} has {key}: the base is quantized; merge needs an F32, F16 or BF16 checkpoint",
                config_path.display()
            )
            .into());
        }
    }

    let mut weights = WeightsContext::new();
    weights.load_model_dir(model_dir)?;
    let mut base_names: Vec<&str> = weights.tensor_names().collect();
    base_names.sort_unstable();
    if let Some(name) = base_names.iter().find(|n| n.ends_with(".scales")) {
        return Err(format!(
            "the base is pre-quantized ({name} found); merge needs an F32, F16 or BF16 checkpoint"
        )
        .into());
    }
    let mut adapters = adapters_by_target(adapter, &weights)?;

    let mut entries = Vec::with_capacity(base_names.len() + 1);
    for name in base_names {
        let meta = weights.get_tensor_metadata(name)?;
        entries.push(Entry {
            name: name.to_string(),
            source: name.to_string(),
            dtype: meta.dtype,
            shape: meta.shape,
            low_rank: adapters.remove(name).map(|(_, low_rank)| low_rank),
        });
    }
    // What is left is an lm_head tied to the embeddings.
    let untie = !adapters.is_empty();
    for (name, (source, low_rank)) in adapters {
        let meta = weights.get_tensor_metadata(&source)?;
        entries.push(Entry {
            name,
            source,
            dtype: meta.dtype,
            shape: meta.shape,
            low_rank: Some(low_rank),
        });
    }
    // Widest dtype first, as the safetensors crate orders tensors, so every
    // tensor starts at a multiple of its element size.
    entries.sort_by(|x, y| y.dtype.cmp(&x.dtype).then_with(|| x.name.cmp(&y.name)));

    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_string(),
        serde_json::json!({ "format": "pt" }),
    );
    let mut offset = 0;
    for e in &entries {
        let len = e.shape.iter().product::<usize>() * e.dtype.size();
        let info = TensorInfo {
            dtype: e.dtype,
            shape: e.shape.clone(),
            data_offsets: (offset, offset + len),
        };
        header.insert(e.name.clone(), serde_json::to_value(info)?);
        offset += len;
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');

    // I/O errors keep their type, so callers can tell them from bad input.
    std::fs::create_dir_all(out_dir).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("cannot create {}: {e}", out_dir.display()),
        )
    })?;
    let path = out_dir.join(MERGED_WEIGHTS_FILE);
    let tmp = path.with_extension("safetensors.tmp");
    let mut file = BufWriter::new(File::create(&tmp)?);
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    for e in &entries {
        let mut written = Ok(());
        weights.read_bytes(&e.source, |bytes, _| {
            written = match &e.low_rank {
                Some(low_rank) => low_rank.write_merged(bytes, &mut file),
                None => file.write_all(bytes),
            };
        })?;
        written.map_err(|err| {
            std::io::Error::new(err.kind(), format!("cannot write {}: {err}", tmp.display()))
        })?;
    }
    file.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, &path)?;

    for name in COPIED_FILES {
        let from = model_dir.join(name);
        if name == "config.json" && untie {
            config["tie_word_embeddings"] = serde_json::Value::Bool(false);
            std::fs::write(out_dir.join(name), serde_json::to_string_pretty(&config)?)?;
        } else if from.is_file() {
            std::fs::copy(&from, out_dir.join(name))?;
        }
    }

    let merged = entries.iter().filter(|e| e.low_rank.is_some()).count();
    Ok(MergeReport {
        weights: path,
        merged,
        copied: entries.len() - merged,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// `W` is `[2, 3]`, `A` `[3, 2]`, `B` `[2, 2]`, scale 0.5. By hand,
    /// `A B` is `[[1, 2], [3, 4], [4, 6]]`, so
    /// `W' = W + 0.5 * (A B)^T = [[1.5, 3.5, 5], [5, 7, 9]]`.
    fn merged(float: Float) -> Vec<f32> {
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let low_rank = LowRank {
            a_t: transpose(&a, 3, 2),
            b: &b,
            in_dim: 3,
            out_dim: 2,
            scale: 0.5,
            float,
        };
        let mut base = Vec::new();
        float.encode(&w, &mut base);
        let mut out = Vec::new();
        low_rank.write_merged(&base, &mut out).unwrap();
        assert_eq!(out.len(), base.len(), "the dtype is kept");
        let mut values = vec![0.0; 6];
        float.decode(&out, &mut values);
        values
    }

    #[test]
    fn adds_the_scaled_transposed_product_in_f32_f16_and_bf16() {
        let expected = vec![1.5, 3.5, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(merged(Float::F32), expected);
        assert_eq!(merged(Float::F16), expected);
        assert_eq!(merged(Float::Bf16), expected);
    }

    #[test]
    fn f16_rounds_the_f32_sum_to_nearest() {
        // 1 + 2^-11 lies halfway between the f16 neighbours 1 and 1 + 2^-10;
        // round to nearest even gives 1. 1 + 0.75 * 2^-10 rounds up.
        let w = [1.0f32];
        let mut base = Vec::new();
        Float::F16.encode(&w, &mut base);
        for (delta, expected) in [
            (2f32.powi(-11), 1.0),
            (0.75 * 2f32.powi(-10), 1.0 + 2f32.powi(-10)),
        ] {
            let b = [delta];
            let low_rank = LowRank {
                a_t: vec![1.0],
                b: &b,
                in_dim: 1,
                out_dim: 1,
                scale: 1.0,
                float: Float::F16,
            };
            let mut out = Vec::new();
            low_rank.write_merged(&base, &mut out).unwrap();
            let mut value = [0.0];
            Float::F16.decode(&out, &mut value);
            assert_eq!(value[0], expected, "delta {delta}");
        }
    }
}

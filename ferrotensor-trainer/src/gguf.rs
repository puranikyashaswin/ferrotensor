//! Writing a Llama-style model as a GGUF file for llama.cpp and Ollama.
//!
//! [`export_gguf`] writes GGUF version 3 (ggml `docs/gguf.md`) with the
//! metadata keys, tensor names and conventions of llama.cpp's
//! `convert_hf_to_gguf.py`: dimensions are listed innermost first, and the
//! rows of the query and key projections are reordered from the Hugging Face
//! rotary layout (the two halves of each head) to llama.cpp's (adjacent
//! pairs). Like [`merge`](crate::merge) the export streams: one row of one
//! tensor is in memory at a time, plus the adapter and the tokenizer.

use crate::checkpoint::LoraCheckpoint;
use crate::merge::{adapters_by_target, write_through_temp_file, Float, LowRank};
use ferrotensor_graph::model_builder::check_supported;
use ferrotensor_loader::{Lfm2Config, ModelLoader, WeightsContext};
use serde_json::Value as Json;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Byte alignment of the tensor data, also written as `general.alignment`.
const ALIGNMENT: usize = 32;

/// `tokenizer.ggml.token_type` values (llama.cpp's `TokenType`).
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;

/// Element type of the 2-D weights in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufPrecision {
    /// f16, with 1-D tensors and norms in f32 (`general.file_type` 1).
    F16,
    /// f32 for every tensor (`general.file_type` 0).
    F32,
}

/// What [`export_gguf`] wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufReport {
    /// Tensors in the file.
    pub tensors: usize,
    /// Tensors with an adapter added.
    pub merged: usize,
}

/// A metadata value, of the GGUF types this writer uses.
#[derive(Clone, Debug, PartialEq)]
enum MetaValue {
    U32(u32),
    F32(f32),
    Bool(bool),
    Str(String),
    Strs(Vec<String>),
    I32s(Vec<i32>),
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    out.extend((text.len() as u64).to_le_bytes());
    out.extend(text.as_bytes());
}

impl MetaValue {
    /// Appends the GGUF value type, then the value.
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Self::U32(v) => {
                out.extend(4u32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            Self::F32(v) => {
                out.extend(6u32.to_le_bytes());
                out.extend(v.to_le_bytes());
            }
            Self::Bool(v) => {
                out.extend(7u32.to_le_bytes());
                out.push(u8::from(*v));
            }
            Self::Str(text) => {
                out.extend(8u32.to_le_bytes());
                put_str(out, text);
            }
            Self::Strs(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(8u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                for item in items {
                    put_str(out, item);
                }
            }
            Self::I32s(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(5u32.to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                for item in items {
                    out.extend(item.to_le_bytes());
                }
            }
        }
    }
}

/// A tensor of the output file and where its rows come from.
struct GgufTensor<'a> {
    name: String,
    /// Base tensor read for it (the embeddings for an `lm_head` tied to them).
    source: String,
    /// Shape in PyTorch order, outermost first.
    shape: Vec<usize>,
    stored: Float,
    target: Float,
    /// Head count of a query or key projection, whose rows are reordered.
    rope_heads: Option<usize>,
    low_rank: Option<LowRank<'a>>,
}

impl GgufTensor<'_> {
    fn cols(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    fn byte_len(&self) -> usize {
        self.shape.iter().product::<usize>() * self.target.width()
    }
}

/// llama.cpp's name for the Hugging Face Llama tensor `hf_name`.
fn gguf_tensor_name(hf_name: &str) -> Option<String> {
    let fixed = match hf_name {
        "model.embed_tokens.weight" => Some("token_embd.weight"),
        "model.norm.weight" => Some("output_norm.weight"),
        "lm_head.weight" => Some("output.weight"),
        _ => None,
    };
    if let Some(name) = fixed {
        return Some(name.to_string());
    }
    let (block, module) = hf_name.strip_prefix("model.layers.")?.split_once('.')?;
    if block.is_empty() || !block.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let short = match module {
        "input_layernorm.weight" => "attn_norm",
        "self_attn.q_proj.weight" => "attn_q",
        "self_attn.k_proj.weight" => "attn_k",
        "self_attn.v_proj.weight" => "attn_v",
        "self_attn.o_proj.weight" => "attn_output",
        "post_attention_layernorm.weight" => "ffn_norm",
        "mlp.gate_proj.weight" => "ffn_gate",
        "mlp.up_proj.weight" => "ffn_up",
        "mlp.down_proj.weight" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{block}.{short}.weight"))
}

/// The output tensor for the Hugging Face tensor `hf_name`, read from
/// `source` with `low_rank` added.
fn describe_tensor<'a>(
    weights: &WeightsContext,
    hf_name: &str,
    source: String,
    low_rank: Option<LowRank<'a>>,
    config: &Lfm2Config,
    precision: GgufPrecision,
) -> Result<GgufTensor<'a>> {
    if hf_name.ends_with(".bias") {
        return Err(format!(
            "tensor {hf_name} is a projection bias; export-gguf writes Llama weights without \
             biases, so Qwen2-style models (q/k/v biases) are not exported"
        )
        .into());
    }
    let name = gguf_tensor_name(hf_name).ok_or_else(|| {
        format!("tensor {hf_name} has no llama.cpp name; export-gguf writes Llama weights only")
    })?;
    let meta = weights.get_tensor_metadata(&source)?;
    let stored = Float::of(meta.dtype).ok_or_else(|| {
        format!(
            "{source} is {:?}: the base is quantized; export-gguf needs an F32, F16 or BF16 checkpoint",
            meta.dtype
        )
    })?;
    // As llama.cpp's converter: 1-D tensors and norms stay f32.
    let target = if precision == GgufPrecision::F32
        || meta.shape.len() < 2
        || name.ends_with("_norm.weight")
    {
        Float::F32
    } else {
        Float::F16
    };
    let heads = config.num_attention_heads;
    let rope_heads = if hf_name.ends_with(".self_attn.q_proj.weight") {
        Some(heads)
    } else if hf_name.ends_with(".self_attn.k_proj.weight") {
        Some(config.num_key_value_heads.unwrap_or(heads))
    } else {
        None
    };
    if let Some(heads) = rope_heads {
        let rows = meta.shape.first().copied().unwrap_or(0);
        if meta.shape.len() != 2 || rows % (2 * heads) != 0 {
            return Err(format!(
                "{hf_name} has shape {:?}; its rows must split into {heads} heads of even size",
                meta.shape
            )
            .into());
        }
    }
    Ok(GgufTensor {
        name,
        source,
        shape: meta.shape,
        stored,
        target,
        rope_heads,
        low_rank,
    })
}

fn read_json(path: &Path) -> Result<Json> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?)
}

/// llama.cpp's name for a byte-level pre-tokenizer it implements: `gpt-2`
/// for GPT-2's split, `smollm` for single digits split off first.
fn pre_tokenizer_name(pre: &Json) -> Option<&'static str> {
    let byte_level = |p: &Json| {
        p["type"] == "ByteLevel" && p["use_regex"] != false && p["add_prefix_space"] != true
    };
    if byte_level(pre) {
        return Some("gpt-2");
    }
    match pre["pretokenizers"].as_array()?.as_slice() {
        [digits, split]
            if pre["type"] == "Sequence"
                && digits["type"] == "Digits"
                && digits["individual_digits"] == true
                && byte_level(split) =>
        {
            Some("smollm")
        }
        _ => None,
    }
}

/// The `tokenizer.*` metadata of the byte-level BPE tokenizer in
/// `model_dir`, padded with unused tokens to the `n_vocab` embedding rows.
fn tokenizer_metadata(
    model_dir: &Path,
    config: &Lfm2Config,
    n_vocab: usize,
) -> Result<Vec<(String, MetaValue)>> {
    let tokenizer = read_json(&model_dir.join("tokenizer.json"))?;
    let config_path = model_dir.join("tokenizer_config.json");
    let tokenizer_config = if config_path.is_file() {
        read_json(&config_path)?
    } else {
        Json::Null
    };
    let model = &tokenizer["model"];
    if model["type"] != "BPE" {
        return Err(format!(
            "tokenizer.json has a {} model; export-gguf needs a byte-level BPE tokenizer",
            model["type"]
        )
        .into());
    }
    let pre = pre_tokenizer_name(&tokenizer["pre_tokenizer"]).ok_or_else(|| {
        format!(
            "tokenizer.json has the pre-tokenizer {}; export-gguf knows the byte-level ones of \
             GPT-2 and SmolLM (llama.cpp `gpt-2` and `smollm`)",
            tokenizer["pre_tokenizer"]
        )
    })?;

    let mut slots: Vec<Option<(String, i32)>> = vec![None; n_vocab];
    let mut place = |token: &str, id: &Json, token_type: i32| -> Result<()> {
        let slot = id
            .as_u64()
            .and_then(|id| slots.get_mut(usize::try_from(id).ok()?))
            .ok_or_else(|| {
                format!(
                    "tokenizer.json gives {token:?} the id {id}, outside the {n_vocab} rows of \
                     the embeddings"
                )
            })?;
        *slot = Some((token.to_string(), token_type));
        Ok(())
    };
    let vocab = model["vocab"]
        .as_object()
        .ok_or("tokenizer.json has no model.vocab")?;
    for (token, id) in vocab {
        place(token, id, NORMAL)?;
    }
    for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
        let token = added["content"]
            .as_str()
            .ok_or("tokenizer.json has an added token without content")?;
        let control =
            added["special"] == true || (token.starts_with("<|") && token.ends_with("|>"));
        place(
            token,
            &added["id"],
            if control { CONTROL } else { USER_DEFINED },
        )?;
    }
    let (tokens, token_types): (Vec<String>, Vec<i32>) = slots
        .into_iter()
        .enumerate()
        .map(|(id, slot)| slot.unwrap_or_else(|| (format!("[PAD{id}]"), UNUSED)))
        .unzip();

    let merges = model["merges"]
        .as_array()
        .ok_or("tokenizer.json has no model.merges")?
        .iter()
        .map(|merge| match merge {
            Json::String(pair) => Some(pair.clone()),
            Json::Array(pair) => match pair.as_slice() {
                [Json::String(left), Json::String(right)] => Some(format!("{left} {right}")),
                _ => None,
            },
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("tokenizer.json has a merge that is neither \"a b\" nor [\"a\", \"b\"]")?;

    let ids: HashMap<&str, u32> = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token.as_str(), id as u32))
        .collect();
    let mut special_ids = Vec::new();
    for (kind, key, from_config) in [
        ("bos", "bos_token_id", config.bos_token_id),
        ("eos", "eos_token_id", config.eos_token_ids.first().copied()),
        ("unk", "unknown_token_id", None),
        ("pad", "padding_token_id", None),
    ] {
        let named = &tokenizer_config[format!("{kind}_token")];
        let id = match named.as_str().or_else(|| named["content"].as_str()) {
            Some(content) => Some(*ids.get(content).ok_or_else(|| {
                format!("tokenizer_config.json names {kind}_token {content:?}, which tokenizer.json does not have")
            })?),
            None => from_config,
        };
        if let Some(id) = id {
            special_ids.push((format!("tokenizer.ggml.{key}"), MetaValue::U32(id)));
        }
    }
    let mut metadata = vec![
        (
            "tokenizer.ggml.model".to_string(),
            MetaValue::Str("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            MetaValue::Str(pre.to_string()),
        ),
        ("tokenizer.ggml.tokens".to_string(), MetaValue::Strs(tokens)),
        (
            "tokenizer.ggml.token_type".to_string(),
            MetaValue::I32s(token_types),
        ),
        ("tokenizer.ggml.merges".to_string(), MetaValue::Strs(merges)),
    ];
    metadata.extend(special_ids);
    for key in ["add_bos_token", "add_eos_token"] {
        if let Some(add) = tokenizer_config[key].as_bool() {
            metadata.push((format!("tokenizer.ggml.{key}"), MetaValue::Bool(add)));
        }
    }
    let jinja = model_dir.join("chat_template.jinja");
    let template = match tokenizer_config["chat_template"].as_str() {
        Some(template) => Some(template.to_string()),
        None if jinja.is_file() => Some(
            std::fs::read_to_string(&jinja)
                .map_err(|e| format!("cannot read {}: {e}", jinja.display()))?,
        ),
        None => None,
    };
    if let Some(template) = template {
        metadata.push((
            "tokenizer.chat_template".to_string(),
            MetaValue::Str(template),
        ));
    }
    Ok(metadata)
}

/// Everything before the tensor data: magic, version, counts, metadata and
/// tensor infos, padded with zeros to the alignment.
fn gguf_header(metadata: &[(String, MetaValue)], tensors: &[GgufTensor<'_>]) -> Vec<u8> {
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((tensors.len() as u64).to_le_bytes());
    header.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        put_str(&mut header, key);
        value.write(&mut header);
    }
    let mut offset = 0;
    for tensor in tensors {
        put_str(&mut header, &tensor.name);
        header.extend((tensor.shape.len() as u32).to_le_bytes());
        for dim in tensor.shape.iter().rev() {
            header.extend((*dim as u64).to_le_bytes());
        }
        let ggml_type: u32 = match tensor.target {
            Float::F32 => 0,
            Float::F16 => 1,
            Float::Bf16 => 30,
        };
        header.extend(ggml_type.to_le_bytes());
        header.extend((offset as u64).to_le_bytes());
        offset += tensor.byte_len().next_multiple_of(ALIGNMENT);
    }
    header.resize(header.len().next_multiple_of(ALIGNMENT), 0);
    header
}

/// Writes the rows of `tensor` from its source `bytes`, with the adapter
/// added, in the target type, then zeros up to the alignment.
fn write_tensor(
    tensor: &GgufTensor<'_>,
    bytes: &[u8],
    out: &mut impl Write,
) -> std::io::Result<()> {
    let cols = tensor.cols();
    let rows = tensor.shape.iter().product::<usize>() / cols;
    let row_bytes = cols * tensor.stored.width();
    let mut row = vec![0.0; cols];
    let mut encoded = Vec::with_capacity(cols * tensor.target.width());
    for r in 0..rows {
        let source_row = match tensor.rope_heads {
            // Row 2k + j of a head is row k + j * head_dim / 2 in Hugging Face.
            Some(heads) => {
                let head_dim = rows / heads;
                let (head, i) = (r / head_dim, r % head_dim);
                head * head_dim + (i % 2) * (head_dim / 2) + i / 2
            }
            None => r,
        };
        let start = source_row * row_bytes;
        tensor
            .stored
            .decode(&bytes[start..start + row_bytes], &mut row);
        if let Some(low_rank) = &tensor.low_rank {
            low_rank.add_to_row(source_row, &mut row);
        }
        encoded.clear();
        tensor.target.encode(&row, &mut encoded);
        out.write_all(&encoded)?;
    }
    let len = tensor.byte_len();
    out.write_all(&[0; ALIGNMENT][..len.next_multiple_of(ALIGNMENT) - len])
}

/// Write the Llama-style model in `model_dir`, with `adapter` added to its
/// weights when given, to the new GGUF file `out`, named `name`.
///
/// # Errors
///
/// Fails, before writing, when `out` exists, when `config.json` is not
/// `model_type` `llama` or has no `max_position_embeddings`, when the base
/// is quantized (`quantization` in `config.json`, `.scales` tensors or a
/// tensor that is not F32, F16 or BF16), when a tensor has no llama.cpp
/// name, when the tokenizer is not byte-level BPE with a pre-tokenizer
/// llama.cpp knows, or when the adapter does not match the base. I/O errors
/// while writing are returned as they occur, as `std::io::Error`.
pub fn export_gguf(
    model_dir: &Path,
    name: &str,
    adapter: Option<&LoraCheckpoint>,
    out: &Path,
    precision: GgufPrecision,
) -> Result<GgufReport> {
    if out.exists() {
        return Err(format!("{} exists; export-gguf writes a new file", out.display()).into());
    }
    let config_path = model_dir.join("config.json");
    let config = ModelLoader::load_config(&config_path)?;
    if config.model_type.as_deref() != Some("llama") {
        return Err(format!(
            "{} has model_type {}; export-gguf writes Llama-style models only",
            config_path.display(),
            config.model_type.as_deref().unwrap_or("(none)")
        )
        .into());
    }
    // The file carries no rotary scaling, so a scaled RoPE would load and
    // give wrong logits.
    check_supported(&config).map_err(|e| format!("{}: {e}", config_path.display()))?;
    if config.quantization.is_some() {
        return Err(format!(
            "{} has quantization: the base is quantized; export-gguf needs an F32, F16 or BF16 checkpoint",
            config_path.display()
        )
        .into());
    }
    let context_length = config
        .max_position_embeddings
        .ok_or_else(|| format!("{} has no max_position_embeddings", config_path.display()))?;
    let heads = config.num_attention_heads;
    let head_dim = config.head_dim.unwrap_or(config.hidden_size / heads);

    let mut weights = WeightsContext::new();
    weights.load_model_dir(model_dir)?;
    let mut hf_names: Vec<&str> = weights.tensor_names().collect();
    hf_names.sort_unstable();
    if let Some(scales) = hf_names.iter().find(|n| n.ends_with(".scales")) {
        return Err(format!(
            "the base is pre-quantized ({scales} found); export-gguf needs an F32, F16 or BF16 checkpoint"
        )
        .into());
    }
    let mut adapters = match adapter {
        Some(adapter) => adapters_by_target(adapter, &weights)?,
        None => HashMap::new(),
    };
    let mut tensors = Vec::with_capacity(hf_names.len() + 1);
    for hf_name in hf_names {
        // llama.cpp computes the rotary frequencies itself.
        if hf_name.ends_with(".rotary_emb.inv_freq") {
            continue;
        }
        let low_rank = adapters.remove(hf_name).map(|(_, low_rank)| low_rank);
        tensors.push(describe_tensor(
            &weights,
            hf_name,
            hf_name.to_string(),
            low_rank,
            &config,
            precision,
        )?);
    }
    // What is left is an lm_head tied to the embeddings.
    for (target, (source, low_rank)) in adapters {
        tensors.push(describe_tensor(
            &weights,
            &target,
            source,
            Some(low_rank),
            &config,
            precision,
        )?);
    }

    let n_vocab = weights
        .get_tensor_metadata("model.embed_tokens.weight")?
        .shape
        .first()
        .copied()
        .unwrap_or(0);
    let u32_of = |key: &str, value: usize| (key.to_string(), MetaValue::U32(value as u32));
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            MetaValue::Str("llama".to_string()),
        ),
        ("general.name".to_string(), MetaValue::Str(name.to_string())),
        u32_of("general.alignment", ALIGNMENT),
        u32_of(
            "general.file_type",
            usize::from(precision == GgufPrecision::F16),
        ),
        u32_of("llama.context_length", context_length),
        u32_of("llama.embedding_length", config.hidden_size),
        u32_of("llama.block_count", config.num_hidden_layers),
        u32_of("llama.feed_forward_length", config.intermediate_size),
        u32_of("llama.attention.head_count", heads),
        u32_of(
            "llama.attention.head_count_kv",
            config.num_key_value_heads.unwrap_or(heads),
        ),
        u32_of("llama.attention.key_length", head_dim),
        u32_of("llama.attention.value_length", head_dim),
        (
            "llama.attention.layer_norm_rms_epsilon".to_string(),
            MetaValue::F32(config.norm_eps),
        ),
        (
            "llama.rope.freq_base".to_string(),
            MetaValue::F32(
                config
                    .rope_theta
                    .or_else(|| config.rope_parameters.as_ref()?.rope_theta)
                    .unwrap_or(10_000.0) as f32,
            ),
        ),
        u32_of("llama.rope.dimension_count", head_dim),
        u32_of("llama.vocab_size", n_vocab),
    ];
    metadata.extend(tokenizer_metadata(model_dir, &config, n_vocab)?);
    let header = gguf_header(&metadata, &tensors);

    write_through_temp_file(out, |file| {
        file.write_all(&header)?;
        for tensor in &tensors {
            let mut written = Ok(());
            weights
                .read_bytes(&tensor.source, |bytes, _| {
                    written = write_tensor(tensor, bytes, &mut *file);
                })
                .map_err(std::io::Error::other)?;
            written?;
        }
        Ok(())
    })?;
    Ok(GgufReport {
        tensors: tensors.len(),
        merged: tensors.iter().filter(|t| t.low_rank.is_some()).count(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::checkpoint::LoraMetadata;
    use std::path::PathBuf;

    /// Reads GGUF bytes back: metadata, tensor infos (name, dims, ggml type,
    /// offset) and where the tensor data starts.
    struct Cursor<'a> {
        bytes: &'a [u8],
        at: usize,
    }

    impl Cursor<'_> {
        fn take<const N: usize>(&mut self) -> [u8; N] {
            let taken = self.bytes[self.at..self.at + N].try_into().unwrap();
            self.at += N;
            taken
        }

        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take())
        }

        fn u64(&mut self) -> u64 {
            u64::from_le_bytes(self.take())
        }

        fn string(&mut self) -> String {
            let len = self.u64() as usize;
            let text = String::from_utf8(self.bytes[self.at..self.at + len].to_vec()).unwrap();
            self.at += len;
            text
        }

        fn value(&mut self) -> MetaValue {
            match self.u32() {
                4 => MetaValue::U32(self.u32()),
                6 => MetaValue::F32(f32::from_le_bytes(self.take())),
                7 => MetaValue::Bool(self.take::<1>()[0] == 1),
                8 => MetaValue::Str(self.string()),
                9 => {
                    let item_type = self.u32();
                    let len = self.u64();
                    match item_type {
                        8 => MetaValue::Strs((0..len).map(|_| self.string()).collect()),
                        5 => MetaValue::I32s(
                            (0..len).map(|_| i32::from_le_bytes(self.take())).collect(),
                        ),
                        other => panic!("array of value type {other}"),
                    }
                }
                other => panic!("value type {other}"),
            }
        }
    }

    struct Parsed {
        metadata: Vec<(String, MetaValue)>,
        tensors: Vec<(String, Vec<u64>, u32, u64)>,
        data_start: usize,
    }

    fn parse(bytes: &[u8]) -> Parsed {
        let mut cursor = Cursor { bytes, at: 0 };
        assert_eq!(&cursor.take::<4>(), b"GGUF");
        assert_eq!(cursor.u32(), 3, "version");
        let n_tensors = cursor.u64();
        let n_metadata = cursor.u64();
        let metadata = (0..n_metadata)
            .map(|_| (cursor.string(), cursor.value()))
            .collect();
        let tensors = (0..n_tensors)
            .map(|_| {
                let name = cursor.string();
                let n_dims = cursor.u32();
                let dims = (0..n_dims).map(|_| cursor.u64()).collect();
                (name, dims, cursor.u32(), cursor.u64())
            })
            .collect();
        Parsed {
            metadata,
            tensors,
            data_start: cursor.at.next_multiple_of(ALIGNMENT),
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ferrotensor_gguf_{}_{tag}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_json(path: &Path, value: &Json) {
        std::fs::write(path, value.to_string()).unwrap();
    }

    /// A byte-level BPE tokenizer with 4 words, 3 added tokens and both
    /// merge formats, and its config.
    fn write_tiny_tokenizer(dir: &Path) {
        write_json(
            &dir.join("tokenizer.json"),
            &serde_json::json!({
                "added_tokens": [
                    { "id": 4, "content": "<|endoftext|>", "special": true },
                    { "id": 5, "content": "<|im_end|>", "special": true },
                    { "id": 6, "content": "<x>", "special": false }
                ],
                "pre_tokenizer": {
                    "type": "Sequence",
                    "pretokenizers": [
                        { "type": "Digits", "individual_digits": true },
                        { "type": "ByteLevel", "add_prefix_space": false, "use_regex": true }
                    ]
                },
                "model": {
                    "type": "BPE",
                    "vocab": { "a": 0, "b": 1, "ab": 2, "abb": 3 },
                    "merges": ["a b", ["ab", "b"]]
                }
            }),
        );
        write_json(
            &dir.join("tokenizer_config.json"),
            &serde_json::json!({
                "bos_token": "<|endoftext|>",
                "eos_token": { "content": "<|im_end|>" },
                "add_bos_token": false,
                "chat_template": "{{ messages }}"
            }),
        );
    }

    /// The tokenizer metadata of [`write_tiny_tokenizer`] for 8 embedding rows,
    /// written out by hand.
    fn tiny_tokenizer_metadata() -> Vec<(String, MetaValue)> {
        let strs = |items: &[&str]| MetaValue::Strs(items.iter().map(|s| s.to_string()).collect());
        vec![
            ("tokenizer.ggml.model".into(), MetaValue::Str("gpt2".into())),
            ("tokenizer.ggml.pre".into(), MetaValue::Str("smollm".into())),
            (
                "tokenizer.ggml.tokens".into(),
                strs(&[
                    "a",
                    "b",
                    "ab",
                    "abb",
                    "<|endoftext|>",
                    "<|im_end|>",
                    "<x>",
                    "[PAD7]",
                ]),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::I32s(vec![1, 1, 1, 1, 3, 3, 4, 5]),
            ),
            ("tokenizer.ggml.merges".into(), strs(&["a b", "ab b"])),
            ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(4)),
            ("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(5)),
            (
                "tokenizer.ggml.add_bos_token".into(),
                MetaValue::Bool(false),
            ),
            (
                "tokenizer.chat_template".into(),
                MetaValue::Str("{{ messages }}".into()),
            ),
        ]
    }

    /// Tensors of the tiny model: 8 tokens, width 8, 2 query heads of size
    /// 4, 1 key/value head, feed-forward width 4, one block, untied output.
    /// Values are multiples of 0.25 in [-5, 5], exact in f16.
    fn tiny_tensors() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("lm_head.weight", vec![8, 8]),
            ("model.embed_tokens.weight", vec![8, 8]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
            ("model.layers.0.post_attention_layernorm.weight", vec![8]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 8]),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.rotary_emb.inv_freq", vec![2]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 8]),
            ("model.norm.weight", vec![8]),
        ]
    }

    fn tiny_values(tensor_idx: usize, len: usize) -> Vec<f32> {
        (0..len)
            .map(|e| ((e + 3 * tensor_idx) % 41) as f32 * 0.25 - 5.0)
            .collect()
    }

    fn tiny_config() -> Json {
        serde_json::json!({
            "model_type": "llama",
            "hidden_size": 8,
            "intermediate_size": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        })
    }

    fn write_tiny_model(tag: &str) -> PathBuf {
        let dir = temp_dir(tag);
        write_json(&dir.join("config.json"), &tiny_config());
        let bytes: Vec<(&str, Vec<usize>, Vec<u8>)> = tiny_tensors()
            .into_iter()
            .enumerate()
            .map(|(idx, (name, shape))| {
                let values = tiny_values(idx, shape.iter().product());
                (
                    name,
                    shape,
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                )
            })
            .collect();
        let views = bytes.iter().map(|(name, shape, bytes)| {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .unwrap();
            (name.to_string(), view)
        });
        let file = safetensors::tensor::serialize(views, &None).unwrap();
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        write_tiny_tokenizer(&dir);
        dir
    }

    /// `values` (`[rows, cols]`) with the rows in `row_order`, encoded as f16
    /// or f32.
    fn expected_bytes(values: &[f32], cols: usize, row_order: &[usize], f16: bool) -> Vec<u8> {
        row_order
            .iter()
            .flat_map(|&r| &values[r * cols..(r + 1) * cols])
            .flat_map(|&v| {
                if f16 {
                    half::f16::from_f32(v).to_le_bytes().to_vec()
                } else {
                    v.to_le_bytes().to_vec()
                }
            })
            .collect()
    }

    /// Reads the exported file and checks the tensor table against the
    /// source: names, reversed dims, types, aligned back-to-back offsets and
    /// bytes. Returns the metadata. `q_row0_delta` is added to the first row
    /// of the query weight in the expectation.
    fn check_export(file: &Path, f16: bool, q_row0_delta: f32) -> Vec<(String, MetaValue)> {
        let bytes = std::fs::read(file).unwrap();
        let parsed = parse(&bytes);
        let gguf_names = [
            "output.weight",
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_v.weight",
            "output_norm.weight",
        ];
        let sources: Vec<(usize, (&str, Vec<usize>))> = tiny_tensors()
            .into_iter()
            .enumerate()
            .filter(|(_, (name, _))| !name.ends_with("inv_freq"))
            .collect();
        assert_eq!(parsed.tensors.len(), gguf_names.len());
        let mut offset = 0;
        for ((name, dims, ggml_type, tensor_offset), (gguf_name, (idx, (hf_name, shape)))) in
            parsed.tensors.iter().zip(gguf_names.iter().zip(sources))
        {
            assert_eq!(name, gguf_name);
            let reversed: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
            assert_eq!(dims, &reversed, "{name}");
            let half = f16 && shape.len() == 2;
            assert_eq!(*ggml_type, u32::from(half), "{name}");
            assert_eq!(*tensor_offset, offset, "{name}");
            assert_eq!(tensor_offset % ALIGNMENT as u64, 0);

            let mut values = tiny_values(idx, shape.iter().product());
            let cols = *shape.last().unwrap();
            let rows = values.len() / cols;
            // Hand-written llama.cpp row order: per head of 4 rows, 0 2 1 3.
            let row_order: Vec<usize> = match hf_name {
                "model.layers.0.self_attn.q_proj.weight" => {
                    values[..cols].iter_mut().for_each(|v| *v += q_row0_delta);
                    vec![0, 2, 1, 3, 4, 6, 5, 7]
                }
                "model.layers.0.self_attn.k_proj.weight" => vec![0, 2, 1, 3],
                _ => (0..rows).collect(),
            };
            let expected = expected_bytes(&values, cols, &row_order, half);
            let start = parsed.data_start + *tensor_offset as usize;
            assert_eq!(&bytes[start..start + expected.len()], expected, "{name}");
            offset += expected.len().next_multiple_of(ALIGNMENT) as u64;
        }
        assert_eq!(bytes.len(), parsed.data_start + offset as usize);
        parsed.metadata
    }

    fn expected_metadata(file_type: u32) -> Vec<(String, MetaValue)> {
        let u32_of = |key: &str, v: u32| (key.to_string(), MetaValue::U32(v));
        let mut metadata = vec![
            (
                "general.architecture".to_string(),
                MetaValue::Str("llama".into()),
            ),
            ("general.name".to_string(), MetaValue::Str("tiny".into())),
            u32_of("general.alignment", 32),
            u32_of("general.file_type", file_type),
            u32_of("llama.context_length", 64),
            u32_of("llama.embedding_length", 8),
            u32_of("llama.block_count", 1),
            u32_of("llama.feed_forward_length", 4),
            u32_of("llama.attention.head_count", 2),
            u32_of("llama.attention.head_count_kv", 1),
            u32_of("llama.attention.key_length", 4),
            u32_of("llama.attention.value_length", 4),
            (
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                MetaValue::F32(1e-5),
            ),
            ("llama.rope.freq_base".to_string(), MetaValue::F32(10000.0)),
            u32_of("llama.rope.dimension_count", 4),
            u32_of("llama.vocab_size", 8),
        ];
        metadata.extend(tiny_tokenizer_metadata());
        metadata
    }

    #[test]
    fn f16_export_has_every_key_tensor_and_byte() {
        let dir = write_tiny_model("f16");
        let out = dir.join("tiny.gguf");
        let report = export_gguf(&dir, "tiny", None, &out, GgufPrecision::F16).unwrap();
        assert_eq!(
            report,
            GgufReport {
                tensors: 12,
                merged: 0
            }
        );
        assert_eq!(check_export(&out, true, 0.0), expected_metadata(1));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn f32_export_keeps_every_tensor_in_f32() {
        let dir = write_tiny_model("f32");
        let out = dir.join("tiny.gguf");
        export_gguf(&dir, "tiny", None, &out, GgufPrecision::F32).unwrap();
        assert_eq!(check_export(&out, false, 0.0), expected_metadata(0));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn an_adapter_is_added_before_the_rows_are_reordered() {
        let dir = write_tiny_model("lora");
        // Rank 1, alpha 1: the delta (A B)^T has B[o] * A[i] at row o, so
        // only row 0 of q_proj changes, by 0.25 everywhere.
        let mut b = vec![0.0; 8];
        b[0] = 1.0;
        let adapter = LoraCheckpoint {
            tensors: HashMap::from([
                (
                    "model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                    (vec![8, 1], vec![0.25; 8]),
                ),
                (
                    "model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
                    (vec![1, 8], b),
                ),
            ]),
            metadata: Some(LoraMetadata {
                rank: 1,
                alpha: 1.0,
                target_modules: vec!["q_proj".to_string()],
                trained_modules: None,
            }),
        };
        let out = dir.join("tiny.gguf");
        let report = export_gguf(&dir, "tiny", Some(&adapter), &out, GgufPrecision::F16).unwrap();
        assert_eq!(
            report,
            GgufReport {
                tensors: 12,
                merged: 1
            }
        );
        check_export(&out, true, 0.25);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_the_tokenizer_section_of_a_tiny_tokenizer() {
        let dir = temp_dir("tokenizer");
        write_tiny_tokenizer(&dir);
        let config: Lfm2Config = serde_json::from_value(tiny_config()).unwrap();
        assert_eq!(
            tokenizer_metadata(&dir, &config, 8).unwrap(),
            tiny_tokenizer_metadata()
        );
        let err = tokenizer_metadata(&dir, &config, 6)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "tokenizer.json gives \"<x>\" the id 6, outside the 6 rows of the embeddings"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn names_the_two_byte_level_pre_tokenizers_and_no_other() {
        let byte_level = serde_json::json!({ "type": "ByteLevel", "add_prefix_space": false });
        assert_eq!(pre_tokenizer_name(&byte_level), Some("gpt-2"));
        let metaspace = serde_json::json!({ "type": "Metaspace", "replacement": "▁" });
        assert_eq!(pre_tokenizer_name(&metaspace), None);
        let no_regex = serde_json::json!({ "type": "ByteLevel", "use_regex": false });
        assert_eq!(pre_tokenizer_name(&no_regex), None);
    }

    /// Exports after `change` edits the tiny model and returns the error;
    /// checks that no file was written.
    fn refusal(tag: &str, change: impl FnOnce(&Path)) -> String {
        let dir = write_tiny_model(tag);
        change(&dir);
        let out = dir.join("refused.gguf");
        let err = export_gguf(&dir, "tiny", None, &out, GgufPrecision::F16).unwrap_err();
        assert!(!err.is::<std::io::Error>(), "{err}");
        assert!(!out.exists() || std::fs::read(&out).unwrap() == b"keep");
        assert!(!dir.join("refused.gguf.tmp").exists());
        std::fs::remove_dir_all(dir).unwrap();
        err.to_string()
    }

    #[test]
    fn refuses_an_existing_file_another_model_a_word_level_tokenizer_and_a_quantized_base() {
        let err = refusal("exists", |dir| {
            std::fs::write(dir.join("refused.gguf"), b"keep").unwrap();
        });
        assert!(
            err.ends_with("refused.gguf exists; export-gguf writes a new file"),
            "{err}"
        );

        let err = refusal("lfm2", |dir| {
            let mut config = tiny_config();
            config["model_type"] = "lfm2".into();
            write_json(&dir.join("config.json"), &config);
        });
        assert!(
            err.ends_with("has model_type lfm2; export-gguf writes Llama-style models only"),
            "{err}"
        );

        let err = refusal("wordlevel", |dir| {
            let path = dir.join("tokenizer.json");
            let mut tokenizer = read_json(&path).unwrap();
            tokenizer["model"]["type"] = "WordLevel".into();
            write_json(&path, &tokenizer);
        });
        assert_eq!(
            err,
            "tokenizer.json has a \"WordLevel\" model; export-gguf needs a byte-level BPE tokenizer"
        );

        let err = refusal("quantized", |dir| {
            let mut config = tiny_config();
            config["quantization"] =
                serde_json::json!({ "group_size": 32, "bits": 4, "mode": "affine" });
            write_json(&dir.join("config.json"), &config);
        });
        assert!(
            err.ends_with(
                "has quantization: the base is quantized; export-gguf needs an F32, F16 or BF16 checkpoint"
            ),
            "{err}"
        );
    }

    #[test]
    fn refuses_scaled_rope_and_qwen2_style_biases() {
        let err = refusal("rope-scaling", |dir| {
            let mut config = tiny_config();
            config["rope_scaling"] = serde_json::json!({ "rope_type": "llama3", "factor": 8.0 });
            write_json(&dir.join("config.json"), &config);
        });
        assert!(
            err.ends_with(
                "config.json: rope_scaling type \"llama3\" is not supported: RoPE runs unscaled"
            ),
            "{err}"
        );

        // A second shard, listed in the index, adds the bias.
        let err = refusal("bias", |dir| {
            let bias = [0u8; 32];
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &bias)
                    .unwrap();
            let name = "model.layers.0.self_attn.q_proj.bias";
            let file = safetensors::tensor::serialize([(name, view)], &None).unwrap();
            std::fs::write(dir.join("bias.safetensors"), file).unwrap();
            write_json(
                &dir.join("model.safetensors.index.json"),
                &serde_json::json!({ "weight_map": {
                    "model.embed_tokens.weight": "model.safetensors",
                    "model.layers.0.self_attn.q_proj.bias": "bias.safetensors"
                } }),
            );
        });
        assert_eq!(
            err,
            "tensor model.layers.0.self_attn.q_proj.bias is a projection bias; export-gguf writes \
             Llama weights without biases, so Qwen2-style models (q/k/v biases) are not exported"
        );
    }
}

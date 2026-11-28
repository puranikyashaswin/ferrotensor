//! Writing a Llama or Qwen2 model as a GGUF file for llama.cpp and Ollama.
//!
//! [`export_gguf`] writes GGUF version 3 (ggml `docs/gguf.md`) with the
//! metadata keys, tensor names and conventions of llama.cpp's
//! `convert_hf_to_gguf.py`: dimensions are listed innermost first, and for
//! the `llama` architecture the rows of the query and key projections are
//! reordered from the Hugging Face rotary layout (the two halves of each
//! head) to llama.cpp's (adjacent pairs). llama.cpp rotates `qwen2` in the
//! Hugging Face layout, so those rows stay as they are, and the query, key
//! and value biases are written next to the weights. Like
//! [`merge`](crate::merge) the export streams: one row of one tensor is in
//! memory at a time, plus the adapter and the tokenizer.

use crate::checkpoint::LoraCheckpoint;
use crate::merge::{adapters_by_target, write_through_temp_file, Float, LowRank};
use ferrotensor_graph::model_builder::check_supported;
use ferrotensor_graph::rope::rope_scaling;
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
const UNKNOWN: i32 = 2;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;
const BYTE: i32 = 6;

/// Element type of the 2-D weights in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufPrecision {
    /// f16, with 1-D tensors and norms in f32 (`general.file_type` 1).
    F16,
    /// f32 for every tensor (`general.file_type` 0).
    F32,
}

/// The llama.cpp architecture a checkpoint is written as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arch {
    /// `model_type` `llama`: no biases, query and key rows reordered.
    Llama,
    /// `model_type` `qwen2`: query, key and value biases, rows as stored.
    Qwen2,
}

impl Arch {
    fn of(model_type: Option<&str>) -> Option<Self> {
        match model_type {
            Some("llama") => Some(Self::Llama),
            Some("qwen2") => Some(Self::Qwen2),
            _ => None,
        }
    }

    /// `general.architecture`, which is also the prefix of the model's keys.
    fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }
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
    F32s(Vec<f32>),
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
            Self::F32s(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(6u32.to_le_bytes());
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

/// llama.cpp's name for the Hugging Face tensor `hf_name` of an `arch`
/// model. The query, key and value biases have one for `qwen2` only.
fn gguf_tensor_name(hf_name: &str, arch: Arch) -> Option<String> {
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
        "input_layernorm.weight" => "attn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "self_attn.q_proj.bias" if arch == Arch::Qwen2 => "attn_q.bias",
        "self_attn.k_proj.bias" if arch == Arch::Qwen2 => "attn_k.bias",
        "self_attn.v_proj.bias" if arch == Arch::Qwen2 => "attn_v.bias",
        _ => return None,
    };
    Some(format!("blk.{block}.{short}"))
}

/// The output tensor for the Hugging Face tensor `hf_name`, read from
/// `source` with `low_rank` added.
fn describe_tensor<'a>(
    weights: &WeightsContext,
    hf_name: &str,
    source: String,
    low_rank: Option<LowRank<'a>>,
    config: &Lfm2Config,
    arch: Arch,
    precision: GgufPrecision,
) -> Result<GgufTensor<'a>> {
    let name = gguf_tensor_name(hf_name, arch).ok_or_else(|| {
        if hf_name.ends_with(".bias") {
            format!(
                "tensor {hf_name} is a bias; the only biases export-gguf writes are the query, \
                 key and value biases of model_type qwen2"
            )
        } else {
            format!(
                "tensor {hf_name} has no llama.cpp name; export-gguf writes Llama and Qwen2 \
                 weights only"
            )
        }
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
    // llama.cpp rotates adjacent pairs for `llama` and the two halves of a
    // head, as Hugging Face stores them, for `qwen2`.
    let heads = config.num_attention_heads;
    let rope_heads = if arch != Arch::Llama {
        None
    } else if hf_name.ends_with(".self_attn.q_proj.weight") {
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

/// The split of Qwen2's pre-tokenizer in `tokenizer.json`. llama.cpp's
/// `qwen2` pre-tokenizer is this expression with the case-insensitive group
/// spelled out.
const QWEN2_SPLIT: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+|\s+(?!\S)|\s+",
);

/// llama.cpp's name for a byte-level pre-tokenizer it implements: `gpt-2`
/// for GPT-2's split, `smollm` for single digits split off first, `qwen2`
/// for Qwen2's own split followed by a byte-level step without one.
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
        [split, bytes]
            if pre["type"] == "Sequence"
                && split["type"] == "Split"
                && split["pattern"]["Regex"] == QWEN2_SPLIT
                && split["behavior"] == "Isolated"
                && split["invert"] != true
                && bytes["type"] == "ByteLevel"
                && bytes["use_regex"] == false
                && bytes["add_prefix_space"] != true =>
        {
            Some("qwen2")
        }
        _ => None,
    }
}

/// Whether a BPE `tokenizer.json` is a SentencePiece tokenizer in the one
/// form llama.cpp's `llama` tokenizer reproduces, TinyLlama's: byte
/// fallback, no `ignore_merges`, no pre-tokenizer, and a normalizer that puts
/// `▁` in front of every piece of text between special tokens and turns each
/// space into `▁`. A Metaspace pre-tokenizer does not count: it leaves out
/// the leading `▁` when the text starts with a space, and with
/// `prepend_scheme` `first` also after a special token, where llama.cpp
/// puts it in.
fn is_sentencepiece(tokenizer: &Json) -> bool {
    let model = &tokenizer["model"];
    let normalizer = &tokenizer["normalizer"];
    let steps = normalizer["normalizers"].as_array().map(Vec::as_slice);
    model["byte_fallback"] == true
        && model["ignore_merges"] != true
        && tokenizer["pre_tokenizer"].is_null()
        && normalizer["type"] == "Sequence"
        && matches!(steps, Some([prepend, replace])
            if prepend["type"] == "Prepend"
                && prepend["prepend"] == "\u{2581}"
                && replace["type"] == "Replace"
                && replace["pattern"]["String"] == " "
                && replace["content"] == "\u{2581}")
}

/// Whether `token` is a byte token of a SentencePiece vocabulary, `<0x41>`.
fn is_byte_token(token: &str) -> bool {
    let bytes = token.as_bytes();
    bytes.len() == 6
        && token.starts_with("<0x")
        && token.ends_with('>')
        && bytes[3..5].iter().all(u8::is_ascii_hexdigit)
}

/// The text a merge of `tokenizer.json` makes: `"a b"` or `["a", "b"]` give
/// `ab`.
fn merged_text(merge: &Json) -> Option<String> {
    match merge {
        Json::String(pair) => pair
            .split_once(' ')
            .map(|(left, right)| left.to_string() + right),
        Json::Array(pair) => match pair.as_slice() {
            [Json::String(left), Json::String(right)] => Some(left.clone() + right),
            _ => None,
        },
        _ => None,
    }
}

/// `tokenizer.ggml.scores` of a SentencePiece vocabulary. `tokenizer.json`
/// keeps no scores, but its merges are listed from the highest SentencePiece
/// score down, so a normal token scores minus one more than the rank of the
/// first merge that makes it. llama.cpp merges the pair whose token scores
/// highest first, so it merges in the order of the list. A normal token no
/// merge makes, a single character, scores below all of them. Byte tokens
/// and the unknown token score 0, added tokens -1000 and padding -10000, the
/// numbers llama.cpp's converter writes for them.
fn sentencepiece_scores(tokens: &[String], token_types: &[i32], merges: &[Json]) -> Vec<f32> {
    let ids: HashMap<&str, usize> = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| (token.as_str(), id))
        .collect();
    let mut from_merges = vec![None; tokens.len()];
    for (rank, merge) in merges.iter().enumerate() {
        if let Some(&id) = merged_text(merge).and_then(|text| ids.get(text.as_str())) {
            from_merges[id].get_or_insert(-((rank + 1) as f32));
        }
    }
    let unmerged = -((merges.len() + 1) as f32);
    token_types
        .iter()
        .zip(from_merges)
        .map(|(&token_type, score)| match token_type {
            NORMAL => score.unwrap_or(unmerged),
            CONTROL | USER_DEFINED => -1000.0,
            UNUSED => -10_000.0,
            _ => 0.0,
        })
        .collect()
}

/// Whether the `normalizer` of a `tokenizer.json` applies NFC, alone or as a
/// step of a sequence.
fn normalizes_nfc(normalizer: &Json) -> bool {
    normalizer["type"] == "NFC"
        || (normalizer["type"] == "Sequence"
            && normalizer["normalizers"]
                .as_array()
                .is_some_and(|steps| steps.iter().any(normalizes_nfc)))
}

/// The `tokenizer.*` metadata of the byte-level BPE or SentencePiece
/// tokenizer in `model_dir`, padded with unused tokens to the `n_vocab`
/// embedding rows.
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
            "tokenizer.json has a {} model; export-gguf writes only BPE tokenizers, byte-level \
             or SentencePiece BPE with byte fallback such as TinyLlama's",
            model["type"]
        )
        .into());
    }
    let pre = pre_tokenizer_name(&tokenizer["pre_tokenizer"]);
    let sentencepiece = pre.is_none() && is_sentencepiece(&tokenizer);
    if pre.is_none() && !sentencepiece {
        return Err(format!(
            "tokenizer.json has the pre-tokenizer {} and the normalizer {}; export-gguf knows the \
             byte-level ones of GPT-2, SmolLM and Qwen2 (llama.cpp `gpt-2`, `smollm` and `qwen2`) \
             and SentencePiece with byte fallback, no pre-tokenizer and a normalizer that puts \
             \u{2581} in front and turns spaces into \u{2581}, as TinyLlama's (llama.cpp `llama`)",
            tokenizer["pre_tokenizer"], tokenizer["normalizer"]
        )
        .into());
    }

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
        let token_type = if !sentencepiece {
            NORMAL
        } else if is_byte_token(token) {
            BYTE
        } else if model["unk_token"] == token.as_str() {
            UNKNOWN
        } else {
            NORMAL
        };
        place(token, id, token_type)?;
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

    let merge_list = model["merges"]
        .as_array()
        .ok_or("tokenizer.json has no model.merges")?;
    let merges = merge_list
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
    // llama.cpp's SentencePiece tokenizer reads scores and no merges.
    let scores = sentencepiece.then(|| sentencepiece_scores(&tokens, &token_types, merge_list));
    let mut metadata = vec![
        (
            "tokenizer.ggml.model".to_string(),
            MetaValue::Str(if sentencepiece { "llama" } else { "gpt2" }.to_string()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            MetaValue::Str(pre.unwrap_or("default").to_string()),
        ),
        ("tokenizer.ggml.tokens".to_string(), MetaValue::Strs(tokens)),
    ];
    if let Some(scores) = scores {
        metadata.push(("tokenizer.ggml.scores".to_string(), MetaValue::F32s(scores)));
    }
    metadata.push((
        "tokenizer.ggml.token_type".to_string(),
        MetaValue::I32s(token_types),
    ));
    if !sentencepiece {
        metadata.push(("tokenizer.ggml.merges".to_string(), MetaValue::Strs(merges)));
    }
    metadata.extend(special_ids);
    for key in ["add_bos_token", "add_eos_token"] {
        if let Some(add) = tokenizer_config[key].as_bool() {
            metadata.push((format!("tokenizer.ggml.{key}"), MetaValue::Bool(add)));
        }
    }
    // Written as convert_hf_to_gguf.py writes it. llama.cpp reads the key only
    // for UGM tokenizers today, so it does not normalize a BPE `qwen2` prompt.
    if normalizes_nfc(&tokenizer["normalizer"]) {
        metadata.push((
            "tokenizer.ggml.normalizer.nfc".to_string(),
            MetaValue::Bool(true),
        ));
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

/// Write the Llama or Qwen2 model in `model_dir`, with `adapter` added to
/// its weights when given, to the new GGUF file `out`, named `name`.
///
/// # Errors
///
/// Fails, before writing, when `out` exists, when `config.json` is not
/// `model_type` `llama` or `qwen2` or has no `max_position_embeddings`, when
/// the base is quantized (`quantization` in `config.json`, `.scales` tensors
/// or a tensor that is not F32, F16 or BF16), when a tensor is not one the
/// export writes for that architecture (a projection bias in a `llama`
/// model, say), when the tokenizer is not byte-level BPE with a
/// pre-tokenizer llama.cpp knows, or when the adapter does not match the
/// base. I/O errors while writing are returned as they occur, as
/// `std::io::Error`.
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
    let arch = Arch::of(config.model_type.as_deref()).ok_or_else(|| {
        format!(
            "{} has model_type {}; export-gguf writes llama and qwen2 models only",
            config_path.display(),
            config.model_type.as_deref().unwrap_or("(none)")
        )
    })?;
    check_supported(&config).map_err(|e| format!("{}: {e}", config_path.display()))?;
    // The file carries no rotary scaling, so a scaled RoPE would load and
    // give wrong logits.
    if rope_scaling(&config)
        .map_err(|e| format!("{}: {e}", config_path.display()))?
        .is_some()
    {
        return Err(format!(
            "{}: scaled RoPE (rope_type \"llama3\") is not supported: the GGUF export writes unscaled RoPE",
            config_path.display()
        )
        .into());
    }
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
            arch,
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
            arch,
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
    // The model's keys carry the architecture name as their prefix.
    let key = |suffix: &str| format!("{}.{suffix}", arch.name());
    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            MetaValue::Str(arch.name().to_string()),
        ),
        ("general.name".to_string(), MetaValue::Str(name.to_string())),
        u32_of("general.alignment", ALIGNMENT),
        u32_of(
            "general.file_type",
            usize::from(precision == GgufPrecision::F16),
        ),
        u32_of(&key("context_length"), context_length),
        u32_of(&key("embedding_length"), config.hidden_size),
        u32_of(&key("block_count"), config.num_hidden_layers),
        u32_of(&key("feed_forward_length"), config.intermediate_size),
        u32_of(&key("attention.head_count"), heads),
        u32_of(
            &key("attention.head_count_kv"),
            config.num_key_value_heads.unwrap_or(heads),
        ),
        u32_of(&key("attention.key_length"), head_dim),
        u32_of(&key("attention.value_length"), head_dim),
        (
            key("attention.layer_norm_rms_epsilon"),
            MetaValue::F32(config.norm_eps),
        ),
        (
            key("rope.freq_base"),
            MetaValue::F32(
                config
                    .rope_theta
                    .or_else(|| config.rope_parameters.as_ref()?.rope_theta)
                    .unwrap_or(10_000.0) as f32,
            ),
        ),
        u32_of(&key("rope.dimension_count"), head_dim),
        u32_of(&key("vocab_size"), n_vocab),
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
                        6 => MetaValue::F32s(
                            (0..len).map(|_| f32::from_le_bytes(self.take())).collect(),
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
        write_model(tag, &tiny_config(), tiny_tensors(), write_tiny_tokenizer)
    }

    /// A model directory with `config`, `tensors` filled by [`tiny_values`]
    /// in f32, and the tokenizer files `write_tokenizer` writes.
    fn write_model(
        tag: &str,
        config: &Json,
        tensors: Vec<(&'static str, Vec<usize>)>,
        write_tokenizer: fn(&Path),
    ) -> PathBuf {
        let dir = temp_dir(tag);
        write_json(&dir.join("config.json"), config);
        let bytes: Vec<(&str, Vec<usize>, Vec<u8>)> = tensors
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
        write_tokenizer(&dir);
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

    /// Checks an export of the tiny Llama model; see [`check_tensors`]. The
    /// row order is hand-written for llama.cpp: per head of 4 rows, 0 2 1 3.
    fn check_export(file: &Path, f16: bool, q_row0_delta: f32) -> Vec<(String, MetaValue)> {
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
        let row_order = |hf_name: &str| match hf_name {
            "model.layers.0.self_attn.q_proj.weight" => Some(vec![0, 2, 1, 3, 4, 6, 5, 7]),
            "model.layers.0.self_attn.k_proj.weight" => Some(vec![0, 2, 1, 3]),
            _ => None,
        };
        check_tensors(
            file,
            f16,
            tiny_tensors(),
            &gguf_names,
            row_order,
            q_row0_delta,
        )
    }

    /// Reads the exported file and checks the tensor table against the
    /// source `tensors`: names, reversed dims, types, aligned back-to-back
    /// offsets and bytes. `row_order` gives the source row of each output
    /// row for the tensors whose rows move. Returns the metadata.
    /// `q_row0_delta` is added to the first row of the query weight in the
    /// expectation.
    fn check_tensors(
        file: &Path,
        f16: bool,
        tensors: Vec<(&'static str, Vec<usize>)>,
        gguf_names: &[&str],
        row_order: impl Fn(&str) -> Option<Vec<usize>>,
        q_row0_delta: f32,
    ) -> Vec<(String, MetaValue)> {
        let bytes = std::fs::read(file).unwrap();
        let parsed = parse(&bytes);
        let sources: Vec<(usize, (&str, Vec<usize>))> = tensors
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
            if hf_name == "model.layers.0.self_attn.q_proj.weight" {
                values[..cols].iter_mut().for_each(|v| *v += q_row0_delta);
            }
            let order = row_order(hf_name).unwrap_or_else(|| (0..rows).collect());
            let expected = expected_bytes(&values, cols, &order, half);
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

    /// The pre-tokenizer of Qwen2.5-0.5B-Instruct's `tokenizer.json`, as it
    /// is spelled there.
    fn qwen2_pre_tokenizer() -> Json {
        serde_json::from_str(
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                 "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
                 "use_regex": false}]}"#,
        )
        .unwrap()
    }

    #[test]
    fn names_the_three_pre_tokenizers_and_no_other() {
        let byte_level = serde_json::json!({ "type": "ByteLevel", "add_prefix_space": false });
        assert_eq!(pre_tokenizer_name(&byte_level), Some("gpt-2"));
        let metaspace = serde_json::json!({ "type": "Metaspace", "replacement": "▁" });
        assert_eq!(pre_tokenizer_name(&metaspace), None);
        let no_regex = serde_json::json!({ "type": "ByteLevel", "use_regex": false });
        assert_eq!(pre_tokenizer_name(&no_regex), None);

        assert_eq!(pre_tokenizer_name(&qwen2_pre_tokenizer()), Some("qwen2"));
        // Another split, or a byte-level step that splits again, is not Qwen2's.
        let mut other_split = qwen2_pre_tokenizer();
        other_split["pretokenizers"][0]["pattern"]["Regex"] = "\\s+".into();
        assert_eq!(pre_tokenizer_name(&other_split), None);
        let mut splits_twice = qwen2_pre_tokenizer();
        splits_twice["pretokenizers"][1]["use_regex"] = true.into();
        assert_eq!(pre_tokenizer_name(&splits_twice), None);
    }

    #[test]
    fn finds_nfc_alone_or_in_a_sequence_of_normalizers() {
        assert!(normalizes_nfc(&serde_json::json!({ "type": "NFC" })));
        assert!(normalizes_nfc(&serde_json::json!({
            "type": "Sequence",
            "normalizers": [{ "type": "Replace" }, { "type": "NFC" }]
        })));
        assert!(!normalizes_nfc(&Json::Null));
        assert!(!normalizes_nfc(&serde_json::json!({ "type": "NFKC" })));
        assert!(!normalizes_nfc(&serde_json::json!({
            "type": "Sequence",
            "normalizers": [{ "type": "Lowercase" }]
        })));
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
            err.ends_with("has model_type lfm2; export-gguf writes llama and qwen2 models only"),
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
            "tokenizer.json has a \"WordLevel\" model; export-gguf writes only BPE tokenizers, \
             byte-level or SentencePiece BPE with byte fallback such as TinyLlama's"
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
            config["rope_scaling"] = serde_json::json!({
                "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
            });
            write_json(&dir.join("config.json"), &config);
        });
        assert!(
            err.ends_with(
                "config.json: scaled RoPE (rope_type \"llama3\") is not supported: the GGUF export writes unscaled RoPE"
            ),
            "{err}"
        );

        // A second shard, listed in the index, adds a bias to the Llama model.
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
            "tensor model.layers.0.self_attn.q_proj.bias is a bias; the only biases export-gguf \
             writes are the query, key and value biases of model_type qwen2"
        );
    }

    /// The tiny model as a Qwen2 checkpoint: query, key and value biases,
    /// the output tied to the embeddings (no `lm_head.weight`).
    fn qwen2_tensors() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight", vec![8, 8]),
            ("model.layers.0.input_layernorm.weight", vec![8]),
            ("model.layers.0.mlp.down_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![4, 8]),
            ("model.layers.0.mlp.up_proj.weight", vec![4, 8]),
            ("model.layers.0.post_attention_layernorm.weight", vec![8]),
            ("model.layers.0.self_attn.k_proj.bias", vec![4]),
            ("model.layers.0.self_attn.k_proj.weight", vec![4, 8]),
            ("model.layers.0.self_attn.o_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.q_proj.bias", vec![8]),
            ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("model.layers.0.self_attn.v_proj.bias", vec![4]),
            ("model.layers.0.self_attn.v_proj.weight", vec![4, 8]),
            ("model.norm.weight", vec![8]),
        ]
    }

    fn qwen2_config() -> Json {
        let mut config = tiny_config();
        config["model_type"] = "qwen2".into();
        config["tie_word_embeddings"] = true.into();
        config["rope_theta"] = 1_000_000.0.into();
        config
    }

    /// [`write_tiny_tokenizer`] with Qwen2's NFC normalizer and pre-tokenizer.
    fn write_qwen2_tokenizer(dir: &Path) {
        write_tiny_tokenizer(dir);
        let path = dir.join("tokenizer.json");
        let mut tokenizer = read_json(&path).unwrap();
        tokenizer["normalizer"] = serde_json::json!({ "type": "NFC" });
        tokenizer["pre_tokenizer"] = qwen2_pre_tokenizer();
        write_json(&path, &tokenizer);
    }

    /// The metadata of the tiny Qwen2 model in f16, written out by hand.
    fn expected_qwen2_metadata() -> Vec<(String, MetaValue)> {
        let u32_of = |key: &str, v: u32| (key.to_string(), MetaValue::U32(v));
        let mut metadata = vec![
            (
                "general.architecture".to_string(),
                MetaValue::Str("qwen2".into()),
            ),
            ("general.name".to_string(), MetaValue::Str("tiny".into())),
            u32_of("general.alignment", 32),
            u32_of("general.file_type", 1),
            u32_of("qwen2.context_length", 64),
            u32_of("qwen2.embedding_length", 8),
            u32_of("qwen2.block_count", 1),
            u32_of("qwen2.feed_forward_length", 4),
            u32_of("qwen2.attention.head_count", 2),
            u32_of("qwen2.attention.head_count_kv", 1),
            u32_of("qwen2.attention.key_length", 4),
            u32_of("qwen2.attention.value_length", 4),
            (
                "qwen2.attention.layer_norm_rms_epsilon".to_string(),
                MetaValue::F32(1e-5),
            ),
            (
                "qwen2.rope.freq_base".to_string(),
                MetaValue::F32(1_000_000.0),
            ),
            u32_of("qwen2.rope.dimension_count", 4),
            u32_of("qwen2.vocab_size", 8),
        ];
        // The tiny tokenizer's entries, with Qwen2's pre-tokenizer name and
        // the NFC flag after `add_bos_token`.
        for (key, value) in tiny_tokenizer_metadata() {
            let nfc_next = key == "tokenizer.ggml.add_bos_token";
            if key == "tokenizer.ggml.pre" {
                metadata.push((key, MetaValue::Str("qwen2".into())));
            } else {
                metadata.push((key, value));
            }
            if nfc_next {
                metadata.push((
                    "tokenizer.ggml.normalizer.nfc".into(),
                    MetaValue::Bool(true),
                ));
            }
        }
        metadata
    }

    #[test]
    fn qwen2_export_has_its_keys_its_biases_and_the_rows_as_stored() {
        let gguf_names = [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.attn_k.bias",
            "blk.0.attn_k.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_q.bias",
            "blk.0.attn_q.weight",
            "blk.0.attn_v.bias",
            "blk.0.attn_v.weight",
            "output_norm.weight",
        ];
        let dir = write_model(
            "qwen2",
            &qwen2_config(),
            qwen2_tensors(),
            write_qwen2_tokenizer,
        );
        let out = dir.join("tiny.gguf");
        let report = export_gguf(&dir, "tiny", None, &out, GgufPrecision::F16).unwrap();
        assert_eq!(
            report,
            GgufReport {
                tensors: 14,
                merged: 0
            }
        );
        // No row moves: llama.cpp rotates qwen2 in the Hugging Face layout.
        let metadata = check_tensors(&out, true, qwen2_tensors(), &gguf_names, |_| None, 0.0);
        assert_eq!(metadata, expected_qwen2_metadata());

        // Each bias is the checkpoint's own bytes: a 1-D tensor stays in f32.
        let checkpoint = std::fs::read(dir.join("model.safetensors")).unwrap();
        let checkpoint = safetensors::SafeTensors::deserialize(&checkpoint).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let parsed = parse(&bytes);
        for (proj, len) in [("q", 8), ("k", 4), ("v", 4)] {
            let hf_name = format!("model.layers.0.self_attn.{proj}_proj.bias");
            let gguf_name = format!("blk.0.attn_{proj}.bias");
            let stored = checkpoint.tensor(&hf_name).unwrap();
            assert_eq!(stored.shape(), [len], "{hf_name}");
            let (_, dims, ggml_type, offset) = parsed
                .tensors
                .iter()
                .find(|(name, ..)| *name == gguf_name)
                .unwrap();
            assert_eq!(dims, &[len as u64], "{gguf_name}");
            assert_eq!(*ggml_type, 0, "{gguf_name}");
            let start = parsed.data_start + *offset as usize;
            assert_eq!(&bytes[start..start + 4 * len], stored.data(), "{gguf_name}");
        }

        // An adapter on the query projection lands on the same, unmoved rows.
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
        let merged = dir.join("merged.gguf");
        let report =
            export_gguf(&dir, "tiny", Some(&adapter), &merged, GgufPrecision::F16).unwrap();
        assert_eq!(report.merged, 1);
        check_tensors(&merged, true, qwen2_tensors(), &gguf_names, |_| None, 0.25);
        std::fs::remove_dir_all(dir).unwrap();

        // Any other bias is refused in a Qwen2 model too.
        let mut tensors = qwen2_tensors();
        tensors.push(("model.layers.0.self_attn.o_proj.bias", vec![8]));
        let dir = write_model(
            "qwen2-o-bias",
            &qwen2_config(),
            tensors,
            write_qwen2_tokenizer,
        );
        let out = dir.join("tiny.gguf");
        let err = export_gguf(&dir, "tiny", None, &out, GgufPrecision::F16).unwrap_err();
        assert_eq!(
            err.to_string(),
            "tensor model.layers.0.self_attn.o_proj.bias is a bias; the only biases export-gguf \
             writes are the query, key and value biases of model_type qwen2"
        );
        assert!(!out.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A SentencePiece tokenizer in TinyLlama's form, with `<unk>` known to
    /// the model only, `<s>` and `</s>` added, one byte token, and `▁▁` with
    /// a small id but the last merge, as Llama 2's runs of `▁` have.
    fn sentencepiece_tokenizer() -> Json {
        serde_json::json!({
            "added_tokens": [
                { "id": 1, "content": "<s>", "special": true },
                { "id": 2, "content": "</s>", "special": true }
            ],
            "normalizer": {
                "type": "Sequence",
                "normalizers": [
                    { "type": "Prepend", "prepend": "\u{2581}" },
                    { "type": "Replace", "pattern": { "String": " " }, "content": "\u{2581}" }
                ]
            },
            "pre_tokenizer": null,
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "byte_fallback": true,
                "vocab": {
                    "<unk>": 0, "<s>": 1, "</s>": 2, "<0x0A>": 3,
                    "\u{2581}\u{2581}": 4, "\u{2581}t": 5, "\u{2581}": 6, "t": 7
                },
                "merges": ["\u{2581} t", ["\u{2581}", "\u{2581}"]]
            }
        })
    }

    fn write_sentencepiece_tokenizer(dir: &Path) {
        write_json(&dir.join("tokenizer.json"), &sentencepiece_tokenizer());
        write_json(
            &dir.join("tokenizer_config.json"),
            &serde_json::json!({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "chat_template": "{{ messages }}"
            }),
        );
    }

    /// The tokenizer metadata of [`write_sentencepiece_tokenizer`] for
    /// `n_vocab` embedding rows (8 or 9), written out by hand.
    fn sentencepiece_metadata(n_vocab: usize) -> Vec<(String, MetaValue)> {
        let mut tokens: Vec<String> = ["<unk>", "<s>", "</s>", "<0x0A>", "\u{2581}\u{2581}"]
            .into_iter()
            .chain(["\u{2581}t", "\u{2581}", "t", "[PAD8]"])
            .map(String::from)
            .collect();
        // `▁t` is made by merge 0 and `▁▁` by merge 1; `▁` and `t` by none.
        let mut scores = vec![
            0.0, -1000.0, -1000.0, 0.0, -2.0, -1.0, -3.0, -3.0, -10_000.0,
        ];
        let mut token_types = vec![
            UNKNOWN, CONTROL, CONTROL, BYTE, NORMAL, NORMAL, NORMAL, NORMAL,
        ];
        token_types.push(UNUSED);
        tokens.truncate(n_vocab);
        scores.truncate(n_vocab);
        token_types.truncate(n_vocab);
        vec![
            (
                "tokenizer.ggml.model".into(),
                MetaValue::Str("llama".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                MetaValue::Str("default".into()),
            ),
            ("tokenizer.ggml.tokens".into(), MetaValue::Strs(tokens)),
            ("tokenizer.ggml.scores".into(), MetaValue::F32s(scores)),
            (
                "tokenizer.ggml.token_type".into(),
                MetaValue::I32s(token_types),
            ),
            ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(1)),
            ("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(2)),
            ("tokenizer.ggml.unknown_token_id".into(), MetaValue::U32(0)),
            (
                "tokenizer.chat_template".into(),
                MetaValue::Str("{{ messages }}".into()),
            ),
        ]
    }

    #[test]
    fn writes_a_sentencepiece_vocabulary_with_scores_from_the_merges() {
        let dir = write_model(
            "sentencepiece",
            &tiny_config(),
            tiny_tensors(),
            write_sentencepiece_tokenizer,
        );
        let config: Lfm2Config = serde_json::from_value(tiny_config()).unwrap();
        assert_eq!(
            tokenizer_metadata(&dir, &config, 9).unwrap(),
            sentencepiece_metadata(9)
        );

        let out = dir.join("tiny.gguf");
        export_gguf(&dir, "tiny", None, &out, GgufPrecision::F16).unwrap();
        let metadata = check_export(&out, true, 0.0);
        let mut expected = expected_metadata(1);
        expected.retain(|(key, _)| !key.starts_with("tokenizer."));
        expected.extend(sentencepiece_metadata(8));
        assert_eq!(metadata, expected);

        // Every written token is the one tokenizer.json gives that id.
        let tokenizer = read_json(&dir.join("tokenizer.json")).unwrap();
        let tokens = match &metadata
            .iter()
            .find(|(key, _)| key == "tokenizer.ggml.tokens")
        {
            Some((_, MetaValue::Strs(tokens))) => tokens.clone(),
            other => panic!("{other:?}"),
        };
        for (token, id) in tokenizer["model"]["vocab"].as_object().unwrap() {
            assert_eq!(&tokens[id.as_u64().unwrap() as usize], token);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_a_metaspace_pre_tokenizer_and_bpe_without_byte_fallback() {
        let refused = |tag: &str, change: fn(&mut Json)| {
            refusal(tag, |dir| {
                write_sentencepiece_tokenizer(dir);
                let mut tokenizer = sentencepiece_tokenizer();
                change(&mut tokenizer);
                write_json(&dir.join("tokenizer.json"), &tokenizer);
            })
        };
        let tail = "export-gguf knows the byte-level ones of GPT-2, SmolLM and Qwen2 (llama.cpp \
                    `gpt-2`, `smollm` and `qwen2`) and SentencePiece with byte fallback, no \
                    pre-tokenizer and a normalizer that puts \u{2581} in front and turns spaces \
                    into \u{2581}, as TinyLlama's (llama.cpp `llama`)";
        // A Metaspace pre-tokenizer leaves out the `▁` that llama.cpp puts
        // in front of " Hello".
        let err = refused("metaspace", |tokenizer| {
            tokenizer["normalizer"] = Json::Null;
            tokenizer["pre_tokenizer"] = serde_json::json!({
                "type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always",
                "split": false
            });
        });
        assert!(
            err.starts_with("tokenizer.json has the pre-tokenizer {"),
            "{err}"
        );
        assert!(err.ends_with(tail), "{err}");
        let err = refused("no-byte-fallback", |tokenizer| {
            tokenizer["model"]["byte_fallback"] = false.into();
        });
        assert!(err.ends_with(tail), "{err}");
        let err = refused("no-prepend", |tokenizer| {
            let steps = tokenizer["normalizer"]["normalizers"]
                .as_array_mut()
                .unwrap();
            steps.remove(0);
        });
        assert!(err.ends_with(tail), "{err}");
    }
}

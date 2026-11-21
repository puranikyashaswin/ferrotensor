//! Tokenizer for LFM2.5 models
//!
//! Loads and uses the tokenizer.json from the model directory.

use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer as _};

/// Wraps a Hugging Face `tokenizer.json` with special-token ids.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Token string to id.
    pub vocab: HashMap<String, u32>,
    /// Id to token string.
    pub id_to_token: HashMap<u32, String>,
    /// Beginning-of-sequence token id: the `bos_token` of the
    /// `tokenizer_config.json` beside `tokenizer.json` when it names one,
    /// else `<|im_start|>`, else `<s>`, else 1.
    pub bos_token_id: u32,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Underlying Hugging Face tokenizer; `None` only for a `tokenizer.json`
    /// that is a plain `model.vocab` map with no model `type`, which encodes
    /// with the whitespace and character reader of [`Tokenizer::encode`].
    pub hf_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
}

/// Largest `tokenizer.json` the loader will read (real ones are well under 100 MB).
const MAX_TOKENIZER_FILE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
}

#[derive(Deserialize)]
struct TokenizerModel {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    vocab: HashMap<String, u32>,
}

impl Tokenizer {
    /// Load tokenizer from tokenizer.json
    ///
    /// Fails, with a message naming the file, if it or the
    /// `tokenizer_config.json` beside it is not a regular file, is too large
    /// or is not valid, or if the vocabulary is empty.
    /// A file the Hugging Face reader refuses is read as a plain `model.vocab`
    /// map only when its model names no `type`; a typed model the Hugging Face
    /// reader refuses is an error, not a silent switch to the plain reader.
    /// In the plain reader two tokens sharing one id are rejected; gaps
    /// between ids are fine.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let file = crate::open_regular_file(path, Some(MAX_TOKENIZER_FILE_BYTES))
            .map_err(|e| format!("Failed to read tokenizer.json: {e}"))?;
        let declared_bos = declared_bos_token(path)?;

        let hf_error = match tokenizers::Tokenizer::from_file(path) {
            Ok(hf_tokenizer) => {
                drop(file);
                return Self::from_hf_tokenizer(path, hf_tokenizer, declared_bos.as_deref());
            }
            Err(e) => e,
        };

        let mut content = String::new();
        std::io::Read::read_to_string(&mut std::io::BufReader::new(file), &mut content)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        // Newer files (Qwen3) store each merge as a `["left", "right"]` pair,
        // which the pinned `tokenizers` reader refuses; read them as "left right".
        if let Some(hf_tokenizer) = read_with_pair_merges(&content) {
            return Self::from_hf_tokenizer(path, hf_tokenizer, declared_bos.as_deref());
        }

        let parsed: TokenizerJson = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;
        if let Some(kind) = parsed.model.kind {
            return Err(format!(
                "{}: cannot load the {kind} tokenizer: {hf_error}",
                path.display()
            ));
        }

        let vocab = parsed.model.vocab;
        if vocab.is_empty() {
            return Err(format!("{}: tokenizer vocabulary is empty", path.display()));
        }
        let mut id_to_token: HashMap<u32, String> = HashMap::with_capacity(vocab.len());
        for (token, &id) in &vocab {
            if id_to_token.insert(id, token.clone()).is_some() {
                return Err(format!(
                    "{}: token id {id} is used by more than one token",
                    path.display()
                ));
            }
        }

        let bos_token_id = *declared_bos
            .as_deref()
            .and_then(|token| vocab.get(token))
            .or_else(|| vocab.get("<|im_start|>"))
            .or_else(|| vocab.get("<s>"))
            .unwrap_or(&1);
        let eos_token_id = *vocab
            .get("<|im_end|>")
            .or_else(|| vocab.get("</s>"))
            .unwrap_or(&2);
        let pad_token_id = *vocab.get("<pad>").unwrap_or(&0);

        Ok(Self {
            vocab,
            id_to_token,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            hf_tokenizer: None,
        })
    }

    /// The tokenizer around a parsed Hugging Face tokenizer, with the special
    /// ids found by name.
    fn from_hf_tokenizer(
        path: &Path,
        hf_tokenizer: tokenizers::Tokenizer,
        declared_bos: Option<&str>,
    ) -> Result<Self, String> {
        let vocab = hf_tokenizer.get_vocab(true);
        if vocab.is_empty() {
            return Err(format!("{}: tokenizer vocabulary is empty", path.display()));
        }
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bos_token_id = declared_bos
            .and_then(|token| hf_tokenizer.token_to_id(token))
            .or_else(|| hf_tokenizer.token_to_id("<|im_start|>"))
            .or_else(|| hf_tokenizer.token_to_id("<s>"))
            .unwrap_or(1);
        let eos_token_id = hf_tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| hf_tokenizer.token_to_id("</s>"))
            .unwrap_or(2);
        let pad_token_id = hf_tokenizer.token_to_id("<pad>").unwrap_or(0);

        Ok(Self {
            vocab,
            id_to_token,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            hf_tokenizer: Some(Arc::new(hf_tokenizer)),
        })
    }

    /// Encode text to token IDs, adding the tokenizer's special tokens
    /// (e.g. a leading BOS) when it defines any.
    ///
    /// # Errors
    ///
    /// If the Hugging Face tokenizer fails to encode the text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_with(text, true)
    }

    /// Encode text without adding special tokens; use this for text that
    /// continues a sequence (an SFT response after its prompt).
    ///
    /// # Errors
    ///
    /// If the Hugging Face tokenizer fails to encode the text.
    pub fn encode_continuation(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_with(text, false)
    }

    fn encode_with(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, String> {
        if let Some(tokenizer) = &self.hf_tokenizer {
            return tokenizer
                .encode(text, add_special_tokens)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| format!("tokenizer cannot encode the text: {e}"));
        }

        let mut tokens = if add_special_tokens {
            vec![self.bos_token_id]
        } else {
            Vec::new()
        };

        // A plain `model.vocab` map has no merges: whole words, then the
        // word after a space, then single characters, then `<unk>`.
        for word in text.split_whitespace() {
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else if let Some(&id) = self.vocab.get(&format!("Ġ{word}")) {
                tokens.push(id);
            } else {
                for c in word.chars() {
                    if let Some(&id) = self.vocab.get(&c.to_string()) {
                        tokens.push(id);
                    } else {
                        tokens.push(self.vocab.get("<unk>").copied().unwrap_or(0));
                    }
                }
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs back to text.
    ///
    /// # Errors
    ///
    /// If the Hugging Face tokenizer fails to decode the ids.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        if let Some(tokenizer) = &self.hf_tokenizer {
            return tokenizer
                .decode(tokens, true)
                .map_err(|e| format!("tokenizer cannot decode {} ids: {e}", tokens.len()));
        }

        let mut result = String::new();
        for &id in tokens {
            if id == self.bos_token_id || id == self.eos_token_id || id == self.pad_token_id {
                continue;
            }
            if let Some(token) = self.id_to_token.get(&id) {
                // Handle Ġ prefix (space before word)
                if let Some(word) = token.strip_prefix('Ġ') {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(word);
                } else {
                    result.push_str(token);
                }
            }
        }
        Ok(result)
    }

    /// The ChatML prompt for one user turn (with an optional system turn),
    /// ending with the assistant header so the model continues as the
    /// assistant: `<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`.
    pub fn chat_prompt(system: Option<&str>, user: &str) -> String {
        Self::conversation_prompt(system, &[user])
    }

    /// The ChatML prompt for a conversation: an optional system turn, then
    /// `turns` alternating user and assistant (user first), then the
    /// assistant header.
    pub fn conversation_prompt(system: Option<&str>, turns: &[impl AsRef<str>]) -> String {
        let roles = ["user", "assistant"].into_iter().cycle();
        let turns = roles.zip(turns.iter().map(AsRef::<str>::as_ref));
        let mut prompt = String::new();
        for (role, text) in system.map(|sys| ("system", sys)).into_iter().chain(turns) {
            prompt.push_str(&format!("<|im_start|>{role}\n{text}<|im_end|>\n"));
        }
        prompt.push_str("<|im_start|>assistant\n");
        prompt
    }

    /// Encode [`chat_prompt`](Self::chat_prompt).
    ///
    /// # Errors
    ///
    /// If the Hugging Face tokenizer fails to encode the prompt.
    pub fn encode_chat(&self, system: Option<&str>, user: &str) -> Result<Vec<u32>, String> {
        self.encode_conversation(system, &[user])
    }

    /// Encode [`conversation_prompt`](Self::conversation_prompt).
    ///
    /// # Errors
    ///
    /// If the Hugging Face tokenizer fails to encode the prompt.
    pub fn encode_conversation(
        &self,
        system: Option<&str>,
        turns: &[impl AsRef<str>],
    ) -> Result<Vec<u32>, String> {
        self.encode(&Self::conversation_prompt(system, turns))
    }
}

/// Id of byte 0 in [`byte_level_tokenizer_json`]: byte `b` is id `b + 3`,
/// after `<pad>` 0, `<s>` 1 and `</s>` 2.
pub const BYTE_LEVEL_FIRST_BYTE_ID: u32 = 3;

/// Number of tokens in [`byte_level_tokenizer_json`].
pub const BYTE_LEVEL_VOCAB_SIZE: usize = 259;

/// A Hugging Face `tokenizer.json` that encodes text as its UTF-8 bytes and
/// decodes them back exactly.
///
/// It is a BPE model without merges over `<pad>` 0, `<s>` 1, `</s>` 2 and
/// the 256 characters the `ByteLevel` pre-tokenizer stands bytes in for,
/// byte `b` at id `b + 3`; [`Tokenizer::from_file`] finds the three special
/// tokens at those ids. They are plain vocabulary entries, not added tokens,
/// so text that contains `<s>` still encodes as its bytes.
pub fn byte_level_tokenizer_json() -> serde_json::Value {
    byte_level_bpe_json(byte_level_tokens(), Vec::new())
}

/// A byte-level BPE `tokenizer.json` of `vocab_size` tokens learned from
/// `corpus`; the same corpus and size always give the same file.
///
/// It extends [`byte_level_tokenizer_json`] with the same ids for the three
/// special tokens and the 256 bytes, then one token per merge of the most
/// frequent adjacent pair, a tie going to the pair of lowest ids. The words
/// are the pieces the `ByteLevel` pre-tokenizer cuts each line into, as at
/// encoding, so any text encodes and decodes back exactly.
///
/// # Errors
///
/// If the corpus runs out of pairs to merge before `vocab_size` tokens.
pub fn train_byte_level_bpe(corpus: &str, vocab_size: usize) -> Result<serde_json::Value, String> {
    let (mut words, counts) = byte_level_words(corpus)?;
    let mut pair_counts: HashMap<(u32, u32), u64> = HashMap::new();
    let mut pair_words: HashMap<(u32, u32), BTreeSet<usize>> = HashMap::new();
    for (index, (word, &count)) in words.iter().zip(&counts).enumerate() {
        for pair in word.windows(2).map(|w| (w[0], w[1])) {
            *pair_counts.entry(pair).or_default() += count;
            pair_words.entry(pair).or_default().insert(index);
        }
    }
    // Entries go stale as counts change; a popped entry whose count is not
    // the pair's current one is pushed back with the current count.
    let mut queue: BinaryHeap<(u64, Reverse<(u32, u32)>)> = pair_counts
        .iter()
        .map(|(&pair, &count)| (count, Reverse(pair)))
        .collect();
    let mut tokens = byte_level_tokens();
    let mut ids: HashMap<String, u32> = tokens.iter().cloned().zip(0..).collect();
    let mut merges = Vec::new();
    while tokens.len() < vocab_size {
        let Some((count, Reverse(pair))) = queue.pop() else {
            return Err(format!(
                "the corpus yields {} tokens, fewer than {vocab_size}",
                tokens.len()
            ));
        };
        let current = pair_counts.get(&pair).copied().unwrap_or(0);
        if current == 0 {
            continue;
        }
        if count != current {
            queue.push((current, Reverse(pair)));
            continue;
        }
        let (left, right) = (&tokens[pair.0 as usize], &tokens[pair.1 as usize]);
        merges.push(format!("{left} {right}"));
        let merged_token = format!("{left}{right}");
        let merged = match ids.get(&merged_token) {
            Some(&id) => id,
            None => {
                let id = tokens.len() as u32;
                ids.insert(merged_token.clone(), id);
                tokens.push(merged_token);
                id
            }
        };
        let mut grown = HashSet::new();
        for index in pair_words.remove(&pair).into_iter().flatten() {
            let word = &mut words[index];
            if !word.windows(2).any(|w| (w[0], w[1]) == pair) {
                continue;
            }
            for old in word.windows(2) {
                if let Some(old_count) = pair_counts.get_mut(&(old[0], old[1])) {
                    *old_count -= counts[index];
                }
            }
            *word = merge_pair(word, pair, merged);
            for new in word.windows(2).map(|w| (w[0], w[1])) {
                *pair_counts.entry(new).or_default() += counts[index];
                pair_words.entry(new).or_default().insert(index);
                grown.insert(new);
            }
        }
        queue.extend(
            grown
                .into_iter()
                .map(|new| (pair_counts[&new], Reverse(new)))
                .filter(|&(count, _)| count > 0),
        );
    }
    Ok(byte_level_bpe_json(tokens, merges))
}

/// The distinct words of `corpus` as byte token ids, in byte order, and how
/// often each occurs.
fn byte_level_words(corpus: &str) -> Result<(Vec<Vec<u32>>, Vec<u64>), String> {
    let pre_tokenizer = ByteLevel::new(false, true, true);
    let mut word_counts: BTreeMap<&[u8], u64> = BTreeMap::new();
    for line in corpus.split_inclusive('\n') {
        let mut pieces = PreTokenizedString::from(line);
        pre_tokenizer
            .pre_tokenize(&mut pieces)
            .map_err(|e| format!("cannot pre-tokenize the corpus: {e}"))?;
        for (_, (start, end), _) in pieces.get_splits(OffsetReferential::Original, OffsetType::Byte)
        {
            *word_counts.entry(&line.as_bytes()[start..end]).or_default() += 1;
        }
    }
    Ok(word_counts
        .into_iter()
        .map(|(word, count)| {
            let ids = word
                .iter()
                .map(|&b| u32::from(b) + BYTE_LEVEL_FIRST_BYTE_ID);
            (ids.collect(), count)
        })
        .unzip())
}

/// `word` with each occurrence of `pair`, from the left, replaced by `merged`.
fn merge_pair(word: &[u32], pair: (u32, u32), merged: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(word.len());
    let mut i = 0;
    while i < word.len() {
        if word.get(i + 1).is_some_and(|&next| (word[i], next) == pair) {
            out.push(merged);
            i += 2;
        } else {
            out.push(word[i]);
            i += 1;
        }
    }
    out
}

/// `<pad>`, `<s>`, `</s>`, then the character standing for each byte.
fn byte_level_tokens() -> Vec<String> {
    ["<pad>", "<s>", "</s>"]
        .into_iter()
        .map(str::to_owned)
        .chain(byte_level_chars().iter().map(char::to_string))
        .collect()
}

/// A byte-level BPE `tokenizer.json` whose token `i` is `tokens[i]`, with
/// `merges` ("left right") in rank order.
fn byte_level_bpe_json(tokens: Vec<String>, merges: Vec<String>) -> serde_json::Value {
    let vocab: serde_json::Map<String, serde_json::Value> = tokens
        .into_iter()
        .zip(0u32..)
        .map(|(token, id)| (token, id.into()))
        .collect();
    let byte_level = serde_json::json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true
    });
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": byte_level,
        "post_processor": null,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges
        }
    })
}

/// The character GPT-2's byte-level BPE, and the `tokenizers` `ByteLevel`
/// pre-tokenizer, uses for each byte: printable Latin-1 bytes stand for
/// themselves, the other 68 take U+0100 onward in byte order.
fn byte_level_chars() -> [char; 256] {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut chars = ['\0'; 256];
    for b in (0..=255u8).filter(|&b| printable(b)) {
        chars[usize::from(b)] = char::from(b);
    }
    for (b, c) in (0..=255u8).filter(|&b| !printable(b)).zip('\u{100}'..) {
        chars[usize::from(b)] = c;
    }
    chars
}

/// The Hugging Face tokenizer of `content` with each `["left", "right"]` merge
/// rewritten as `"left right"`, or `None` when it has no pair merges or is still refused.
///
/// The `None` cases leave the caller to report the error of the first read.
fn read_with_pair_merges(content: &str) -> Option<tokenizers::Tokenizer> {
    let Ok(mut json) = serde_json::from_str::<serde_json::Value>(content) else {
        return None;
    };
    let merges = json.get_mut("model")?.get_mut("merges")?.as_array_mut()?;
    let mut rewrote = false;
    for merge in merges.iter_mut() {
        if let Some([left, right]) = merge.as_array().map(Vec::as_slice) {
            let joined = format!("{} {}", left.as_str()?, right.as_str()?);
            *merge = serde_json::Value::String(joined);
            rewrote = true;
        }
    }
    if !rewrote {
        return None;
    }
    let Ok(tokenizer) = json.to_string().parse::<tokenizers::Tokenizer>() else {
        return None;
    };
    Some(tokenizer)
}

/// The `bos_token` that the `tokenizer_config.json` beside `tokenizer_path`
/// declares, as a string or as an added-token object; `None` when the file
/// is absent or declares none (`null` or no key).
fn declared_bos_token(tokenizer_path: &Path) -> Result<Option<String>, String> {
    let config_path = tokenizer_path.with_file_name("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let file = crate::open_regular_file(&config_path, Some(MAX_TOKENIZER_FILE_BYTES))
        .map_err(|e| format!("Failed to read tokenizer_config.json: {e}"))?;
    let config: serde_json::Value = serde_json::from_reader(std::io::BufReader::new(file))
        .map_err(|e| format!("Failed to parse {}: {e}", config_path.display()))?;
    let bos = &config["bos_token"];
    Ok(bos
        .as_str()
        .or_else(|| bos["content"].as_str())
        .map(str::to_owned))
}

/// Next-token sampler: temperature, top-k and nucleus (top-p) filtering.
///
/// `temperature <= 0` means greedy decoding. `top_k == 0` and
/// `top_p >= 1` disable the respective filters.
pub struct Sampler {
    /// Softmax temperature.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens (0 = all).
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probability reaches `top_p`.
    pub top_p: f32,
    rng: rand::rngs::StdRng,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new(0.7, 0, 0.9, None)
    }
}

impl Sampler {
    /// A sampler; `seed` makes the draws reproducible.
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: Option<u64>) -> Self {
        use rand::SeedableRng;
        let rng = match seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        Self {
            temperature,
            top_k,
            top_p,
            rng,
        }
    }

    /// Whether [`sample`](Self::sample) returns the [`argmax`] of the logits.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }

    /// Pick a token id from one row of logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        use rand::Rng;
        if logits.is_empty() {
            return 0;
        }
        if self.is_greedy() {
            return argmax(logits);
        }

        // Candidates sorted by logit, highest first; NaNs sort last.
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if self.top_k > 0 {
            order.truncate(self.top_k);
        }

        let max = logits[order[0]];
        let mut probs: Vec<f32> = order
            .iter()
            .map(|&i| ((logits[i] - max) / self.temperature).exp())
            .collect();
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);

        if self.top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= self.top_p {
                    keep = i + 1;
                    break;
                }
            }
            order.truncate(keep);
            probs.truncate(keep);
        }

        let total: f32 = probs.iter().sum();
        let mut r = self.rng.gen::<f32>() * total;
        for (&idx, &p) in order.iter().zip(&probs) {
            if r < p {
                return idx as u32;
            }
            r -= p;
        }
        order[order.len() - 1] as u32
    }
}

/// Index of the largest logit (NaN-safe).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(0.0, 0, 1.0, Some(1));
        assert!(s.is_greedy());
        assert_eq!(s.sample(&[0.1, 3.0, -1.0, 2.9]), 1);
    }

    #[test]
    fn argmax_returns_the_last_of_equal_maxima() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 3.0, 0.0]), 3);
        assert_eq!(argmax(&[-0.0, 0.0, -0.0]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(1.5, 1, 1.0, Some(7));
        assert!(s.is_greedy());
        assert!(!Sampler::new(1.5, 2, 1.0, None).is_greedy());
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.0, 0.5, 0.4]), 1);
        }
    }

    #[test]
    fn top_k_never_returns_tokens_outside_the_k_best() {
        let mut s = Sampler::new(5.0, 2, 1.0, Some(3));
        for _ in 0..200 {
            let t = s.sample(&[1.0, 4.0, 3.9, 0.0, -2.0]);
            assert!(t == 1 || t == 2, "sampled {t}");
        }
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // Token 0 holds almost all of the mass, so top_p=0.5 keeps only it.
        let mut s = Sampler::new(1.0, 0, 0.5, Some(9));
        for _ in 0..50 {
            assert_eq!(s.sample(&[10.0, 0.0, 0.0, 0.0]), 0);
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let logits = [0.3, 0.2, 0.1, 0.25, 0.15];
        let mut a = Sampler::new(1.0, 0, 1.0, Some(42));
        let mut b = Sampler::new(1.0, 0, 1.0, Some(42));
        let xs: Vec<u32> = (0..32).map(|_| a.sample(&logits)).collect();
        let ys: Vec<u32> = (0..32).map(|_| b.sample(&logits)).collect();
        assert_eq!(xs, ys);
    }

    #[test]
    fn chat_prompt_matches_chatml() {
        assert_eq!(
            Tokenizer::chat_prompt(Some("Be brief."), "Hi"),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// A word-level tokenizer with ChatML special tokens: `<|im_start|>` 1,
    /// `<|im_end|>` 2, `system` 3, `user` 4, `assistant` 5, `Hi` 6,
    /// `Hello` 7, `Bye` 8, `Be` 9, `brief` 10, `.` 11.
    fn chatml_tokenizer() -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
        let words = [
            "<unk>",
            "<|im_start|>",
            "<|im_end|>",
            "system",
            "user",
            "assistant",
            "Hi",
            "Hello",
            "Bye",
            "Be",
            "brief",
            ".",
        ];
        let vocab: HashMap<String, u32> =
            (0u32..).zip(words).map(|(id, w)| (w.into(), id)).collect();
        let special = |id: u32| {
            serde_json::json!({
                "id": id, "content": words[id as usize], "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        };
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [special(1), special(2)],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" }
        });
        let hf: tokenizers::Tokenizer = json.to_string().parse()?;
        Ok(Tokenizer {
            id_to_token: vocab.iter().map(|(w, &id)| (id, w.clone())).collect(),
            vocab,
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: 0,
            hf_tokenizer: Some(Arc::new(hf)),
        })
    }

    type TestResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Writes `tokenizer` as `tokenizer.json`, and `config` as
    /// `tokenizer_config.json` when given, into a fresh temp dir, then loads it.
    fn load_with_config(
        tag: &str,
        tokenizer: &serde_json::Value,
        config: Option<&str>,
    ) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
        let dir = std::env::temp_dir().join(format!(
            "ferrotensor-tokenizer-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string())?;
        if let Some(config) = config {
            std::fs::write(dir.join("tokenizer_config.json"), config)?;
        }
        let loaded = Tokenizer::from_file(dir.join("tokenizer.json"));
        std::fs::remove_dir_all(&dir)?;
        loaded.map_err(Into::into)
    }

    /// A word-level Hugging Face tokenizer: `<|endoftext|>` 0,
    /// `<|im_start|>` 1, `<|im_end|>` 2, `Hi` 3.
    fn endoftext_chatml_json() -> serde_json::Value {
        let vocab =
            serde_json::json!({"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "Hi": 3});
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" },
            "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<|endoftext|>" }
        })
    }

    #[test]
    fn declared_bos_token_replaces_the_chatml_guess() -> TestResult {
        let json = endoftext_chatml_json();
        let guessed = load_with_config("bos-none", &json, None)?;
        assert!(guessed.hf_tokenizer.is_some());
        assert_eq!(guessed.bos_token_id, 1, "no config: <|im_start|> as before");
        for (tag, config, bos) in [
            ("bos-string", r#"{"bos_token": "<|endoftext|>"}"#, 0),
            (
                "bos-object",
                r#"{"bos_token": {"content": "<|endoftext|>", "special": true}}"#,
                0,
            ),
            ("bos-null", r#"{"bos_token": null}"#, 1),
            ("bos-unknown", r#"{"bos_token": "<s>"}"#, 1),
        ] {
            let tok = load_with_config(tag, &json, Some(config))?;
            assert_eq!(tok.bos_token_id, bos, "{config}");
            assert_eq!(tok.eos_token_id, 2, "{config}");
            assert_eq!(tok.pad_token_id, 0, "{config}");
            // The Hugging Face encoder never reads bos_token_id.
            assert_eq!(tok.encode("Hi Hi")?, guessed.encode("Hi Hi")?, "{config}");
            assert_eq!(tok.encode("Hi Hi")?, [3, 3], "{config}");
        }
        Ok(())
    }

    #[test]
    fn fallback_reader_uses_the_declared_bos_token() -> TestResult {
        let json =
            serde_json::json!({ "model": { "vocab": endoftext_chatml_json()["model"]["vocab"] } });
        let guessed = load_with_config("fallback-none", &json, None)?;
        assert!(guessed.hf_tokenizer.is_none());
        assert_eq!(guessed.encode("Hi")?, [1, 3]);
        let declared = load_with_config(
            "fallback-declared",
            &json,
            Some(r#"{"bos_token": "<|endoftext|>"}"#),
        )?;
        assert_eq!(declared.encode("Hi")?, [0, 3]);
        assert_eq!(declared.decode(&[1, 3, 2])?, "<|im_start|>Hi");
        Ok(())
    }

    #[test]
    fn byte_level_tokenizer_loads_with_the_special_ids_the_loader_expects() -> TestResult {
        let tok = load_with_config("byte-level", &byte_level_tokenizer_json(), None)?;
        assert!(
            tok.hf_tokenizer.is_some(),
            "parsed by the Hugging Face reader"
        );
        assert_eq!(tok.vocab.len(), BYTE_LEVEL_VOCAB_SIZE);
        assert_eq!(
            (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id),
            (0, 1, 2)
        );
        // By hand: 'H' 72, 'i' 105, ' ' 32, '\n' 10, 'é' is C3 A9; each plus 3.
        assert_eq!(tok.encode("Hi \n")?, [75, 108, 35, 13]);
        assert_eq!(tok.encode("é")?, [0xC3 + 3, 0xA9 + 3]);
        assert_eq!(tok.vocab["Ġ"], 35, "GPT-2's character for the space byte");
        assert_eq!(
            tok.encode("<s></s>")?.len(),
            7,
            "specials in text stay bytes"
        );
        Ok(())
    }

    #[test]
    fn byte_level_tokenizer_maps_every_byte_and_round_trips() -> TestResult {
        let tok = load_with_config("byte-round-trip", &byte_level_tokenizer_json(), None)?;
        // U+0000..=U+007F are the single bytes 0..=127; U+0080..=U+00FF are
        // two bytes, C2 or C3 then 80..=BF, so together every byte occurs.
        for scalar in 0u32..=0xFF {
            let text = char::from_u32(scalar).ok_or("scalar")?.to_string();
            let expected: Vec<u32> = text.bytes().map(|b| u32::from(b) + 3).collect();
            assert_eq!(tok.encode(&text)?, expected, "U+{scalar:04X}");
            assert_eq!(tok.decode(&expected)?, text, "U+{scalar:04X}");
        }
        for text in [
            "plain ASCII.",
            "héllo → 世界\n\ttab",
            "  two spaces\r\n",
            "🦀<pad>",
        ] {
            assert_eq!(tok.decode(&tok.encode(text)?)?, text);
            assert_eq!(tok.encode(text)?.len(), text.len());
        }
        Ok(())
    }

    #[test]
    fn bpe_skips_pairs_a_merge_used_up_and_counts_what_the_corpus_yields() -> TestResult {
        // Words "aaaa" and " aab". Merging "a a" (4 times) turns "aaaa" into
        // "aa aa" and " aab" into " aa b", so (' ', a) and (a, b) drop to 0
        // with their queue entries left behind. The three count-1 pairs then
        // merge in id order (' ' 35, "aa" 259, 'b' 101), and " aa b" uses up
        // (aa, b) on the way. That is four merges, 263 tokens in all.
        let json = train_byte_level_bpe("aaaa aab", 263)?;
        assert_eq!(
            json["model"]["merges"],
            serde_json::json!(["a a", "Ġ aa", "aa aa", "Ġaa b"])
        );
        assert_eq!(
            train_byte_level_bpe("aaaa aab", 264).err().as_deref(),
            Some("the corpus yields 263 tokens, fewer than 264")
        );
        Ok(())
    }

    #[test]
    fn bpe_learns_the_most_frequent_pairs_first() -> TestResult {
        // The words are "ab" once and " ab" twice: (a, b) occurs 3 times and
        // (' ', a) twice, so "a b" merges first, then " " with "ab".
        let json = train_byte_level_bpe("ab ab ab", 261)?;
        assert_eq!(json["model"]["merges"], serde_json::json!(["a b", "Ġ ab"]));
        assert_eq!(json["model"]["vocab"]["ab"], 259);
        assert_eq!(json["model"]["vocab"]["Ġab"], 260);
        let tok = load_with_config("bpe-merges", &json, None)?;
        assert_eq!(tok.encode("ab ab")?, [259, 260]);
        assert_eq!(
            train_byte_level_bpe("ab ab ab", 262).err().as_deref(),
            Some("the corpus yields 261 tokens, fewer than 262")
        );
        Ok(())
    }

    /// Three lines with a tab, a blank line, leading spaces and non-ASCII
    /// UTF-8. Repeated three times, every pair inside "the", " the", " sat"
    /// and " on" occurs 6 or 12 times, so these words become single tokens
    /// within the first 9 merges; most other pairs occur 3 times.
    const BPE_LINES: &str =
        "the cat sat on the mat.\n\tthe dog sat on the log.\n\n  naïve café, 世界\n";

    #[test]
    fn bpe_extends_the_byte_ids_and_round_trips_any_text() -> TestResult {
        let corpus = BPE_LINES.repeat(3);
        let json = train_byte_level_bpe(&corpus, 280)?;
        let tok = load_with_config("bpe-round-trip", &json, None)?;
        assert_eq!(tok.vocab.len(), 280);
        assert_eq!(
            (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id),
            (0, 1, 2)
        );
        for (token, id) in byte_level_tokens().iter().zip(0u32..) {
            assert_eq!(tok.vocab[token], id, "{token:?}");
        }
        let words = ["the", "Ġsat", "Ġon", "Ġthe"].map(|word| tok.vocab[word]);
        assert_eq!(tok.encode("the sat on the")?, words);
        for text in [
            corpus.as_str(),
            "\n",
            "\t\t tabs and  spaces \r\n\n",
            "unseen: 🦀 Ωμέγα <s></s><pad>",
            "\u{0}\u{7f}\u{80}\u{ff}",
        ] {
            assert_eq!(tok.decode(&tok.encode(text)?)?, text, "{text:?}");
        }
        Ok(())
    }

    #[test]
    fn bpe_training_is_deterministic() -> TestResult {
        let corpus = BPE_LINES.repeat(3);
        let first = train_byte_level_bpe(&corpus, 280)?.to_string();
        for _ in 0..3 {
            assert_eq!(train_byte_level_bpe(&corpus, 280)?.to_string(), first);
        }
        Ok(())
    }

    #[test]
    fn broken_tokenizer_config_is_an_error() {
        let err = load_with_config("config-broken", &endoftext_chatml_json(), Some("{"))
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|e| e.contains("tokenizer_config.json")),
            "{err:?}"
        );
    }

    #[test]
    fn a_failing_encoder_is_an_error_not_the_plain_reader() -> TestResult {
        // A WordLevel model whose `unk_token` is missing from its vocabulary
        // fails on any word it does not know.
        let json = serde_json::json!({
            "model": { "type": "WordLevel", "vocab": {"Hi": 0, "<s>": 1}, "unk_token": "<unk>" }
        });
        let tok = load_with_config("encoder-fails", &json, None)?;
        assert!(tok.hf_tokenizer.is_some());
        assert_eq!(tok.encode_continuation("Hi")?, [0]);
        for err in [tok.encode("Bye"), tok.encode_continuation("Bye")] {
            let err = err.err().ok_or("an unknown word encoded")?;
            assert!(
                err.starts_with("tokenizer cannot encode the text: "),
                "{err}"
            );
            assert!(err.contains("[UNK]"), "{err}");
        }
        let chat = tok.encode_chat(None, "Hi").err().ok_or("chat encoded")?;
        assert!(chat.starts_with("tokenizer cannot encode"), "{chat}");
        Ok(())
    }

    #[test]
    fn a_typed_model_the_hugging_face_reader_refuses_is_an_error() {
        // A vocabulary the plain reader could use, under a model type the
        // Hugging Face reader does not know.
        let json = serde_json::json!({ "model": { "type": "NoSuchModel", "vocab": {"Hi": 0} } });
        let err = load_with_config("typed-refused", &json, None)
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref().is_some_and(|e| e.contains("tokenizer.json")
                && e.contains("cannot load the NoSuchModel tokenizer")),
            "{err:?}"
        );
    }

    /// A byte-level BPE `tokenizer.json` laid out as Qwen3's is: an NFC
    /// normalizer, a regex split then `ByteLevel` pre-tokenizer, three added
    /// special tokens outside `model.vocab`, and `merges` as `["left", "right"]`
    /// pairs when `pairs` is set, else as "left right" strings (the Qwen2.5
    /// layout). Ids: the 256 bytes 0..=255, `hi` 256, `Ġhi` 257, the first two
    /// bytes of U+4E16 258, all three 259; `<|endoftext|>` 260,
    /// `<|im_start|>` 261, `<|im_end|>` 262.
    fn qwen3_style_json(pairs: bool) -> serde_json::Value {
        let chars = byte_level_chars();
        let byte = |b: usize| chars[b].to_string();
        let first_two = format!("{}{}", byte(0xE4), byte(0xB8));
        let all_three = format!("{first_two}{}", byte(0x96));
        let merges = [
            ("h".to_owned(), "i".to_owned()),
            ("Ġ".to_owned(), "hi".to_owned()),
            (byte(0xE4), byte(0xB8)),
            (first_two.clone(), byte(0x96)),
        ];
        let merges: Vec<serde_json::Value> = merges
            .iter()
            .map(|(left, right)| {
                if pairs {
                    serde_json::json!([left, right])
                } else {
                    serde_json::json!(format!("{left} {right}"))
                }
            })
            .collect();
        let tokens: Vec<String> = chars
            .iter()
            .map(char::to_string)
            .chain(["hi".to_owned(), "Ġhi".to_owned(), first_two, all_three])
            .collect();
        let vocab: serde_json::Map<String, serde_json::Value> = tokens
            .into_iter()
            .zip(0u32..)
            .map(|(token, id)| (token, id.into()))
            .collect();
        let special = |id: u32, content: &str| {
            serde_json::json!({
                "id": id, "content": content, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        };
        let byte_level = serde_json::json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
            "use_regex": false
        });
        let split = serde_json::json!({
            "type": "Split",
            "pattern": { "Regex": r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+" },
            "behavior": "Isolated",
            "invert": false
        });
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                special(260, "<|endoftext|>"),
                special(261, "<|im_start|>"),
                special(262, "<|im_end|>")
            ],
            "normalizer": { "type": "NFC" },
            "pre_tokenizer": { "type": "Sequence", "pretokenizers": [split, byte_level] },
            "post_processor": byte_level,
            "decoder": byte_level,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": "", "end_of_word_suffix": "",
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": merges
            }
        })
    }

    #[test]
    fn merges_as_pairs_load_and_round_trip() -> TestResult {
        let tok = load_with_config("pair-merges", &qwen3_style_json(true), None)?;
        assert!(
            tok.hf_tokenizer.is_some(),
            "read by the Hugging Face reader"
        );
        assert_eq!(
            tok.vocab.len(),
            263,
            "260 vocabulary entries, 3 added tokens"
        );
        assert_eq!((tok.bos_token_id, tok.eos_token_id), (261, 262));
        // By hand: 'H' 72, 'i' 105, " hi" is one merged token, '世' is E4 B8 96
        // and '界' is E7 95 8C, 'é' is C3 A9, '\n' 10.
        assert_eq!(tok.encode("Hi hi")?, [72, 105, 257]);
        assert_eq!(tok.encode("hi")?, [256]);
        assert_eq!(tok.encode("世界")?, [259, 0xE7, 0x95, 0x8C]);
        assert_eq!(tok.encode("é")?, [0xC3, 0xA9]);
        assert_eq!(
            tok.encode("<|im_start|>user\n")?,
            [261, 117, 115, 101, 114, 10]
        );
        assert_eq!(tok.encode("a<|im_end|>")?, [97, 262]);
        for text in [
            "Hi hi",
            "世界 hello 你好",
            "héllo → 🦀\n\ttab",
            "  two  spaces\r\n",
        ] {
            assert_eq!(tok.decode(&tok.encode(text)?)?, text);
        }
        // Special tokens are in the ids and dropped on decode.
        let chat = tok.encode("<|im_start|>你好<|im_end|>")?;
        assert_eq!((chat.first(), chat.last()), (Some(&261), Some(&262)));
        assert_eq!(tok.decode(&chat)?, "你好");
        Ok(())
    }

    #[test]
    fn merges_as_strings_still_load_and_match_the_pair_form() -> TestResult {
        let strings = load_with_config("string-merges", &qwen3_style_json(false), None)?;
        let pairs = load_with_config("pair-merges-match", &qwen3_style_json(true), None)?;
        assert!(strings.hf_tokenizer.is_some());
        assert_eq!(strings.vocab, pairs.vocab);
        assert_eq!(strings.encode("Hi hi")?, [72, 105, 257]);
        for text in [
            "Hi hi",
            "世界 hello 你好",
            "<|im_start|>user\n你好<|im_end|>",
        ] {
            let ids = strings.encode(text)?;
            assert_eq!(ids, pairs.encode(text)?, "{text}");
            assert_eq!(strings.decode(&ids)?, pairs.decode(&ids)?, "{text}");
        }
        Ok(())
    }

    #[test]
    fn a_merge_pair_that_is_not_two_strings_is_an_error() {
        let mut json = qwen3_style_json(true);
        json["model"]["merges"][0] = serde_json::json!(["h", "i", "x"]);
        let err = load_with_config("bad-pair", &json, None)
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|e| e.contains("cannot load the BPE tokenizer")),
            "{err:?}"
        );
    }

    #[test]
    fn one_turn_conversation_encodes_like_encode_chat(
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tok = chatml_tokenizer()?;
        for system in [None, Some("Be brief.")] {
            assert_eq!(
                tok.encode_conversation(system, &["Hi"])?,
                tok.encode_chat(system, "Hi")?
            );
        }
        assert_eq!(tok.encode_chat(None, "Hi")?, [1, 4, 6, 2, 1, 5]);
        Ok(())
    }

    #[test]
    fn two_turn_conversation_is_the_turns_in_chatml(
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tok = chatml_tokenizer()?;
        let turns = ["Hi", "Hello", "Bye"];
        let user_hi = [1, 4, 6, 2];
        let assistant_hello = [1, 5, 7, 2];
        let user_bye = [1, 4, 8, 2];
        let header = [1, 5];
        let without_system = [&user_hi[..], &assistant_hello, &user_bye, &header].concat();
        assert_eq!(tok.encode_conversation(None, &turns)?, without_system);
        let system = [1, 3, 9, 10, 11, 2];
        assert_eq!(
            tok.encode_conversation(Some("Be brief."), &turns)?,
            [&system[..], &without_system].concat()
        );
        Ok(())
    }
}

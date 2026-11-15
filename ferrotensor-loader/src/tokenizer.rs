//! Tokenizer for LFM2.5 models
//!
//! Loads and uses the tokenizer.json from the model directory.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

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
    /// Underlying Hugging Face tokenizer, if it loaded.
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
    vocab: HashMap<String, u32>,
}

impl Tokenizer {
    /// Load tokenizer from tokenizer.json
    ///
    /// Fails, with a message naming the file, if it or the
    /// `tokenizer_config.json` beside it is not a regular file, is too large
    /// or is not valid, or if the vocabulary is empty.
    /// In the fallback reader (a plain `model.vocab` map) two tokens sharing
    /// one id are rejected; gaps between ids are fine.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let file = crate::open_regular_file(path, Some(MAX_TOKENIZER_FILE_BYTES))
            .map_err(|e| format!("Failed to read tokenizer.json: {e}"))?;
        let declared_bos = declared_bos_token(path)?;

        if let Ok(hf_tokenizer) = tokenizers::Tokenizer::from_file(path) {
            drop(file);
            let vocab = hf_tokenizer.get_vocab(true);
            if vocab.is_empty() {
                return Err(format!("{}: tokenizer vocabulary is empty", path.display()));
            }
            let id_to_token: HashMap<u32, String> =
                vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
            let bos_token_id = declared_bos
                .as_deref()
                .and_then(|token| hf_tokenizer.token_to_id(token))
                .or_else(|| hf_tokenizer.token_to_id("<|im_start|>"))
                .or_else(|| hf_tokenizer.token_to_id("<s>"))
                .unwrap_or(1);
            let eos_token_id = hf_tokenizer
                .token_to_id("<|im_end|>")
                .or_else(|| hf_tokenizer.token_to_id("</s>"))
                .unwrap_or(2);
            let pad_token_id = hf_tokenizer.token_to_id("<pad>").unwrap_or(0);

            return Ok(Self {
                vocab,
                id_to_token,
                bos_token_id,
                eos_token_id,
                pad_token_id,
                hf_tokenizer: Some(Arc::new(hf_tokenizer)),
            });
        }

        let mut content = String::new();
        std::io::Read::read_to_string(&mut std::io::BufReader::new(file), &mut content)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let parsed: TokenizerJson = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

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

    /// Encode text to token IDs, adding the tokenizer's special tokens
    /// (e.g. a leading BOS) when it defines any.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, true)
    }

    /// Encode text without adding special tokens; use this for text that
    /// continues a sequence (an SFT response after its prompt).
    pub fn encode_continuation(&self, text: &str) -> Vec<u32> {
        self.encode_with(text, false)
    }

    fn encode_with(&self, text: &str, add_special_tokens: bool) -> Vec<u32> {
        if let Some(tokenizer) = &self.hf_tokenizer {
            if let Ok(encoding) = tokenizer.encode(text, add_special_tokens) {
                return encoding.get_ids().to_vec();
            }
        }

        let mut tokens = if add_special_tokens {
            vec![self.bos_token_id]
        } else {
            Vec::new()
        };

        // Simple character/byte fallback encoding
        // Real BPE would be more complex, but this works for basic testing
        for word in text.split_whitespace() {
            // Try exact match first
            if let Some(&id) = self.vocab.get(word) {
                tokens.push(id);
            } else {
                // Fallback: encode character by character with Ġ prefix for space
                let word_with_space = format!("Ġ{}", word);
                if let Some(&id) = self.vocab.get(&word_with_space) {
                    tokens.push(id);
                } else {
                    // Character by character
                    for c in word.chars() {
                        if let Some(&id) = self.vocab.get(&c.to_string()) {
                            tokens.push(id);
                        } else {
                            // Unknown token - use a common fallback
                            tokens.push(self.vocab.get("<unk>").copied().unwrap_or(0));
                        }
                    }
                }
            }
        }

        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        if let Some(tokenizer) = &self.hf_tokenizer {
            if let Ok(decoded) = tokenizer.decode(tokens, true) {
                return decoded;
            }
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
        result
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
    pub fn encode_chat(&self, system: Option<&str>, user: &str) -> Vec<u32> {
        self.encode_conversation(system, &[user])
    }

    /// Encode [`conversation_prompt`](Self::conversation_prompt).
    pub fn encode_conversation(&self, system: Option<&str>, turns: &[impl AsRef<str>]) -> Vec<u32> {
        self.encode(&Self::conversation_prompt(system, turns))
    }
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

    /// Pick a token id from one row of logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        use rand::Rng;
        if logits.is_empty() {
            return 0;
        }
        if self.temperature <= 0.0 || self.top_k == 1 {
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
        assert_eq!(s.sample(&[0.1, 3.0, -1.0, 2.9]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(1.5, 1, 1.0, Some(7));
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
            assert_eq!(tok.encode("Hi Hi"), guessed.encode("Hi Hi"), "{config}");
            assert_eq!(tok.encode("Hi Hi"), [3, 3], "{config}");
        }
        Ok(())
    }

    #[test]
    fn fallback_reader_uses_the_declared_bos_token() -> TestResult {
        let json =
            serde_json::json!({ "model": { "vocab": endoftext_chatml_json()["model"]["vocab"] } });
        let guessed = load_with_config("fallback-none", &json, None)?;
        assert!(guessed.hf_tokenizer.is_none());
        assert_eq!(guessed.encode("Hi"), [1, 3]);
        let declared = load_with_config(
            "fallback-declared",
            &json,
            Some(r#"{"bos_token": "<|endoftext|>"}"#),
        )?;
        assert_eq!(declared.encode("Hi"), [0, 3]);
        assert_eq!(declared.decode(&[1, 3, 2]), "<|im_start|>Hi");
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
    fn one_turn_conversation_encodes_like_encode_chat(
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let tok = chatml_tokenizer()?;
        for system in [None, Some("Be brief.")] {
            assert_eq!(
                tok.encode_conversation(system, &["Hi"]),
                tok.encode_chat(system, "Hi")
            );
        }
        assert_eq!(tok.encode_chat(None, "Hi"), [1, 4, 6, 2, 1, 5]);
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
        assert_eq!(tok.encode_conversation(None, &turns), without_system);
        let system = [1, 3, 9, 10, 11, 2];
        assert_eq!(
            tok.encode_conversation(Some("Be brief."), &turns),
            [&system[..], &without_system].concat()
        );
        Ok(())
    }
}

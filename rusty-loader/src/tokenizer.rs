//! Tokenizer for LFM2.5 models
//!
//! Loads and uses the tokenizer.json from the model directory.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Wraps a Hugging Face `tokenizer.json` with special-token ids.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// Token string to id.
    pub vocab: HashMap<String, u32>,
    /// Id to token string.
    pub id_to_token: HashMap<u32, String>,
    /// Beginning-of-sequence token id.
    pub bos_token_id: u32,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Underlying Hugging Face tokenizer, if it loaded.
    pub hf_tokenizer: Option<Arc<tokenizers::Tokenizer>>,
}

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
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        if let Ok(hf_tokenizer) = tokenizers::Tokenizer::from_file(path.as_ref()) {
            let vocab = hf_tokenizer.get_vocab(true);
            let id_to_token: HashMap<u32, String> =
                vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
            let bos_token_id = hf_tokenizer
                .token_to_id("<|im_start|>")
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

        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read tokenizer.json: {}", e))?;

        let parsed: TokenizerJson = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse tokenizer.json: {}", e))?;

        let vocab = parsed.model.vocab;
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        // Find special tokens
        let bos_token_id = *vocab
            .get("<|im_start|>")
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
        let mut prompt = String::new();
        if let Some(sys) = system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(sys);
            prompt.push_str("<|im_end|>\n");
        }
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user);
        prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
        prompt
    }

    /// Encode [`chat_prompt`](Self::chat_prompt).
    pub fn encode_chat(&self, system: Option<&str>, user: &str) -> Vec<u32> {
        self.encode(&Self::chat_prompt(system, user))
    }
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
}

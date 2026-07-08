//! Tokenizer for LFM2.5 models
//!
//! Loads and uses the tokenizer.json from the model directory.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
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

    /// Encode text to token IDs (simple whitespace + subword split)
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(tokenizer) = &self.hf_tokenizer {
            if let Ok(encoding) = tokenizer.encode(text, true) {
                return encoding.get_ids().to_vec();
            }
        }

        let mut tokens = vec![self.bos_token_id];

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
                if token.starts_with('Ġ') {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(&token[2..]); // Skip Ġ (2 bytes)
                } else {
                    result.push_str(token);
                }
            }
        }
        result
    }

    /// Encode with chat template format
    pub fn encode_chat(&self, system: Option<&str>, user: &str) -> Vec<u32> {
        let mut prompt = String::new();

        if let Some(sys) = system {
            prompt.push_str("<|im_start|>system\n");
            prompt.push_str(sys);
            prompt.push_str("\n<|im_end|>\n");
        }

        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(user);
        prompt.push_str("\n<|im_end|>\n<|im_start|>assistant\n");

        self.encode(&prompt)
    }
}

/// Sampling utilities
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

impl Sampler {
    pub fn sample(&self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            // Greedy
            return logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }

        // Temperature scaling
        let scaled: Vec<f32> = logits.iter().map(|x| x / self.temperature).collect();

        // Softmax
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|x| (x - max_val).exp()).sum();
        let probs: Vec<f32> = scaled
            .iter()
            .map(|x| (x - max_val).exp() / exp_sum)
            .collect();

        // Top-p sampling
        let mut sorted_indices: Vec<usize> = (0..probs.len()).collect();
        sorted_indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

        let mut cumsum = 0.0;
        let mut nucleus_indices = Vec::new();
        for &idx in &sorted_indices {
            cumsum += probs[idx];
            nucleus_indices.push(idx);
            if cumsum >= self.top_p {
                break;
            }
        }

        // Renormalize
        let nucleus_sum: f32 = nucleus_indices.iter().map(|&i| probs[i]).sum();
        let normalized: Vec<f32> = nucleus_indices
            .iter()
            .map(|&i| probs[i] / nucleus_sum)
            .collect();

        // Sample
        let r: f32 = rand::random();
        let mut cumsum = 0.0;
        for (i, &p) in normalized.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return nucleus_indices[i] as u32;
            }
        }

        nucleus_indices[0] as u32
    }
}

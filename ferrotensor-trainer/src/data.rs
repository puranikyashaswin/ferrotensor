//! Data Pipeline for Training
//!
//! Provides Dataset trait and DataLoader for efficient training data loading.
//! Supports SFT (response-masked) and pretraining (next-token) sample formats.

use ferrotensor_loader::Tokenizer;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Trait for datasets that provide training samples
pub trait Dataset: Send + Sync {
    /// Number of samples in the dataset
    fn len(&self) -> usize;

    /// Get a single sample by index
    /// Returns (input_ids, target_ids) as token vectors
    fn get(&self, idx: usize) -> (Vec<u32>, Vec<u32>);

    /// Optional per-token loss mask aligned with targets (1 = include, 0 = ignore)
    fn get_loss_mask(&self, idx: usize) -> Option<Vec<u8>> {
        let (_, targets) = self.get(idx);
        Some(vec![1u8; targets.len()])
    }

    /// Check if dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One training example with optional response-only loss mask.
#[derive(Clone, Debug)]
pub struct TrainingExample {
    /// Tokens fed to the model.
    pub input_ids: Vec<u32>,
    /// Next-token targets, aligned with `input_ids`.
    pub target_ids: Vec<u32>,
    /// 1 = compute loss, 0 = ignore. Same length as `target_ids`.
    pub loss_mask: Vec<u8>,
}

impl TrainingExample {
    /// Next-token example over a whole sequence: inputs are all but the last
    /// token, targets all but the first, every target supervised.
    pub fn from_next_token(sequence: Vec<u32>) -> Self {
        if sequence.len() < 2 {
            let len = sequence.len();
            return Self {
                input_ids: sequence.clone(),
                target_ids: sequence,
                loss_mask: vec![1u8; len],
            };
        }
        let input = sequence[..sequence.len() - 1].to_vec();
        let target = sequence[1..].to_vec();
        let loss_mask = vec![1u8; target.len()];
        Self {
            input_ids: input,
            target_ids: target,
            loss_mask,
        }
    }

    /// Build an SFT example where loss is only computed on response tokens.
    ///
    /// `prompt_tokens` are the tokens up to (and including) the assistant prefix,
    /// `response_tokens` are the generated / supervised response ids.
    pub fn from_sft_prompt_response(prompt_tokens: Vec<u32>, response_tokens: Vec<u32>) -> Self {
        let mut full = prompt_tokens.clone();
        full.extend(response_tokens.iter().copied());
        if full.len() < 2 {
            return Self::from_next_token(full);
        }

        let input = full[..full.len() - 1].to_vec();
        let target = full[1..].to_vec();

        // Positions in `target` whose corresponding prediction recovers a response token.
        // The first response token appears at index `prompt_tokens.len()` in `full`,
        // so its target index is `prompt_tokens.len() - 1`.
        let mut loss_mask = vec![0u8; target.len()];
        if !prompt_tokens.is_empty() && !response_tokens.is_empty() {
            let first_response_target = prompt_tokens.len().saturating_sub(1);
            loss_mask[first_response_target..].fill(1);
        } else {
            loss_mask.fill(1);
        }

        Self {
            input_ids: input,
            target_ids: target,
            loss_mask,
        }
    }
}

/// Simple in-memory text dataset for LLM training
#[derive(Clone)]
pub struct TextDataset {
    samples: Vec<TrainingExample>,
}

impl TextDataset {
    /// An empty dataset.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Add a sample to the dataset
    pub fn add_sample(&mut self, input_ids: Vec<u32>, target_ids: Vec<u32>) {
        let loss_mask = vec![1u8; target_ids.len()];
        self.samples.push(TrainingExample {
            input_ids,
            target_ids,
            loss_mask,
        });
    }

    /// Add a prepared example (with its loss mask).
    pub fn add_example(&mut self, example: TrainingExample) {
        self.samples.push(example);
    }

    /// Create from list of token sequences (next-token prediction)
    pub fn from_sequences(sequences: Vec<Vec<u32>>) -> Self {
        let samples = sequences
            .into_iter()
            .map(TrainingExample::from_next_token)
            .collect();
        Self { samples }
    }

    /// Create from a JSONL instruction dataset with `{prompt,response}` objects.
    /// Uses chat-template encoding and response-only loss masks when `use_chat_template`.
    pub fn from_jsonl_sft<P: AsRef<Path>>(
        path: P,
        tokenizer: &Tokenizer,
        use_chat_template: bool,
        max_len: usize,
    ) -> Result<Self, std::io::Error> {
        #[derive(Deserialize)]
        struct JsonlSample {
            prompt: String,
            response: String,
            #[serde(default)]
            system: Option<String>,
        }

        let content = fs::read_to_string(path)?;
        let mut samples = Vec::new();
        for (line_no, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let sample = serde_json::from_str::<JsonlSample>(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "line {}: expected {{\"prompt\": ..., \"response\": ...}}: {e}",
                        line_no + 1
                    ),
                )
            })?;
            if let Some(example) = build_sft_example(
                tokenizer,
                &sample.prompt,
                &sample.response,
                sample.system.as_deref(),
                use_chat_template,
                max_len,
            ) {
                samples.push(example);
            }
        }
        Ok(Self { samples })
    }

    /// Create from a JSONL instruction dataset with `{prompt,response}` objects.
    pub fn from_jsonl<P: AsRef<Path>>(
        path: P,
        tokenizer: &Tokenizer,
    ) -> Result<Self, std::io::Error> {
        Self::from_jsonl_sft(path, tokenizer, true, usize::MAX)
    }

    /// Create from a plain text file, one training sample per non-empty line.
    pub fn from_text_file<P: AsRef<Path>>(
        path: P,
        tokenizer: &Tokenizer,
    ) -> Result<Self, std::io::Error> {
        let content = fs::read_to_string(path)?;
        let sequences = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| tokenizer.encode(line))
            .collect();
        Ok(Self::from_sequences(sequences))
    }

    /// Create from a flat binary file of little-endian u32 token ids.
    pub fn from_token_bin<P: AsRef<Path>>(
        path: P,
        sequence_length: usize,
    ) -> Result<Self, std::io::Error> {
        let bytes = fs::read(path)?;
        let mut tokens = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            tokens.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Self::from_packed_tokens(tokens, sequence_length))
    }

    /// Create fixed-length training windows from a packed token stream.
    pub fn from_packed_tokens(tokens: Vec<u32>, sequence_length: usize) -> Self {
        if sequence_length < 2 || tokens.len() < sequence_length {
            return Self::new();
        }

        let mut samples = Vec::new();
        let stride = sequence_length.saturating_sub(1).max(1);
        let mut start = 0usize;
        while start + sequence_length <= tokens.len() {
            let seq = tokens[start..start + sequence_length].to_vec();
            samples.push(TrainingExample::from_next_token(seq));
            start += stride;
        }

        Self { samples }
    }

    /// Greedily pack short sequences into longer windows up to `max_len`.
    pub fn pack_sequences(sequences: Vec<Vec<u32>>, max_len: usize) -> Self {
        if max_len < 2 {
            return Self::new();
        }

        let mut packed = Vec::new();
        let mut current = Vec::new();
        for seq in sequences {
            if seq.is_empty() {
                continue;
            }
            if current.len() + seq.len() > max_len && current.len() >= 2 {
                packed.push(current);
                current = Vec::new();
            }
            current.extend(seq);
        }
        if current.len() >= 2 {
            packed.push(current);
        }

        Self::from_sequences(packed)
    }

    /// All examples, in insertion order.
    pub fn examples(&self) -> &[TrainingExample] {
        &self.samples
    }
}

/// Tokenize one SFT pair into a response-masked example of at most
/// `max_len` input tokens.
///
/// With `use_chat_template` the prompt is wrapped in the ChatML template
/// and the response is followed by the end-of-turn token, so the model also
/// learns where to stop. The response is tokenized without special tokens.
/// Long examples are truncated at the end (the prompt is kept whole);
/// examples whose prompt leaves no room for a response token are dropped.
pub fn build_sft_example(
    tokenizer: &Tokenizer,
    prompt: &str,
    response: &str,
    system: Option<&str>,
    use_chat_template: bool,
    max_len: usize,
) -> Option<TrainingExample> {
    let (prompt_tokens, response_tokens) = if use_chat_template {
        let mut response_part = tokenizer.encode_continuation(response);
        response_part.push(tokenizer.eos_token_id);
        (tokenizer.encode_chat(system, prompt), response_part)
    } else {
        (
            tokenizer.encode(prompt),
            tokenizer.encode_continuation(response),
        )
    };

    if prompt_tokens.is_empty() || response_tokens.is_empty() {
        return None;
    }
    // The first supervised target sits at index prompt_len - 1.
    if max_len < 2 || prompt_tokens.len() > max_len {
        return None;
    }

    let mut example = TrainingExample::from_sft_prompt_response(prompt_tokens, response_tokens);
    if example.input_ids.len() > max_len {
        example.input_ids.truncate(max_len);
        example.target_ids.truncate(max_len);
        example.loss_mask.truncate(max_len);
    }
    if example.input_ids.len() < 2 || !example.loss_mask.contains(&1) {
        return None;
    }
    Some(example)
}

impl Default for TextDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset for TextDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, idx: usize) -> (Vec<u32>, Vec<u32>) {
        let sample = &self.samples[idx];
        (sample.input_ids.clone(), sample.target_ids.clone())
    }

    fn get_loss_mask(&self, idx: usize) -> Option<Vec<u8>> {
        Some(self.samples[idx].loss_mask.clone())
    }
}

/// Batch of training data
pub struct Batch {
    /// Unpadded inputs, one per sequence.
    pub input_ids: Vec<Vec<u32>>,
    /// Unpadded targets, one per sequence.
    pub target_ids: Vec<Vec<u32>>,
    /// Inputs right-padded to `max_seq_len` with the pad token.
    pub padded_input_ids: Vec<Vec<u32>>,
    /// Targets right-padded to `max_seq_len` with the pad token.
    pub padded_target_ids: Vec<Vec<u32>>,
    /// 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<Vec<u8>>,
    /// Per-sample loss masks aligned with padded targets (1 = train, 0 = ignore).
    pub loss_masks: Vec<Vec<u8>>,
    /// Number of sequences.
    pub batch_size: usize,
    /// Length of the longest sequence.
    pub max_seq_len: usize,
}

/// DataLoader for iterating over datasets with batching
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    pad_token_id: u32,
    shuffle: bool,
    base_indices: Vec<usize>,
    indices: Vec<usize>,
    current_idx: usize,
    seed: Option<u64>,
    epoch: u64,
}

impl<D: Dataset> DataLoader<D> {
    /// A loader over the whole dataset, padding with token 0.
    pub fn new(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        Self::new_with_pad_token(dataset, batch_size, shuffle, 0)
    }

    /// A loader over only `indices` of the dataset (e.g. one rank's shard).
    pub fn new_with_indices(
        dataset: D,
        batch_size: usize,
        shuffle: bool,
        pad_token_id: u32,
        indices: Vec<usize>,
    ) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size: batch_size.max(1),
            pad_token_id,
            shuffle,
            base_indices: indices.clone(),
            indices,
            current_idx: 0,
            seed: None,
            epoch: 0,
        }
    }

    /// A loader over the whole dataset, padding with `pad_token_id`.
    pub fn new_with_pad_token(
        dataset: D,
        batch_size: usize,
        shuffle: bool,
        pad_token_id: u32,
    ) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        Self::new_with_indices(dataset, batch_size, shuffle, pad_token_id, indices)
    }

    /// Make shuffling reproducible: epoch `e` uses an order derived from
    /// `seed` and `e` only (see [`set_epoch`](Self::set_epoch)).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Select the epoch whose order the next [`reset`](Self::reset) produces.
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Rewind to the first batch, reshuffling when shuffling is on. With a
    /// seed the order depends only on the seed and the epoch; without one it
    /// is seeded from the clock.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        self.indices.clone_from(&self.base_indices);
        if self.shuffle {
            let seed = match self.seed {
                Some(seed) => seed ^ self.epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                None => {
                    use std::time::{SystemTime, UNIX_EPOCH};
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0)
                }
            };
            let mut rng = SimpleRng::new(seed);
            // Fisher-Yates
            for i in (1..self.indices.len()).rev() {
                let j = (rng.next() >> 33) as usize % (i + 1);
                self.indices.swap(i, j);
            }
        }
    }

    /// Number of batches per epoch
    pub fn num_batches(&self) -> usize {
        self.indices.len().div_ceil(self.batch_size)
    }

    /// Total number of samples
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end];

        let mut input_ids = Vec::with_capacity(batch_indices.len());
        let mut target_ids = Vec::with_capacity(batch_indices.len());
        let mut loss_masks_raw = Vec::with_capacity(batch_indices.len());
        let mut max_seq_len = 0usize;

        for &idx in batch_indices {
            let (inp, tgt) = self.dataset.get(idx);
            let mask = self
                .dataset
                .get_loss_mask(idx)
                .unwrap_or_else(|| vec![1u8; tgt.len()]);
            max_seq_len = max_seq_len.max(inp.len()).max(tgt.len());
            input_ids.push(inp);
            target_ids.push(tgt);
            loss_masks_raw.push(mask);
        }

        let mut padded_input_ids = Vec::with_capacity(batch_indices.len());
        let mut padded_target_ids = Vec::with_capacity(batch_indices.len());
        let mut attention_mask = Vec::with_capacity(batch_indices.len());
        let mut loss_masks = Vec::with_capacity(batch_indices.len());
        for ((inp, tgt), loss_mask) in input_ids.iter().zip(&target_ids).zip(&loss_masks_raw) {
            let valid_len = inp.len().min(tgt.len());
            let mut padded_inp = inp.clone();
            padded_inp.resize(max_seq_len, self.pad_token_id);
            let mut padded_tgt = tgt.clone();
            padded_tgt.resize(max_seq_len, self.pad_token_id);
            let mut attn = vec![0u8; max_seq_len];
            attn.iter_mut().take(valid_len).for_each(|m| *m = 1);
            let mut padded_loss = vec![0u8; max_seq_len];
            for (dst, src) in padded_loss.iter_mut().zip(loss_mask.iter()).take(valid_len) {
                *dst = *src;
            }
            padded_input_ids.push(padded_inp);
            padded_target_ids.push(padded_tgt);
            attention_mask.push(attn);
            loss_masks.push(padded_loss);
        }

        self.current_idx = end;

        Some(Batch {
            batch_size: batch_indices.len(),
            input_ids,
            target_ids,
            padded_input_ids,
            padded_target_ids,
            attention_mask,
            loss_masks,
            max_seq_len,
        })
    }
}

/// Simple LCG random number generator (no external deps)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next(&mut self) -> u64 {
        // LCG constants from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_dataset() {
        let dataset = TextDataset::from_sequences(vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
        assert_eq!(dataset.len(), 2);

        let (inp, tgt) = dataset.get(0);
        assert_eq!(inp, vec![1, 2, 3]);
        assert_eq!(tgt, vec![2, 3, 4]);
        assert_eq!(dataset.get_loss_mask(0).unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn sft_example_masks_only_response_tokens() {
        let prompt = vec![10, 11, 12];
        let response = vec![20, 21];
        let example = TrainingExample::from_sft_prompt_response(prompt, response);
        // full = [10,11,12,20,21]
        // input = [10,11,12,20]
        // target = [11,12,20,21]
        assert_eq!(example.input_ids, vec![10, 11, 12, 20]);
        assert_eq!(example.target_ids, vec![11, 12, 20, 21]);
        assert_eq!(example.loss_mask, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_dataloader_batching() {
        let dataset = TextDataset::from_sequences(vec![
            vec![1, 2],
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![9, 10],
        ]);
        let loader = DataLoader::new(dataset, 2, false);
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_dataloader_pads_and_masks_batches() {
        let dataset = TextDataset::from_sequences(vec![vec![1, 2, 3, 4], vec![5, 6, 7]]);
        let mut loader = DataLoader::new_with_pad_token(dataset, 2, false, 99);
        let batch = loader.next().expect("expected batch");
        assert_eq!(batch.max_seq_len, 3);
        assert_eq!(batch.padded_input_ids[0], vec![1, 2, 3]);
        assert_eq!(batch.padded_input_ids[1], vec![5, 6, 99]);
        assert_eq!(batch.padded_target_ids[1], vec![6, 7, 99]);
        assert_eq!(batch.attention_mask[0], vec![1, 1, 1]);
        assert_eq!(batch.attention_mask[1], vec![1, 1, 0]);
        assert_eq!(batch.loss_masks[1], vec![1, 1, 0]);
    }

    #[test]
    fn packed_bin_and_txt_share_next_token_mask_semantics() {
        // Pretrain windows from packed tokens should mark every target as supervised.
        let packed = TextDataset::from_packed_tokens(vec![1, 2, 3, 4, 5, 6], 4);
        assert!(!packed.is_empty());
        for idx in 0..packed.len() {
            let mask = packed.get_loss_mask(idx).unwrap();
            assert!(mask.iter().all(|&m| m == 1));
            let (inp, tgt) = packed.get(idx);
            assert_eq!(inp.len(), tgt.len());
            assert_eq!(mask.len(), tgt.len());
        }

        // Raw short sequences (txt line analogue) use the same next-token mask rule.
        let lines = TextDataset::from_sequences(vec![vec![9, 8, 7, 6]]);
        assert_eq!(lines.get_loss_mask(0).unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn sft_dataloader_preserves_response_mask_under_padding() {
        let mut dataset = TextDataset::new();
        dataset.add_example(TrainingExample::from_sft_prompt_response(
            vec![1, 2, 3],
            vec![4, 5],
        ));
        dataset.add_example(TrainingExample::from_sft_prompt_response(
            vec![10, 11],
            vec![12],
        ));
        let mut loader = DataLoader::new_with_pad_token(dataset, 2, false, 0);
        let batch = loader.next().expect("batch");
        // First example mask was [0,0,1,1]; padding should keep prompt=0 and pad=0.
        assert_eq!(batch.loss_masks[0].iter().filter(|&&m| m == 1).count(), 2);
        assert!(batch.loss_masks[0].iter().take(2).all(|&m| m == 0));
        // Second example response-only positions stay marked.
        assert!(batch.loss_masks[1].contains(&1));
        assert!(batch.loss_masks[1].contains(&0));
    }
}

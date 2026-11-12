//! Greedy generation with and without the KV cache, on a tiny model with no
//! download.
//!
//! Builds the same tiny random model as `lora_finetune_tiny`, trains its LoRA
//! adapters briefly on the fixed pattern (so the next token is a clear
//! choice), then generates 12 tokens from a 4-token prompt twice:
//!
//! - without a cache: every step runs the whole sequence again;
//! - with a cache: the prompt runs once, then each step feeds only the new
//!   token and reads the earlier keys and values from the cache.
//!
//! The program exits non-zero if the two token sequences differ.
//!
//! Run with: `cargo run --release -p ferrotensor-trainer --example generate_with_kv_cache`
//!
//! Output from the CI job "GPU tests (software Vulkan)" (Mesa lavapipe):
//!
//! ```text
//! device: llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan)
//! trained LoRA for 30 steps, final loss 0.5063
//! prompt:            [3, 10, 17, 24]
//! without KV cache:  [31, 38, 45, 52, 59, 2, 9, 16, 23, 30, 37, 44]
//! with KV cache:     [31, 38, 45, 52, 59, 2, 9, 16, 23, 30, 37, 44]
//! training pattern:  [31, 38, 45, 52, 59, 2, 9, 16, 23, 30, 37, 44]
//! largest logit difference between the two: 0.00e0
//! tokens that continue the pattern: 12 of 12
//! ```

#[path = "common/mod.rs"]
mod common;

use common::*;
use ferrotensor_autograd::AdamW;
use ferrotensor_graph::lfm::Lfm2Model;

const TRAIN_STEPS: usize = 30;
const PROMPT: usize = 4;
const NEW_TOKENS: usize = SEQ - PROMPT;

/// Index of the largest value (the first one on a tie).
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Greedily generate `n` tokens after `prompt`. Returns the new tokens and
/// the logits each one was picked from.
fn greedy(
    gpu: &Gpu,
    model: &Lfm2Model,
    prompt: &[u32],
    n: usize,
    use_cache: bool,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut cache = if use_cache {
        Some(
            model
                .new_kv_cache(prompt.len() + n)
                .expect("a Llama-style model supports a KV cache"),
        )
    } else {
        None
    };
    let mut tokens = prompt.to_vec();
    let mut rows = Vec::with_capacity(n);
    let mut logits = model.forward(&gpu.engine, &tokens, 0, cache.as_mut(), None);
    for i in 0..n {
        let host = pollster::block_on(logits.data.to_vec(&gpu.ctx));
        let last = host[host.len() - VOCAB..].to_vec();
        let next = argmax(&last);
        rows.push(last);
        tokens.push(next);
        if i + 1 == n {
            break;
        }
        logits = if use_cache {
            // Only the new token, at its position; the cache holds the rest.
            model.forward(&gpu.engine, &[next], tokens.len() - 1, cache.as_mut(), None)
        } else {
            model.forward(&gpu.engine, &tokens, 0, None, None)
        };
    }
    (tokens[prompt.len()..].to_vec(), rows)
}

fn main() {
    let dir = TempDir::new("generate");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern();

    let model = build_model(&gpu, dir.path());
    let params = sorted_lora_params(&model);
    let mut trainer = trainer(&gpu, model, AdamW::new(gpu.engine.clone(), 1e-2));
    let mut loss = 0.0;
    for _ in 0..TRAIN_STEPS {
        loss = trainer
            .train_step(&seq[..SEQ - 1], &seq[1..], &params)
            .expect("train step")
            .loss;
    }
    println!("trained LoRA for {TRAIN_STEPS} steps, final loss {loss:.4}");
    let model = trainer.model;

    let prompt = &seq[..PROMPT];
    let (plain, plain_rows) = greedy(&gpu, &model, prompt, NEW_TOKENS, false);
    let (cached, cached_rows) = greedy(&gpu, &model, prompt, NEW_TOKENS, true);
    println!("prompt:            {prompt:?}");
    println!("without KV cache:  {plain:?}");
    println!("with KV cache:     {cached:?}");
    println!("training pattern:  {:?}", &seq[PROMPT..]);

    let max_diff = plain_rows
        .iter()
        .flatten()
        .zip(cached_rows.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("largest logit difference between the two: {max_diff:.2e}");
    assert_eq!(
        plain, cached,
        "greedy generation should not depend on the KV cache"
    );
    let matched = plain
        .iter()
        .zip(&seq[PROMPT..])
        .take_while(|(a, b)| a == b)
        .count();
    println!("tokens that continue the pattern: {matched} of {NEW_TOKENS}");
}

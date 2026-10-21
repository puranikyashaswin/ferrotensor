//! A tiny character-level transformer trained from scratch on a few
//! Shakespeare sonnets (public domain, `data/sonnets.txt`, ~3.7 KB).
//!
//! The corpus is tiny, so the model starts to memorise it: this shows the
//! training loop and the building blocks working end to end, not a useful
//! language model.
//!
//! Run with: cargo run --release -p ferrotensor --example char_transformer
//! Set FERROTENSOR_DEVICE=gpu to run on the GPU (the default is the CPU, which is
//! faster for a model this small).

use std::collections::BTreeMap;

use ferrotensor::optim::{LrScheduler, WarmupCosineLr};
use ferrotensor::prelude::*;
use ferrotensor::{no_grad, Tensor as T};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CONTEXT: usize = 48;
const BATCH: usize = 16;
const DIM: usize = 64;
const HEADS: usize = 4;
const LAYERS: usize = 2;
const STEPS: usize = 600;

/// Pre-norm transformer block: `x + attn(ln(x))`, then `x + mlp(ln(x))`.
struct Block {
    ln1: nn::LayerNorm,
    attn: nn::MultiHeadAttention,
    ln2: nn::LayerNorm,
    up: nn::Linear,
    down: nn::Linear,
}

impl Block {
    fn new(device: &Device) -> Self {
        Self {
            ln1: nn::LayerNorm::new(&[DIM], device),
            attn: nn::MultiHeadAttention::new(DIM, HEADS, device).causal(true),
            ln2: nn::LayerNorm::new(&[DIM], device),
            up: nn::Linear::new(DIM, 4 * DIM, device),
            down: nn::Linear::new(4 * DIM, DIM, device),
        }
    }

    fn forward(&self, x: &T) -> T {
        let x = x.add(&self.attn.forward(&self.ln1.forward(x)));
        let h = self
            .down
            .forward(&self.up.forward(&self.ln2.forward(&x)).gelu());
        x.add(&h)
    }

    fn parameters(&self) -> Vec<T> {
        [
            self.ln1.parameters(),
            self.attn.parameters(),
            self.ln2.parameters(),
            self.up.parameters(),
            self.down.parameters(),
        ]
        .concat()
    }
}

struct CharLm {
    tok: nn::Embedding,
    pos: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: nn::LayerNorm,
    head: nn::Linear,
}

impl CharLm {
    fn new(vocab: usize, device: &Device) -> Self {
        Self {
            tok: nn::Embedding::new(vocab, DIM, device),
            pos: nn::Embedding::new(CONTEXT, DIM, device),
            blocks: (0..LAYERS).map(|_| Block::new(device)).collect(),
            ln_f: nn::LayerNorm::new(&[DIM], device),
            head: nn::Linear::new(DIM, vocab, device),
        }
    }

    /// `ids` holds `batch * len` token ids (row-major); returns `[batch, len, vocab]`.
    fn forward(&self, ids: &[usize], batch: usize, len: usize) -> T {
        let positions: Vec<usize> = (0..len).collect();
        let x = self.tok.forward_ids(ids).reshape(&[batch, len, DIM]);
        let mut x = x.add(&self.pos.forward_ids(&positions));
        for block in &self.blocks {
            x = block.forward(&x);
        }
        self.head.forward(&self.ln_f.forward(&x))
    }

    fn parameters(&self) -> Vec<T> {
        let mut p = [self.tok.parameters(), self.pos.parameters()].concat();
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.extend(self.ln_f.parameters());
        p.extend(self.head.parameters());
        p
    }

    /// Sample `n` characters after `prompt` at the given temperature.
    fn generate(
        &self,
        prompt: &[usize],
        n: usize,
        temperature: f32,
        rng: &mut StdRng,
    ) -> Vec<usize> {
        let mut ids = prompt.to_vec();
        for _ in 0..n {
            let window = &ids[ids.len().saturating_sub(CONTEXT)..];
            let logits = no_grad(|| self.forward(window, 1, window.len()));
            let vocab = logits.shape[2];
            let all = logits.to_vec();
            let last = &all[all.len() - vocab..];
            let max = last.iter().cloned().fold(f32::MIN, f32::max);
            let weights: Vec<f32> = last
                .iter()
                .map(|l| ((l - max) / temperature).exp())
                .collect();
            let mut r = rng.gen::<f32>() * weights.iter().sum::<f32>();
            let mut next = vocab - 1;
            for (i, w) in weights.iter().enumerate() {
                if r < *w {
                    next = i;
                    break;
                }
                r -= w;
            }
            ids.push(next);
        }
        ids
    }
}

fn main() {
    let device = match std::env::var("FERROTENSOR_DEVICE").as_deref() {
        Ok("gpu") => Device::new(),
        _ => Device::cpu(),
    };
    println!("device: {} ({})", device.name(), device.backend());

    let text = include_str!("data/sonnets.txt");
    let chars: Vec<char> = text.chars().collect();
    let vocab: BTreeMap<char, usize> = {
        let mut set: Vec<char> = chars.clone();
        set.sort();
        set.dedup();
        set.into_iter().enumerate().map(|(i, c)| (c, i)).collect()
    };
    let itos: Vec<char> = vocab.keys().copied().collect();
    let data: Vec<usize> = chars.iter().map(|c| vocab[c]).collect();
    println!(
        "corpus: {} characters, vocabulary {}",
        data.len(),
        vocab.len()
    );

    let model = CharLm::new(vocab.len(), &device);
    let n_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("model: {LAYERS} layers, dim {DIM}, {HEADS} heads, {n_params} parameters");

    let mut opt = optim::AdamW::new(model.parameters(), 3e-3)
        .weight_decay(0.01)
        .clip_grad_norm(1.0);
    let schedule = WarmupCosineLr::new(3e-3, 3e-4, 30, STEPS);
    let mut rng = StdRng::seed_from_u64(42);

    let start = std::time::Instant::now();
    for step in 0..STEPS {
        schedule.apply(&mut opt, step);
        let (mut xs, mut ys) = (Vec::new(), Vec::new());
        for _ in 0..BATCH {
            let s = rng.gen_range(0..data.len() - CONTEXT - 1);
            xs.extend_from_slice(&data[s..s + CONTEXT]);
            ys.extend(data[s + 1..s + CONTEXT + 1].iter().map(|&t| t as f32));
        }
        opt.zero_grad();
        let logits = model.forward(&xs, BATCH, CONTEXT);
        let targets = T::from_data(&ys, &[BATCH * CONTEXT], &device);
        let loss = F::cross_entropy(&logits.reshape(&[BATCH * CONTEXT, vocab.len()]), &targets);
        loss.backward();
        opt.step();
        if step % 100 == 0 || step == STEPS - 1 {
            println!("step {step:>3}  loss {:.3}", loss.item());
        }
    }
    println!(
        "trained {STEPS} steps in {:.1}s",
        start.elapsed().as_secs_f32()
    );

    let prompt = "Shall I compare";
    let ids: Vec<usize> = prompt.chars().map(|c| vocab[&c]).collect();
    let out = model.generate(&ids, 200, 0.8, &mut rng);
    println!("\n--- sample (prompt: {prompt:?}) ---");
    println!("{}", out.iter().map(|&i| itos[i]).collect::<String>());
}

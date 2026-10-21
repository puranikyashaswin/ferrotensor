# Use cases

Four small programs that show where the `rusty` crate fits today. Each lives
in [`rusty/examples/`](../rusty/examples), runs with one command, and the
output below was captured from a real run on an Apple M2 (Metal) with Rust
1.99.1. Numbers that depend on random initialisation vary from run to
run; timings are single-machine snapshots, not benchmarks.

All commands assume you are in the repository root.

| Example | Shows | Device |
|---------|-------|--------|
| [`mlp_two_moons`](#1-train-an-mlp-with-a-dataloader) | `DataLoader`, `Sequential`, `AdamW`, LR schedule | CPU (GPU optional) |
| [`char_transformer`](#2-a-character-level-transformer-from-scratch) | attention, LayerNorm, a custom `Module`-style model, sampling | CPU (GPU optional) |
| [`gpu_matmul_attention`](#3-gpu-matmul-and-attention) | CPU/GPU parity and where the GPU wins | GPU |
| [`similarity_search`](#4-similarity-search) | embeddings + batched dot-product search | GPU if present |

## 1. Train an MLP with a DataLoader

**Where it fits:** classic small-model training with no Python and no GPU
required. Everything here is the public `rusty` API: `Dataset` /
`TensorDataset` / `DataLoader`, `nn::Sequential`, `F::cross_entropy`,
`optim::AdamW` and a `WarmupCosineLr` schedule.

```bash
cargo run --release -p rusty --example mlp_two_moons
```

```
device: CPU (Cpu)
train: 800 points, test: 400 points
model: 2-32-32-2 MLP, 1218 parameters
epoch  0  train loss 0.4292  test accuracy 85.8%
epoch 10  train loss 0.0319  test accuracy 98.5%
epoch 20  train loss 0.0150  test accuracy 97.5%
epoch 30  train loss 0.0146  test accuracy 99.2%
epoch 40  train loss 0.0187  test accuracy 99.5%
epoch 50  train loss 0.0101  test accuracy 99.5%
epoch 59  train loss 0.0081  test accuracy 99.5%
```

The data are two noisy interleaved half-circles generated in the example, so
this is a smoke test of the training stack, not a benchmark. The whole run
takes about half a second on the CPU.

`RUSTY_DEVICE=gpu` trains the same model on the GPU. It works (final test
accuracy 99.5% in our run) but takes tens of seconds instead of half a second:
a 1,218-parameter model gives the GPU nothing to amortise its per-op dispatch
cost over. Use the CPU device for models this small.

## 2. A character-level transformer from scratch

**Where it fits:** understanding or prototyping a transformer end to end in
Rust. The example builds a two-layer pre-norm GPT-style model out of
`nn::Embedding`, `nn::LayerNorm`, `nn::MultiHeadAttention` (causal) and
`nn::Linear`, trains it with AdamW, gradient clipping and warmup+cosine LR on
[a few Shakespeare sonnets](../rusty/examples/data/sonnets.txt) (public
domain, 3.7 KB), then samples text.

```bash
cargo run --release -p rusty --example char_transformer
```

```
device: CPU (Cpu)
corpus: 3700 characters, vocabulary 54
model: 2 layers, dim 64, 4 heads, 109622 parameters
step   0  loss 4.577
step 100  loss 2.280
step 200  loss 1.971
step 300  loss 1.588
step 400  loss 1.037
step 500  loss 0.722
step 599  loss 0.603
trained 600 steps in 47.6s

--- sample (prompt: "Shall I compare") ---
Shall I comparestemaf look con.
Lovet ive
Which to not me man's boyesh'd doth gloks ote tand warith crief worinng.
Lot not love t, no bnce with thi by thou ven merovem! it everWh wok d.


Then imowf  iforth suow f h
```

What this does and does not show:

- The loss falls from 4.6 (about `ln 54 = 4.0` plus init noise) to 0.6 on
  training windows. With 3.7 KB of text and 110k parameters the model is
  starting to memorise the corpus; there is no held-out set, so this says
  nothing about generalisation.
- The samples are partly English-shaped, partly garbage. It is a working
  training loop, not a language model.
- On this machine the GPU path (`RUSTY_DEVICE=gpu`) took 141 s for the same
  600 steps versus 44-48 s on the CPU. Each op is a separate GPU submission
  and tiny kernels do not pay for it. Improving that is a known open item (see
  [good-first-issues](good-first-issues.md)).

## 3. GPU matmul and attention

**Where it fits:** this is where the GPU backend earns its keep: large dense
ops. The example runs the same matmul and causal attention on a CPU tensor
and a GPU tensor, checks they agree, and times both (best of several runs,
including readback).

```bash
cargo run --release -p rusty --example gpu_matmul_attention
```

```
gpu: Apple M2 (Metal)

matmul [n, n] @ [n, n]
     n     cpu ms     gpu ms   speedup   max |diff|
   256        2.1        1.4      1.5x      1.53e-5
   512       10.0        5.2      1.9x      3.05e-5
  1024       89.5       23.0      3.9x      5.34e-5
  2048     1213.8      444.4      2.7x      1.07e-4

causal scaled-dot-product attention, [1, 8 heads, seq, 64]
   seq     cpu ms     gpu ms   speedup   max |diff|
   128        3.6        4.7      0.8x      4.77e-7
   512       51.5       21.5      2.4x      7.15e-7
```

Read this honestly: the GPU is 2-4x faster on large matmuls here, not orders
of magnitude, and slower at small sizes. The 2048x2048 matmul is about 39
GFLOPS on the GPU (2 n^3 flops / 0.444 s), and that run shared the machine
with other jobs, so expect noise. The agreement column (differences of
1e-5 to 1e-4 for matmul, ~1e-7 for attention) is the number the test suite
enforces. For the project's benchmark harness see `benchmarks/RESULTS.md`
(TODO: link once the benchmarks branch lands).

## 4. Similarity search

**Where it fits:** retrieval over embeddings. The search is one
`[queries, dim] @ [dim, docs]` matmul on the device followed by a top-k on the
host. The example embeds text with hashed character trigrams into 512
dimensions, so it needs no model weights, but it is a *lexical* embedding:
shared words and typos match, synonyms do not. Swap in embeddings from a
real model and the search code is unchanged.

```bash
cargo run --release -p rusty --example similarity_search
```

```
device: Apple M2 (Metal)

query: "forgot my password"
  0.545  How do I reset my password?
  0.380  Steps to change your account password
query: "revenue growth this quarter"
  0.584  Revenue increased 12% quarter over quarter
  0.513  Quarterly revenue grew by twelve percent
query: "kittens on a ledge"
  0.511  A kitten is sleeping on the window ledge
  0.154  Where to find scenic walking routes by the lake
query: "how to install with cargo"
  0.564  Install the package with cargo add
  0.458  Run cargo install to get the command line tool
query: "lake hiking trails"
  0.643  Best hiking trails near the mountain lake
  0.260  Where to find scenic walking routes by the lake
```

Top results are right for all five queries. Second results are
sometimes weak (the second hit for "kittens on a ledge" is unrelated: the
score 0.154 shows it), which is what a lexical embedding gives you.

## What does not fit (yet)

- **Large-model training or multi-GPU.** Not supported. The LoRA fine-tuning
  path for small Llama-style models is experimental; see the README.
- **Big training jobs on the `rusty` tensor API.** Tensors are `f32` only, there
  are no convolutions, and per-op dispatch overhead makes small GPU workloads
  slower than the CPU.

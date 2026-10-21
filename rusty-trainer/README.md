# rusty-trainer

Training loops for [Rusty](../README.md)'s LLM stack: `Trainer` (response-masked
cross-entropy, gradient accumulation averaged over the window, global
gradient-norm clipping, learning-rate schedules, optional `GradScaler`),
SFT and pretraining datasets with a seeded `DataLoader`, LoRA adapter
save/load (safetensors with rank/alpha/targets metadata) and resumable
training checkpoints, plus a file-based gradient sync prototype for
multi-process runs. **Experimental.**

Most people use it through `rusty-cli`:

```bash
cargo run -p rusty-cli --release -- train \
  --model ./data/models/SmolLM2-135M \
  --data ./data/alpaca/train.jsonl --out ./data/runs/sft
```

A walkthrough with measured numbers is in
[docs/examples/lora-sft.md](../docs/examples/lora-sft.md). For the library API
see `Trainer::new`, `Trainer::train_step_masked`, `masked_cross_entropy` and
the `checkpoint` module.

GPU tests (a tiny random model trained for a few steps, resume round trip,
KV-cache decoding) are ignored by default:

```bash
cargo test -p rusty-trainer --test gpu_training -- --ignored --test-threads=1
```

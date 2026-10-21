# rusty-trainer

Training loops for [Rusty](../README.md)'s LLM stack: `Trainer` (gradient
accumulation, gradient clipping, optional `GradScaler` for FP16 loss scaling),
masked cross-entropy for response-only SFT loss, text and packed-token
datasets with a `DataLoader`, checkpoint save/resume, and a file-based
gradient sync prototype for multi-process runs. **Experimental.**

Most people use it through `rusty-cli`:

```bash
cargo run -p rusty-cli --release -- train \
  --lite --task sft --model ./models/tinyllama \
  --data ./data/train.jsonl --out ./outputs/run1
```

A walkthrough with real output is planned at
[docs/examples/lora-sft.md](../docs/examples/lora-sft.md) (written by another
branch). For the library API see `Trainer::new`, `Trainer::train_step_masked`
and `Trainer::train_batch`.

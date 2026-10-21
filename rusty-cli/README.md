# rusty-cli

The `rusty-cli` command line for [Rusty](../README.md): `train` (LoRA SFT or
pretraining), `eval` and `infer`, with consumer-hardware defaults behind
`--lite`.

```bash
cargo run -p rusty-cli --release -- --help     # all commands and flags

cargo run -p rusty-cli --release -- train \
  --lite --task sft \
  --model ./models/tinyllama \
  --data ./data/train.jsonl \
  --out ./outputs/run1
```

SFT data is `.json`/`.jsonl` with `{"prompt": ..., "response": ...}`
(optional `"system"`); pretraining takes `.txt` or packed `.bin` token files.
`--help` also lists `--demo`, `--monitor` and `--benchmark`.

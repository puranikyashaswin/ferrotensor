#!/usr/bin/env bash
# Run the LoRA SFT walkthrough (docs/examples/lora-sft.md) on any Llama-style
# Hugging Face model and write one JSON summary of the result.
#
#   scripts/lora-run.sh HuggingFaceTB/SmolLM2-360M
#   scripts/lora-run.sh HuggingFaceTB/SmolLM2-1.7B --steps 250 --seq-len 256
#
# Same data, split and hyperparameters as the walkthrough unless overridden:
# 2,000 Alpaca examples for training, 64 held out, LoRA rank 8 on q/k/v/o,
# lr 5e-4, 20 warmup steps, 8 sequences per step. Everything goes under data/
# (gitignored); the summary is data/runs/<model>/summary.json. Needs the `hf`
# CLI (pip install huggingface_hub). Respects WGPU_BACKEND.
set -euo pipefail

usage() { sed -n '2,13p' "$0"; exit "${1:-0}"; }
[ $# -ge 1 ] || usage 2
repo="$1"; shift
steps=250
seq_len=256
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) steps="$2"; shift 2 ;;
    --seq-len) seq_len="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
name="${repo##*/}"
model_dir="data/models/$name"
run_dir="data/runs/$name"
mkdir -p "$run_dir" data/alpaca

command -v hf >/dev/null || { echo "needs the hf CLI: pip install huggingface_hub" >&2; exit 1; }

echo "== model $repo"
hf download "$repo" --include "*.safetensors" --include "*.json" --local-dir "$model_dir" >/dev/null

if [ ! -s data/alpaca/train.jsonl ] || [ ! -s data/alpaca/val.jsonl ]; then
  echo "== data (alpaca-cleaned, walkthrough split)"
  hf download yahma/alpaca-cleaned alpaca_data_cleaned.json \
    --repo-type dataset --local-dir data/alpaca-cleaned >/dev/null
  python3 - <<'PY'
import json, random
rows = json.load(open("data/alpaca-cleaned/alpaca_data_cleaned.json"))
out = []
for r in rows:
    p = r["instruction"].strip()
    if r.get("input", "").strip():
        p += "\n\n" + r["input"].strip()
    resp = r["output"].strip()
    if len(p) + len(resp) <= 500 and resp:
        out.append({"prompt": p, "response": resp})
random.Random(0).shuffle(out)
for name, rs in (("val", out[:64]), ("train", out[64:2064])):
    with open(f"data/alpaca/{name}.jsonl", "w") as f:
        for x in rs:
            f.write(json.dumps(x) + "\n")
PY
fi

echo "== build"
cargo build --release -q -p ferrotensor-cli
bin=./target/release/ferrotensor

# Peak memory: BSD time on macOS, GNU time on Linux.
if [ "$(uname)" = Darwin ]; then time_args=(-l); else time_args=(-v); fi

echo "== train ($steps steps, seq-len $seq_len)"
/usr/bin/time "${time_args[@]}" "$bin" train \
  --model "$model_dir" \
  --data data/alpaca/train.jsonl --eval-data data/alpaca/val.jsonl \
  --out "$run_dir" --seq-len "$seq_len" \
  --lr 5e-4 --warmup-steps 20 --grad-accum 8 --max-steps "$steps" \
  --log-every 10 --eval-every 50 --save-every 50 \
  --sample-prompt "Give three tips for staying healthy." \
  2>&1 | tee "$run_dir/train.log"

adapter="$run_dir/lora_adapters.safetensors"
echo "== eval with the adapter"
"$bin" eval --model "$model_dir" --data data/alpaca/val.jsonl --lora "$adapter" \
  2>&1 | tee "$run_dir/eval.log"

echo "== generate (greedy), base and adapter"
prompts=("Translate 'good morning' into French." "Give three tips for staying healthy.")
: > "$run_dir/generate.log"
for p in "${prompts[@]}"; do
  for mode in base adapter; do
    lora=()
    [ "$mode" = adapter ] && lora=(--lora "$adapter")
    printf '### %s | %s\n' "$mode" "$p" >> "$run_dir/generate.log"
    "$bin" generate --model "$model_dir" ${lora[@]+"${lora[@]}"} \
      --prompt "$p" --temperature 0 --max-new-tokens 60 >> "$run_dir/generate.log" 2>&1
  done
done

python3 - "$run_dir" "$repo" "$steps" "$seq_len" "$(git rev-parse --short HEAD)" <<'PY'
import json, re, sys
from pathlib import Path

run, repo, steps, seq_len, commit = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
train = (run / "train.log").read_text()
evals = (run / "eval.log").read_text()

def find(pattern, text, cast=str):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None

metrics = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()]
step_rows = [m for m in metrics if "loss" in m and "seconds" in m]
seconds = [m["seconds"] for m in step_rows]
# Peak resident memory: "maximum resident set size" (bytes, macOS) or
# "Maximum resident set size (kbytes)" (Linux).
mac = find(r"(\d+)\s+maximum resident set size", train, int)
linux = find(r"Maximum resident set size \(kbytes\):\s*(\d+)", train, int)
peak_bytes = mac if mac is not None else (linux * 1024 if linux is not None else None)

generations = []
for block in (run / "generate.log").read_text().split("### ")[1:]:
    head, _, body = block.partition("\n")
    mode, _, prompt = head.partition(" | ")
    generations.append({"mode": mode, "prompt": prompt, "output": body.strip()})

summary = {
    "model": repo,
    "commit": commit,
    "device": find(r"device:\s*(.+)", train),
    "model_shape": find(r"model:\s*\S+ \((.+)\)", train),
    "trainable_params": find(r"lora \((\d+) trainable params\)", train, int),
    "steps": steps,
    "seq_len": seq_len,
    "heldout_loss_before": find(r"eval before: loss ([\d.]+)", train, float),
    "heldout_loss_after": find(r"loss ([\d.]+), perplexity", evals, float),
    "heldout_ppl_after": find(r"perplexity ([\d.]+)", evals, float),
    "train_loss_curve": [{"step": m["step"], "loss": m["loss"]} for m in step_rows],
    "seconds_per_step_median": sorted(seconds)[len(seconds) // 2] / 10 if seconds else None,
    "peak_memory_gb": round(peak_bytes / 1e9, 2) if peak_bytes else None,
    "adapter_mb": round((run / "lora_adapters.safetensors").stat().st_size / 1e6, 1),
    "generations": generations,
}
(run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"\nsummary: {run / 'summary.json'}")
print(f"held-out loss {summary['heldout_loss_before']} -> {summary['heldout_loss_after']}, "
      f"peak {summary['peak_memory_gb']} GB, {summary['seconds_per_step_median']} s/step")
PY

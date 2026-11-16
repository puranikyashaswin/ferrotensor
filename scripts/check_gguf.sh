#!/bin/bash
# Compares a greedy completion of a GGUF file from `ferrotensor export-gguf`
# in llama.cpp with `ferrotensor generate` on the model directory it came from.
# Usage: scripts/check_gguf.sh LLAMA_SIMPLE MODEL_DIR FILE.gguf ["prompt"]
# LLAMA_SIMPLE is llama.cpp's llama-simple binary, which samples greedily.
# Set FERROTENSOR to the ferrotensor binary when it is not on PATH.
set -euo pipefail
llama=$1 model=$2 gguf=$3 prompt=${4:-"The capital of France is"}
ours=$("${FERROTENSOR:-ferrotensor}" generate --model "$model" --raw --prompt "$prompt" \
  --temperature 0 --max-new-tokens 16 --json | jq -r .text)
theirs=$("$llama" -m "$gguf" -n 16 "$prompt" 2>/dev/null)
echo "ferrotensor: $ours"
echo "llama.cpp:   $theirs"
if [[ "$theirs" == *"$ours"* ]]; then echo "match"; else echo "mismatch"; exit 1; fi

#!/bin/bash
# Rusty ML - Model Download Script
# Downloads popular models for testing and fine-tuning

set -e

MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"

print_header() {
    echo "================================================================"
    echo "   Rusty ML - Model Downloader"
    echo "================================================================"
    echo ""
}

download_tinyllama() {
    echo "[INFO] Downloading TinyLlama-1.1B..."
    echo "       This is a small model (~2GB) perfect for testing."
    echo ""
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
            --local-dir "$MODELS_DIR/tinyllama" \
            --include "*.safetensors" "*.json" "tokenizer.model" "tokenizer.json"
        echo ""
        echo "[SUCCESS] TinyLlama downloaded to $MODELS_DIR/tinyllama"
    else
        echo "[ERROR] huggingface-cli not found!"
        echo ""
        echo "Install it with:"
        echo "  pip install huggingface-hub"
        echo ""
        echo "Then run this script again."
        exit 1
    fi
}

download_phi2() {
    echo "[INFO] Downloading Phi-2 (2.7B)..."
    echo "       Microsoft's efficient reasoning model."
    echo ""
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download microsoft/phi-2 \
            --local-dir "$MODELS_DIR/phi2" \
            --include "*.safetensors" "*.json" "tokenizer.model"
        echo ""
        echo "[SUCCESS] Phi-2 downloaded to $MODELS_DIR/phi2"
    else
        echo "[ERROR] huggingface-cli not found!"
        exit 1
    fi
}

download_gemma() {
    echo "[INFO] Downloading Gemma-2B..."
    echo "       Google's efficient open model."
    echo ""
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download google/gemma-2b \
            --local-dir "$MODELS_DIR/gemma2b" \
            --include "*.safetensors" "*.json" "tokenizer.model"
        echo ""
        echo "[SUCCESS] Gemma-2B downloaded to $MODELS_DIR/gemma2b"
    else
        echo "[ERROR] huggingface-cli not found!"
        exit 1
    fi
}

print_usage() {
    echo "USAGE:"
    echo "  ./scripts/download_model.sh [MODEL]"
    echo ""
    echo "AVAILABLE MODELS:"
    echo "  tinyllama    TinyLlama-1.1B (recommended for testing)"
    echo "  phi2         Microsoft Phi-2 (2.7B)"
    echo "  gemma        Google Gemma-2B"
    echo ""
    echo "EXAMPLES:"
    echo "  ./scripts/download_model.sh tinyllama"
    echo "  ./scripts/download_model.sh phi2"
    echo ""
    echo "After downloading, run:"
    echo "  cargo run -p rusty-cli --release -- generate --model ./models/tinyllama --prompt \"Hello\""
}

# Main
print_header

case "${1:-}" in
    tinyllama)
        download_tinyllama
        ;;
    phi2)
        download_phi2
        ;;
    gemma)
        download_gemma
        ;;
    "")
        echo "[INFO] No model specified, downloading TinyLlama (recommended)..."
        echo ""
        download_tinyllama
        ;;
    *)
        echo "[ERROR] Unknown model: $1"
        echo ""
        print_usage
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo "[NEXT STEPS]"
echo ""
echo "1. Create a training dataset:"
echo "   echo '[{\"prompt\": \"Hello\", \"response\": \"Hi there!\"}]' > data/train.json"
echo ""
echo "2. Run LoRA fine-tuning:"
echo "   cargo run -p rusty-cli --release -- train --model $MODELS_DIR/tinyllama --data data/train.json --out outputs/run1"
echo ""
echo "3. Generate with the adapter:"
echo "   cargo run -p rusty-cli --release -- generate --model $MODELS_DIR/tinyllama --lora outputs/run1/lora_adapters.safetensors --prompt \"Hello\""
echo "================================================================"

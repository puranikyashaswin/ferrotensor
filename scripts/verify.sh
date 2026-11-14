#!/bin/bash
# Ferrotensor - quick verification
# Runs the checks CI runs on every pull request, on this machine, without a GPU:
# formatting, clippy, the workspace tests and the examples that need no GPU.
# For the examples that need a GPU run: scripts/run-examples.sh --gpu

set -e

cd "$(dirname "$0")/.."

echo "================================================================"
echo "   Ferrotensor - Quick Verification"
echo "================================================================"
echo ""

echo "[1/5] Checking Rust version..."
rustc --version
echo ""

echo "[2/5] Checking formatting..."
cargo fmt --all -- --check
echo "       Formatting OK"
echo ""

echo "[3/5] Running clippy..."
cargo clippy --workspace --all-targets -- -D warnings
echo "       Clippy OK"
echo ""

echo "[4/5] Running the workspace tests..."
cargo test --workspace
echo "       Tests OK"
echo ""

echo "[5/5] Running the examples that need no GPU..."
scripts/run-examples.sh --cpu
echo ""

echo "================================================================"
echo "   VERIFICATION COMPLETE"
echo "================================================================"
echo ""
echo "Formatting, clippy, tests and the CPU examples all passed."
echo ""
echo "Next steps:"
echo "  1. GPU examples:        ./scripts/run-examples.sh --gpu"
echo "  2. Download a model:    ./scripts/download_model.sh tinyllama"
echo "  3. Run demo:            cargo run -p ferrotensor-cli --release -- --demo"
echo "  4. Run benchmarks:      cargo run -p ferrotensor-benchmarks --release"
echo ""

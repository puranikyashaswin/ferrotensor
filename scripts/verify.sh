#!/bin/bash
# Ferrotensor ML - Quick Verification Script
# Verifies that the workspace builds and the basic GPU example runs.

set -e

echo "================================================================"
echo "   Ferrotensor ML - Quick Verification"
echo "================================================================"
echo ""

echo "[1/4] Checking Rust version..."
rustc --version
echo ""

echo "[2/4] Checking workspace compiles..."
cargo check --workspace
echo "       Workspace OK"
echo ""

echo "[3/4] Building basic_tensor example..."
cargo build --example basic_tensor --release -p ferrotensor
echo "       Build complete"
echo ""

echo "[4/4] Running GPU example (requires a WebGPU-capable GPU)..."
cargo run --example basic_tensor --release -p ferrotensor
echo ""

echo "================================================================"
echo "   VERIFICATION COMPLETE"
echo "================================================================"
echo ""
echo "Core build and GPU example succeeded."
echo ""
echo "Next steps:"
echo "  1. Download a model:    ./scripts/download_model.sh tinyllama"
echo "  2. Run demo:            cargo run -p ferrotensor-cli --release -- --demo"
echo "  3. Run benchmarks:      cargo run -p ferrotensor-benchmarks --release"
echo ""

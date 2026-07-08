# Rusty GPU Benchmarks

## Results (Apple M2, Metal Backend)

These numbers are kernel-level GPU measurements, not end-to-end training throughput. Use them as backend health checks, not as proof that a full training run will match the same utilization.

```
╔════════════════════════════════════════════════════════════════╗
║             RUSTY ML FRAMEWORK - GPU BENCHMARKS                ║
╚════════════════════════════════════════════════════════════════╝

GPU Information:
    Name:    Apple M2
    Backend: Metal
    Type:    IntegratedGpu

Running benchmarks...

─────────────────────────────────────────────────────────────────
  MatMul [ 512x 512x 512]:     5.004 ms  │     53.64 GFLOPS
  MatMul [1024x1024x1024]:    19.637 ms  │    109.36 GFLOPS
  MatMul [2048x2048x2048]:   150.242 ms  │    114.35 GFLOPS
  MatMul [4096x4096x4096]:  1138.546 ms  │    120.71 GFLOPS
─────────────────────────────────────────────────────────────────
```

## Performance Analysis

### MatMul

| Size | Time (ms) | GFLOPS | Notes |
|------|-----------|--------|-------|
| 512³ | 5.0 | 54 | Kernel launch overhead |
| 1024³ | 19.6 | 109 | Sweet spot |
| 2048³ | 150.2 | 114 | Compute bound |
| 4096³ | 1138.5 | **121** | Peak performance |

### Key Observations

1. **GPU acceleration verified** - 120 GFLOPS on Apple M2
2. **Custom WGSL kernels** — Compiled and running on Metal (core ops verified)
3. MatMul benchmarks are reproducible on tested hardware — see `benchmarks/RESULTS.md`

## Practical Interpretation

- The most dependable current use case is lightweight LoRA SFT / continued LM for small models on ~8GB consumer GPUs (see README **Fine-tuning Models** / `rusty-cli train --lite`).
- Benchmark results here do not yet reflect full training costs such as model loading, optimizer state, padding, checkpointing, or CPU/GPU synchronization overhead.
- When comparing against other frameworks, prefer end-to-end fine-tuning measurements on the same model and dataset, not kernel GFLOPS alone.

### Lite training recipe (not a FLOPS number)

```bash
cargo run -p rusty-cli --release -- train --lite --task sft \
  --model ./models/tinyllama --data ./data/train.jsonl --out ./outputs/run1
```

### M2 Theoretical Comparison

- M2 GPU: ~3.6 TFLOPS FP32 peak
- Rusty achieving: ~121 GFLOPS
- Utilization: ~3.4% (naive matmul, no tiling optimization)

**Note:** Current implementation uses 8x8 tiles. Moving to 16x16 or 32x32 tiles would significantly improve utilization.

## Run Benchmarks

```bash
cd /path/to/rusty
cargo run -p rusty-benchmarks --release
```

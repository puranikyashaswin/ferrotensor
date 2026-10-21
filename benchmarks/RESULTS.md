# Ferrotensor GPU kernel benchmarks

Kernel-level GPU timings for `ferrotensor-backend`. They are not end-to-end training
throughput: model loading, optimizer state, host/device copies and
synchronisation are not included.

## Setup

- Hardware: Apple M2 (10-core GPU), 8 GB unified memory, MacBook Air
- OS: macOS 27.2; backend: Metal via wgpu 0.19
- Toolchain: rustc 1.99.1, `--release`
- Command: `cargo run -p ferrotensor-benchmarks --release`
- Method: each case is dispatched in batches (about 20 ms per sample, at most
  64 calls) followed by one GPU wait; the table shows the median per-call time.
  "Before" and "after" binaries were run alternately three times each, and the
  best median of the three is reported, because other processes were using the
  GPU and single runs varied by up to 2x.
- FLOP conventions: matmul `2*M*N*K`; causal attention forward
  `2*B*H*S^2*D` (half of `4*B*H*S^2*D`), backward 2.5x the forward.
  Row kernels and element-wise ops report bytes read plus written per second.

## Before / after

"Before" is the commit "Rewrite the benchmark suite as a library with bin and
bench targets": the original matmul, softmax and RMSNorm kernels, and the
first correct flash attention kernels. The original flash attention computed
wrong results, so it has no meaningful timing. "After" is the commit
"Document every public item in ferrotensor-graph", which has every kernel change
below.

| kernel | shape | before | after | speed-up |
|---|---|---:|---:|---:|
| matmul | 512x512x512 | 2.47 ms (109 GFLOPS) | 0.65 ms (414 GFLOPS) | 3.8x |
| matmul | 1024x1024x1024 | 18.25 ms (118 GFLOPS) | 3.83 ms (561 GFLOPS) | 4.8x |
| matmul | 2048x2048x2048 | 141.06 ms (122 GFLOPS) | 28.89 ms (595 GFLOPS) | 4.9x |
| matmul | 4096x4096x4096 | 1114.32 ms (123 GFLOPS) | 225.09 ms (611 GFLOPS) | 5.0x |
| matmul | 512x2048x8192 | 142.88 ms (120 GFLOPS) | 28.91 ms (594 GFLOPS) | 4.9x |
| matmul | 1000x1000x1000 | 18.30 ms (109 GFLOPS) | 3.83 ms (523 GFLOPS) | 4.8x |
| matmul_bt | 1024x1024x1024 | 47.14 ms (46 GFLOPS) | 3.82 ms (562 GFLOPS) | 12.3x |
| matmul_at | 1024x1024x1024 | 15.26 ms (141 GFLOPS) | 3.82 ms (562 GFLOPS) | 4.0x |
| attention_fwd | b1 h16 s512 d64 causal | 31.93 ms (17 GFLOPS) | 10.68 ms (50 GFLOPS) | 3.0x |
| attention_fwd | b1 h16 s1024 d64 causal | 103.34 ms (21 GFLOPS) | 27.40 ms (78 GFLOPS) | 3.8x |
| attention_fwd | b1 h16 s2048 d64 causal | 370.32 ms (23 GFLOPS) | 92.66 ms (93 GFLOPS) | 4.0x |
| attention_fwd | b1 h8 s1024 d128 causal | 132.12 ms (16 GFLOPS) | 42.55 ms (50 GFLOPS) | 3.1x |
| attention_bwd | b1 h16 s512 d64 causal | 161.04 ms (8 GFLOPS) | 39.69 ms (34 GFLOPS) | 4.1x |
| attention_bwd | b1 h16 s1024 d64 causal | 581.93 ms (9 GFLOPS) | 130.62 ms (41 GFLOPS) | 4.5x |
| softmax | 4096x1024 | 2.31 ms (14 GB/s) | 0.45 ms (74 GB/s) | 5.1x |
| rms_norm | 2048x4096 | 3.86 ms (17 GB/s) | 0.87 ms (77 GB/s) | 4.5x |
| add | 4M | 0.66 ms (72 GB/s) | 0.62 ms (78 GB/s) | 1.1x |
| silu | 4M | 0.45 ms (72 GB/s) | 0.48 ms (66 GB/s) | 0.9x |

Element-wise ops (`add`, `silu`) are memory-bound and unchanged; the
differences are noise.

## What changed

- **matmul / matmul_bt / matmul_at**: one register-blocked kernel. Each 16x16
  workgroup computes a 64x64 output tile, each thread 4x4 outputs held in
  `vec4` registers, with K staged through workgroup memory in slices of 16.
  `matmul_bt` was previously an untiled loop.
- **attention forward**: K/V tiles, query and accumulator as `vec4`, 128
  queries per workgroup.
- **attention backward**: preprocess fused into the dQ pass; K/V and Q/dO
  staged as `vec4` tiles; key/query ranges outside the causal/window band are
  skipped per workgroup.
- **softmax / RMSNorm**: one 256-thread workgroup per row with a tree reduction
  instead of one thread per row.

## Tried and not kept

Under the same best-of-3 timing, these gave no measurable gain:

- matmul with 8x4, 4x8 or 8x8 outputs per thread (`vec4` registers):
  550-620 GFLOPS, within noise of 4x4. With plain arrays instead of `vec4`,
  8x8 fell to about 90 GFLOPS (register spills).
- matmul K slice of 32 instead of 16: slower (about 370 GFLOPS at 4096^3).
- Coalescing the A/B tile loads along K: no change beyond noise.
- Attention forward with 256 queries per workgroup: faster at s512 but about
  30% slower at s1024 and s2048.

## Context

The M2 GPU peaks at about 3.6 TFLOPS FP32, so matmul reaches about 17% of
peak. Apple's MPS reaches a much larger fraction; closing that gap needs
`simdgroup_matrix` operations, which WGSL does not expose.

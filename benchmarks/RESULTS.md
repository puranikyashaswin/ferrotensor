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

# `ferrotensor` Tensor API: CPU and GPU devices

End-to-end timings through the `ferrotensor` crate's `Tensor` API, including
the host-side cost of every op (autograd bookkeeping, buffers, bind groups,
submission) and a read back to the host at the end, as user code does.

## Setup

- Hardware: Apple M2 (10-core GPU), 8 GB unified memory, MacBook Air
- OS: macOS 27.2; backend: Metal via wgpu 0.19
- Toolchain: rustc 1.99.1, `--release`
- Command: `cargo run -p ferrotensor-benchmarks --release --bin tensor-api`
- Method: three warm-up iterations, then at least 10 timed iterations (up to
  200 or 2 s); each case reports the median. "Before" and "after" binaries
  were run alternately three times each; the table shows the best median of
  the three.
- "Before" is commit "Add a Tensor API benchmark for CPU and GPU devices"
  (one submission per op); "after" is commit "Gather embedding rows on the
  GPU without reading the ids back".
- The benchmark prints a checksum of each case's output bits. On every case
  the GPU checksum is the same before and after, so the GPU results are bit
  for bit unchanged.

## Before / after

| case | CPU | GPU before | GPU after | GPU speed-up |
|---|---:|---:|---:|---:|
| 100 element-wise ops on `[64]`, then `to_vec` | 0.046 ms | 10.63 ms | 0.82 ms | 13x |
| 100 element-wise ops on `[4096]`, then `to_vec` | 1.38 ms | 10.73 ms | 0.84 ms | 12.8x |
| MLP 784-256-10, one AdamW step, batch 64, then `loss.item()` | 12.45 ms | 10.80 ms | 3.98 ms | 2.7x |
| matmul 512x512x512, then `to_vec` | 10.30 ms | 5.22 ms | 2.77 ms | 1.9x |

The CPU column is the best of all six runs; the CPU path did not change.

The examples, timed with `/usr/bin/time -p` around
`FERROTENSOR_DEVICE=cpu|gpu cargo run --release -p ferrotensor --example <name>`
(binaries built once, then run directly). Their weights are not seeded, so
the final losses differ from run to run.

| example | CPU | GPU before | GPU after |
|---|---:|---:|---:|
| `char_transformer` (600 steps, "trained 600 steps in") | 42.5 s (44.4 s with the before binary) | 105.0 s | 31.9 s |
| `mlp_two_moons` (60 epochs, wall time, two runs) | 0.19-0.24 s | 12.46 s, 16.60 s | 2.21 s, 1.80 s |

## What changed

- **Batched submission.** Ops are recorded and submitted together at the next
  read back, host write, or every 256 ops, instead of one submit per op
  (a submit costs about 117 µs of CPU on this machine).
- **One compute pass per batch.** On Metal each compute pass costs about
  40 µs of GPU time; consecutive dispatches in one pass cost about 13 µs each
  in a 100-dispatch micro test.
- **Op arguments in one buffer.** Uniforms and shape metadata go into one
  buffer written once per batch, instead of a new buffer per op. Bind group
  layouts are cached.
- **Buffer pool.** Op outputs reuse freed tensor buffers of the same size (up
  to 128 MiB per device), which also skips wgpu's zero fill of new buffers.
  This took the 100-op chain on `[4096]` from 1.77 ms to 0.90 ms and the MLP
  step from 5.1 ms to 4.0 ms in single runs.
- **Spin-wait for read backs.** wgpu 0.19's blocking wait on Metal sleeps in
  1 ms steps (`wgpu-hal` `metal/device.rs`, `Device::wait`), so every
  `to_vec`/`item` cost at least about 1.3 ms. Waits now busy-poll for up to
  10 ms before blocking; a read back after the GPU is idle takes about
  0.18 ms.

## What is still slow

- 100 tiny ops on the GPU still take about 18x as long as on the CPU: most of
  it is GPU time per dispatch (about 9-13 µs each, serialised within the
  pass), plus one round trip to read the result. Fusing element-wise chains
  into one kernel would be the next step.
- Several ops read back to the host on their own and force a sync:
  `cross_entropy` with class-index targets, `argmax`/`argmin`, `cat`, the
  backward passes of `index_select` and `Embedding`, and gradient clipping
  (one `item()` per parameter).
- Full reductions (`sum()` over a whole tensor) use one GPU thread per
  output, so a large `sum()` runs on a single thread.

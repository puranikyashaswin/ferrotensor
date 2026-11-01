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

## Full reductions and argmax on the GPU

Same machine, toolchain and method as above (best median of three
alternating runs of each binary), with
`cargo run -p ferrotensor-benchmarks --release --bin tensor-api`. "Before" is
commit "Benchmark full sum and argmax in tensor-api"; "after" is commit "Run
argmax and argmin on the GPU without reading the tensor back". Inputs are
seeded `randn`.

| case | CPU | GPU before | GPU after | GPU speed-up |
|---|---:|---:|---:|---:|
| `sum()` of `[4096, 4096]`, then `item` | 18.27 ms | 2169.35 ms | 1.19 ms | 1820x |
| `sum()` of `[1000000]`, then `item` | 0.96 ms | 129.20 ms | 0.35 ms | 374x |
| `argmax_dim(-1)` of `[1024, 4096]`, then `to_vec` | 5.66 ms | 8.40 ms | 0.99 ms | 8.5x |
| `argmax()` of `[4096, 4096]` | 17.25 ms | 27.69 ms | 1.19 ms | 23x |

The CPU column is the best of all six runs; the CPU path did not change.

- `sum()`, `max()` and `min()` over a whole tensor used one GPU thread. The
  `reduce_all` kernel now spreads the first pass over 256 workgroups of 256
  threads (1024 workgroups measured 1.23 ms and 0.39 ms on the two `sum()`
  cases in a single run, so 256 stays) and a one-workgroup pass adds the 256 partials.
- The GPU now adds in a different order from the CPU's left-to-right loop,
  so the GPU `sum()` checksums changed; the other cases' checksums are the
  same before and after, so `argmax` gives the same indices. Against an
  `f64` sum of 16.7M uniform values in `[-2, 2)` (printed from
  `full_reductions_match_cpu_on_gpu`), the GPU's error is 4.5e-11 of
  `sum(|x|)` and the CPU's 5.9e-9.
- `argmax_dim` read the whole tensor back to the host; `argmax()`/`argmin()`
  did too. The `arg_reduce` kernel keeps them on the GPU; `argmax()` and
  `argmin()` read back only the index.

## Index backward passes, `cat` and `cross_entropy` on the GPU

Same machine, toolchain and method as above (best median of three
alternating runs of each binary), with
`cargo run -p ferrotensor-benchmarks --release --bin tensor-api`. "Before" is
commit "Benchmark embedding, index_select, cat and cross_entropy in
tensor-api"; "after" is commit "Compute class-id cross_entropy on the GPU".
Inputs are seeded `randn`; each case ends with a read back.

| case | CPU before | CPU after | GPU before | GPU after | GPU speed-up |
|---|---:|---:|---:|---:|---:|
| `Embedding` `[32000, 256]`, 4096 ids, forward + backward, read the gradient | 7.10 ms | 7.08 ms | 13.83 ms | 8.89 ms | 1.6x |
| `index_select(0, 1024 ids)` of `[2048, 2048]`, backward | 8.75 ms | 8.74 ms | 8.68 ms | 5.04 ms | 1.7x |
| `index_select(1, 1024 ids)` of `[2048, 2048]`, backward | 27.29 ms | 26.58 ms | 14.16 ms | 7.91 ms | 1.8x |
| `cat` of 8 `[512, 512]` along dim 0 | 0.55 ms | 0.56 ms | 5.58 ms | 1.50 ms | 3.7x |
| `cat` of 8 `[512, 512]` along dim 1 | 0.58 ms | 0.60 ms | 5.58 ms | 1.48 ms | 3.8x |
| `cross_entropy` `[4096, 32000]`, class ids, forward + backward | 4226.86 ms | 517.16 ms | 749.64 ms | 64.58 ms | 11.6x |
| MLP 784-256-10 AdamW step, batch 64 | 12.16 ms | 12.26 ms | 3.69 ms | 2.13 ms | 1.7x |

- **Index backward passes.** The new `scatter_rows` kernel gives each
  output element a thread that adds its row's incoming gradient rows. It
  reads them through `(row, position)` pairs sorted by row, then position,
  and a table of where each row's pairs start. `index_select` sorts its
  indices on the host, where they already are. `Embedding` keeps its ids on
  the device and sorts them there with a bitonic sort (`sort_ids`, one
  dispatch per compare-and-swap step: 78 for 4096 ids). WGSL has no float
  atomics; adding with a compare-and-swap loop on `u32` bits would make the
  order of the additions, and so the rounding, change from run to run. The
  sorted version adds in position order, as the CPU loop does, so the
  gradients are deterministic and match the CPU bit for bit (same checksums
  as before, and `scatter_backward_matches_cpu_on_gpu` compares them with
  `assert_eq!`). The `Embedding` case is still slower on the GPU than on the
  CPU because it reads back the whole 32.8 MB gradient.
- **`cat`** read every input back and uploaded the result; the
  `copy_block` kernel now copies each input into its slice of the output on
  the device. Same checksums: the values are copied exactly.
- **`cross_entropy` with class ids** read the targets back and built a
  dense `[rows, classes]` weight tensor on the host (524 MB here), then took
  `log_softmax`, a product and a sum. The `cross_entropy_rows` kernel now
  gives each row a workgroup of 256 threads that writes the row's loss
  (log-sum-exp minus the target logit); `cross_entropy_grad` writes softmax
  minus one-hot times the row's incoming gradient. The CPU path uses the same
  per-row reference, which is why it got faster too. The "before" runs
  peaked at 3.1 GB resident (`/usr/bin/time -l`) on this 8 GB machine and
  their medians varied (CPU 4.2 s to 8.8 s, GPU 0.75 s to 1.17 s across the
  three runs); the "after" runs peaked at 1.6 GB and varied by under 4%.
  The loss is now computed in a different order, so the `cross_entropy` and
  MLP checksums changed; `cross_entropy_matches_cpu_on_gpu` checks the GPU
  against the CPU to 1e-5 relative on the loss and 1e-6 absolute on the
  gradient, up to `[64, 32000]`.
- The other cases' checksums are the same before and after, and their times
  moved by at most 11% (`sum()` of `[4096, 4096]` on the GPU, 1.08 to 1.20 ms;
  that op did not change).

## What is still slow

- 100 tiny ops on `[64]` on the GPU still take about 17x as long as on the
  CPU (0.82 ms against 0.049 ms in the runs above): most of it is GPU time
  per dispatch (about 9-13 µs each, serialised within the pass), plus one
  round trip to read the result. Fusing element-wise chains
  into one kernel would be the next step.
- `F::nll_loss` with class ids still reads tensor targets back and builds a
  dense weight tensor on the host, as `cross_entropy` did. `cross_entropy`
  with soft labels uses `log_softmax`, whose kernel gives each row one
  thread: slow on wide rows such as a 32000-token vocabulary.
- The `Embedding` sort uses one dispatch per bitonic step (78 for 4096 ids,
  136 for 65536); sorting blocks of 512 in workgroup memory would cut that
  to a handful.
- `argmax_dim` gives each output row a workgroup of 256 threads, so on a
  short axis (10 classes) most threads idle.

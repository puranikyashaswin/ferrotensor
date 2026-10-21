# rusty-benchmarks

GPU kernel benchmarks for `rusty-backend`: matmul (square, LLM-shaped, odd
sizes and the two transposed variants used in backward passes), flash
attention forward and backward, softmax, RMSNorm and element-wise ops.

## Run

```bash
cargo run -p rusty-benchmarks --release
```

`cargo bench -p rusty-benchmarks` runs the same suite. Options go after `--`:

```bash
cargo run -p rusty-benchmarks --release -- --quick            # smaller sizes, ~8 s
cargo run -p rusty-benchmarks --release -- --filter attention  # cases whose name contains "attention"
```

A full run takes about 15 s on an Apple M2 and allocates at most a few hundred
MB of GPU memory. The output is a Markdown table:

```text
GPU: Apple M2 (Metal, IntegratedGpu)
Mode: full

| kernel | shape | median time | throughput |
|---|---|---:|---:|
| matmul | 4096x4096x4096 | 225.090 ms | 611.1 GFLOPS |
...
```

## How it measures

Each case first runs 3 warm-up calls. It then dispatches the kernel in batches
of up to 64 calls (about 20 ms of work per batch) and waits for the GPU once
per batch, so per-call CPU overhead overlaps with GPU work. The reported time
is the median per-call time over 5–30 batches. Throughput uses `2*M*N*K` FLOPs
for matmul, `2*B*H*S^2*D` for causal attention forward (2.5x that for the
backward pass), and bytes read plus written for the row and element-wise
kernels.

Timings are wall-clock, so other GPU users on the machine add noise. When
comparing two versions, alternate the binaries over several runs and compare
the best result of each.

## Results

See [RESULTS.md](RESULTS.md) for recorded numbers and the hardware they came
from.

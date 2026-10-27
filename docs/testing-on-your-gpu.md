# Testing Ferrotensor on your GPU

Ferrotensor runs on any GPU wgpu supports: Metal on macOS, Vulkan on Linux and
Windows, DX12 on Windows. So far every number in this repository was measured
on one machine, an Apple M2 through Metal. Results from AMD, Intel and NVIDIA
GPUs, and from Vulkan or DX12, are the most useful thing you can contribute.

## 1. See which GPU Ferrotensor uses

```bash
cargo install --git https://github.com/puranikyashaswin/ferrotensor ferrotensor-cli
ferrotensor info
```

`info` is not in a crates.io release yet, so this installs the CLI from the
repository's `main` branch.

This lists every adapter wgpu can open on your machine and marks the one
Ferrotensor picks with `*`. If your discrete GPU is not the one marked, say so
in your report: wgpu chooses the high-performance adapter it finds first.

## 2. Check that the kernels are correct on it

```bash
ferrotensor info --check
```

This runs matmul (at sizes that are not multiples of any tile size), softmax
and RMSNorm on the selected GPU and compares each with a CPU reference. Each
line says `ok` or `FAIL` with the largest error and the tolerance. The command
exits with an error if any kernel fails.

## 3. Run the tests and benchmarks (optional)

From a checkout of the repository:

```bash
FERROTENSOR_RUN_GPU_TESTS=1 cargo test --workspace -- --ignored --test-threads=1
cargo run -p ferrotensor-benchmarks --release --bin tensor-api
```

The first command runs every kernel against its CPU reference on your GPU.
The second prints CPU and GPU timings for the Tensor API.

## 4. Report what you found

Open an issue with the output of:

```bash
ferrotensor info --check --json
```

Whether everything passed or something failed, the report tells us which
hardware, driver and backend Ferrotensor has now been tried on. If you also ran
the tests or the benchmark, paste their output too.

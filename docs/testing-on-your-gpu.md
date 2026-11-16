# Testing Ferrotensor on your GPU

Ferrotensor runs through wgpu: Metal on macOS, Vulkan on Linux and Windows,
DX12 on Windows. It has been tested on an NVIDIA A10G (24 GB) through Vulkan,
an Apple M2 (8 GB) through Metal, and software Vulkan (Mesa lavapipe) in CI;
the kernel benchmarks also ran on a Tesla T4 through Vulkan
([results](../benchmarks/RESULTS.md)). The GPU test suite also runs in CI
through Metal on GitHub's Apple silicon runners (a virtualised GPU, reported
as `Apple Paravirtual device`) and through DX12 on WARP (`Microsoft Basic
Render Driver`, Microsoft's software adapter). On WARP the CLI end-to-end
tests are skipped because they take too long there, and the rest of the suite
(2 h 22 min) runs on `main`, weekly and on demand; a pull request runs only
the backend kernel tests and the Tensor API parity tests (forward values and
gradients against the CPU) there. These runs check
correctness on the two backends; they are not real-hardware results and say
nothing about speed. Other GPUs are untested. Results from AMD and Intel
GPUs, and from DX12 on a real GPU, are the most useful thing you can
contribute.

The full DX12 run is not a merge gate and does not fail the workflow, because
it is not reliably green. Its workspace GPU suite passed once (16 November
2026, before a merge of `main`). In the next full run every test passed except
`block_checkpointing` (6 of 7) and `no_grad_inference` (2 of 3) in
`ferrotensor-trainer`: in each the second test's readback panicked with
`Error in Buffer::get_mapped_range: Validation Error` (`Buffer is not
mapped`), and every later test in that binary failed with `no GPU adapter
found for backends dx12`. The cause is not known; the readback ignores the
map error, so the log does not show why mapping failed. A rerun of that job
had not finished when this was written.

On 16 November 2026 one A10G (24 GB) ran `info --check` (passed), the gradient
parity tests (4 passed, worst absolute difference 9.239e-7) and the forward
parity tests (10 passed), then the commands `plan`, `finetune`, `chat`,
`merge`, `export-gguf` and `pretrain`; the numbers are in
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md#real-hardware-run-16-november-2026).
The full GPU test suite did not finish in that run: it stopped at the Qwen2
tests, which need the fixture described in section 3 (they now skip when it
is missing).

## 1. See which GPU Ferrotensor uses

```bash
cargo install ferrotensor-cli
ferrotensor info
```

`info` is in `ferrotensor-cli` 0.4.0 and later.

This lists every adapter wgpu can open on your machine and marks the one
Ferrotensor picks with `*`. If your discrete GPU is not the one marked, say so
in your report: wgpu chooses the high-performance adapter it finds first.

To try one backend only, set `WGPU_BACKEND` (`vulkan`, `metal`, `dx12` or
`gl`, or a comma list):

```bash
WGPU_BACKEND=vulkan ferrotensor info --check
```

Ferrotensor's tests and the CLI respect it too. On Windows this is the way to
compare Vulkan with DX12 on the same GPU. On DX12, `WGPU_DX12_COMPILER=dxc`
builds the kernels with DXC instead of the default FXC; it needs
`dxcompiler.dll` and `dxil.dll` (from the Windows SDK) on the `PATH`, and
without them wgpu falls back to FXC with a warning. With FXC, opening the GPU
took about 9 minutes on the software adapter WARP in CI.

If the CLI says `no GPU adapter found for backends ...`, wgpu saw no GPU on
those backends: update the GPU driver (on Linux, install the Vulkan driver for
your GPU) or set `WGPU_BACKEND` to another backend. If it says
`GPU adapter ... refused to create a device`, the driver rejected the request;
the text after the colon is the driver's reason, and it belongs in your report.

## 2. Check that the kernels are correct on it

```bash
ferrotensor info --check
```

This runs matmul (at sizes that are not multiples of any tile size), softmax
and RMSNorm on the selected GPU and compares each with a CPU reference. Each
line says `ok` or `FAIL` with the largest error and the tolerance. The command
exits with 1 if any kernel fails and with 2 if no GPU adapter is found.

## 3. Run the tests and benchmarks (optional)

From a checkout of the repository:

```bash
FERROTENSOR_RUN_GPU_TESTS=1 cargo test --workspace -- --ignored --test-threads=1
cargo run -p ferrotensor-benchmarks --release --bin tensor-api
```

The first command runs the GPU test suites on your GPU, which compare the kernels with CPU references and the LoRA gradients with PyTorch fixtures.
The second prints CPU and GPU timings for the Tensor API.

### The Qwen2 fixture (four tests)

The four tests in `ferrotensor-trainer/tests/qwen2_family.rs` compare a tiny
random Qwen2 model with Hugging Face transformers. They need a fixture that
`scripts/gen_qwen2_fixture.py` writes, which needs PyTorch and Transformers
(`pip install torch transformers safetensors`):

```bash
python scripts/gen_qwen2_fixture.py /tmp/qwen2-fixture
FERROTENSOR_QWEN2_FIXTURE=/tmp/qwen2-fixture \
  cargo test --release -p ferrotensor-trainer --test qwen2_family -- --ignored --test-threads=1
```

Without `FERROTENSOR_QWEN2_FIXTURE` the four tests print a skip message and
pass, so the rest of the suite still runs. A path that is set but wrong makes
them fail. CI generates the fixture and sets the variable.

## 4. Or measure everything in one go

From a checkout of the repository:

```bash
scripts/hardware-report.sh           # add --quick for a short run
```

This builds in release mode, runs `ferrotensor info --check`, the kernel
benchmarks and the Tensor API benchmarks, and writes one JSON report (plus the
raw tables) to `reports/`. Attach the JSON to your issue: it is the same
measurement every other number in this repository comes from.

## 5. Report what you found

Open an issue with the output of:

```bash
ferrotensor info --check --json
```

Whether everything passed or something failed, the report records which
hardware, driver and backend Ferrotensor has now been tried on. If you also ran
the tests or the benchmark, paste their output too.

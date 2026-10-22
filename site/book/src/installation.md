# Installation

Ferrotensor 0.2.0 is published to crates.io and PyPI.

## Rust library

You need Rust 1.88 or newer (`rustup` is enough). On macOS no other setup is
needed; on Windows and Linux a Vulkan or DX12 driver gives you the GPU.

```bash
cargo add ferrotensor
```

Then, in your code:

```rust,ignore
use ferrotensor::prelude::*;

let device = Device::best(); // the GPU when there is one, else the CPU
let x = Tensor::from_data(&[1., 2., 3., 4.], &[2, 2], &device);
```

No GPU? `Device::cpu()` runs the same code.

## Command-line tool

The `ferrotensor` command fine-tunes small Llama-style models with LoRA and
runs `train`, `eval` and `generate`:

```bash
cargo install ferrotensor-cli
```

Prebuilt binaries are attached to each
[GitHub release](https://github.com/puranikyashaswin/ferrotensor/releases).
The [LoRA walkthrough](examples/lora-sft.md) goes through a full run.

## Python

```bash
pip install ferrotensor
```

Wheels are built for macOS arm64 and Linux x86_64 as abi3 wheels for
CPython 3.10 and newer. On other platforms pip builds from the source
distribution, which needs a Rust toolchain. Continue with the
[Python quickstart](python.md).

## From a checkout

```bash
git clone https://github.com/puranikyashaswin/ferrotensor.git
cd ferrotensor
cargo run --release -p ferrotensor --example quickstart
```

Output on an Apple M2 (the device line changes with your hardware):

```text
device: Apple M2 (Metal)
x @ w = [7.5, 2.0, 18.0, 2.0] (shape [2, 2])
dL/dw = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
```

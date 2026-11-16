# Installation

Ferrotensor {{ft_version}} is published to crates.io and PyPI.

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

The `ferrotensor` command fine-tunes small Llama-style models with LoRA. It
checks the GPU (`info --check`), plans what fits (`plan`), trains (`finetune`,
`train`), evaluates and samples (`eval`, `generate`), chats (`chat`), merges an
adapter into the base weights (`merge`), writes a GGUF file (`export-gguf`) and
trains a small byte-level model from scratch (`pretrain`):

```bash
cargo install ferrotensor-cli
ferrotensor info --check
```

`--model` takes a model directory or a Hugging Face id already in the local
cache; nothing is downloaded. `plan`, `finetune`, `chat`, `merge`,
`export-gguf` and `pretrain` are tested in CI, on software Vulkan where they
use the GPU. `export-gguf` writes GGUF v3, and loading the file in llama.cpp
is not yet checked. Most commands take `--json` and print one JSON object, and
every command exits with 0 on success, 1 on a runtime error and 2 when the
input is invalid or the machine cannot meet the request;
[AGENTS.md](https://github.com/puranikyashaswin/ferrotensor/blob/main/AGENTS.md)
lists both.

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

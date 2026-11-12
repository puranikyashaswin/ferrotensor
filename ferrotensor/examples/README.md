# Examples for the `ferrotensor` crate

Every example runs on the CPU by default. Set `FERROTENSOR_DEVICE=gpu` (where
the example says so) to use the GPU. Run one with:

```bash
cargo run --release -p ferrotensor --example <name>
```

| Example | What it shows |
|---------|---------------|
| `quickstart` | A matmul and its gradient on the best available device |
| `mlp_two_moons` | An MLP trained with `DataLoader` on a synthetic dataset |
| `char_transformer` | A character-level transformer trained from scratch on a few sonnets, then sampled |
| `gpu_matmul_attention` | The same matmul and attention on CPU and GPU, checked against each other and timed |
| `similarity_search` | Nearest-neighbour search over hashed character-trigram embeddings |

Real output for each is in [docs/use-cases.md](../../docs/use-cases.md).

The low-level examples for the LLM stack live next to the crates they use:

- [`ferrotensor-backend/examples`](../../ferrotensor-backend/examples): context
  creation, tensor upload, and individual kernels through `ComputeEngine`
- [`ferrotensor-graph/examples`](../../ferrotensor-graph/examples): a Llama
  block and causal flash attention checked against a CPU reference
- [`ferrotensor-trainer/examples`](../../ferrotensor-trainer/examples): the
  training stack on a tiny random model built in a temporary directory, so
  nothing is downloaded. Each needs a GPU, checks its own results and runs
  in CI:

  | Example | What it shows |
  |---------|---------------|
  | `lora_finetune_tiny` | LoRA training; the loss falls, and a saved adapter reloads into a fresh model with the same loss |
  | `resume_training` | A run stopped halfway, saved with its optimizer state and resumed, matches an unbroken run |
  | `generate_with_kv_cache` | Greedy generation gives the same tokens with and without the KV cache |

  Run one with `cargo run --release -p ferrotensor-trainer --example <name>`.

For fine-tuning a real model, see the
[LoRA walkthrough](../../docs/examples/lora-sft.md).

## Adding an example

Create a `.rs` file in this directory; Cargo finds it. Put a doc comment at
the top with the command to run it and its real output.

## Troubleshooting

`Device::gpu()` panics or returns an error when wgpu finds no adapter (a
machine without Metal or Vulkan, or a CI runner). Use `Device::cpu()`.

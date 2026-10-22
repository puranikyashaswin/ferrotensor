# ferrotensor (Python)

Python bindings for [ferrotensor](https://github.com/puranikyashaswin/ferrotensor),
a pure-Rust ML framework: PyTorch-style tensors, autograd, `nn` layers and
optimizers that run on the CPU or on a GPU through wgpu (Metal, Vulkan, DX12)
with hand-written WGSL kernels. No CUDA, no NumPy, no other runtime
dependencies.

## Install

Wheels are built for macOS arm64 and Linux x86_64 (manylinux), as abi3
wheels that work on CPython 3.10 and newer:

```bash
pip install ferrotensor
```

On other platforms pip builds from the source distribution, which needs a
Rust toolchain. To build from a checkout of the repository:

```bash
pip install ./ferrotensor-py
```

For development, `maturin develop` installs an editable build into the
active virtual environment:

```bash
cd ferrotensor-py
pip install maturin pytest
maturin develop
pytest
```

## Quickstart

```python
import ferrotensor as ft
import ferrotensor.nn.functional as F
from ferrotensor import nn, optim

# Tensors and autograd
x = ft.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = (x @ x).relu().sum()
y.backward()
print(x.grad)  # tensor([[7.0, 11.0], [9.0, 13.0]])

# A model, a loss and an optimizer: learn XOR
model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
opt = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
inputs = ft.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = [0, 1, 1, 0]
for step in range(200):
    opt.zero_grad()
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()
    opt.step()
print(loss.item() < 0.05)  # True
print(model(inputs).argmax(dim=1).tolist())  # [0.0, 1.0, 1.0, 0.0]

# Save and load parameters (safetensors)
model.save("xor.safetensors")
model.load("xor.safetensors")
```

`examples/mlp.py` trains a small MLP on two spirals with a warmup + cosine
learning-rate schedule, then saves and reloads it:

```bash
python examples/mlp.py
python examples/mlp.py --device gpu
```

## Devices

Tensors and layers take a `device=` argument: a `ft.Device`, `"cpu"`,
`"gpu"` or `"best"` (the GPU when there is one). The default is the CPU.

```python
import ferrotensor as ft

print(ft.Device.gpu_available())
dev = ft.Device("best")
a = ft.randn(256, 256, device=dev)
b = a.to("cpu")
```

`ft.Device("gpu")` raises `RuntimeError` when wgpu finds no adapter. Tensors
on different devices cannot be mixed in one op (`ValueError`); move them
with `.to(device)`.

## What is covered

- `Tensor`: creation (`tensor`, `zeros`, `ones`, `full`, `randn`, `rand`,
  `arange`, `linspace`, `eye`, `*_like`), operators `+ - * / ** @` and
  comparisons with broadcasting, int/slice indexing, reductions (`sum`,
  `mean`, `max`, `min`, `var`, `argmax`, `argmin`), shape ops (`reshape`,
  `view`, `flatten`, `squeeze`, `unsqueeze`, `permute`, `transpose`, `T`,
  `expand`, `narrow`, `chunk`, `split`, `index_select`), `cat`, `stack`,
  `where`, element-wise math and activations, `softmax`, `log_softmax`.
- Autograd: `requires_grad`, `backward()`, `.grad`, `detach()`, `no_grad`
  (context manager or decorator).
- `nn`: `Module`, `Sequential`, `ModuleList`, `Linear`, `Embedding`,
  `LayerNorm`, `RMSNorm`, `Dropout`, `MultiHeadAttention`, `ReLU`,
  `LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `Flatten`,
  `Identity`, `state_dict()` / `load_state_dict()`.
- `nn.functional`: `mse_loss`, `l1_loss`, `smooth_l1_loss`,
  `cross_entropy` (class indices, with negative indices ignored, or
  probabilities), `nll_loss`, `binary_cross_entropy(_with_logits)`,
  `layer_norm`, `rms_norm`, `linear`, `dropout`,
  `scaled_dot_product_attention`, `one_hot`, `accuracy`,
  `cosine_similarity`.
- `optim`: `SGD` (momentum, Nesterov, weight decay), `AdamW`,
  `clip_grad_norm`, `clip_grad_value`, and the schedules `ConstantLR`,
  `StepLR`, `CosineAnnealingLR`, `WarmupCosineLR`, `WarmupLinearLR`.
- `save` / `load`: dicts of tensors as safetensors (loads F32, F16, BF16 and
  F64 as float32).

Bad arguments raise `ValueError`, `IndexError` or `TypeError` with a message
naming the shapes involved; the Rust code never panics across the boundary.

## Differences from PyTorch

- Every tensor is float32. Index tensors (`argmax`, class targets) hold
  integral float values.
- `nn.Linear.weight` has shape `[in_features, out_features]`: the layer
  computes `x @ weight + bias`.
- `Tensor.max(dim)` / `min(dim)` return values only; use `argmax(dim)` for
  indices. `var()` is the population variance (divides by n).
- `==` compares identity (so tensors stay hashable); use `a.eq(b)` for an
  element-wise comparison.
- Schedulers set `optimizer.lr` for step 0 when created; call
  `scheduler.step()` after each `optimizer.step()`.

## Tests

```bash
pytest                              # CPU
FERROTENSOR_GPU_TESTS=1 pytest      # also GPU parity tests
```

## License

MIT or Apache-2.0, at your option.

# Python quickstart

The `ferrotensor` Python package binds the Rust tensor API: PyTorch-style
tensors, autograd, `nn`, `optim` and safetensors, on the CPU or the GPU. It
has no runtime dependencies (no NumPy).

```bash
pip install ferrotensor
```

## Learn XOR

```python
import ferrotensor as ft
from ferrotensor import nn, optim
import ferrotensor.nn.functional as F

model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
opt = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
x = ft.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
for _ in range(200):
    opt.zero_grad()
    loss = F.cross_entropy(model(x), [0, 1, 1, 0])  # XOR
    loss.backward()
    opt.step()
print(model(x).argmax(dim=1).tolist())  # [0.0, 1.0, 1.0, 0.0]
```

## Use the GPU

Tensors and layers take a `device=` argument: `"cpu"` (the default),
`"gpu"`, `"best"` or an `ft.Device`.

```python
import ferrotensor as ft

print(ft.Device.gpu_available())
dev = ft.Device("best")
a = ft.randn(256, 256, device=dev)
```

## A complete example

`ferrotensor-py/examples/mlp.py` trains an MLP on two spirals with a
warmup + cosine learning-rate schedule, then saves and reloads it:

```python
{{#include ../../../ferrotensor-py/examples/mlp.py}}
```

The [Python API](python-reference.md) page lists everything that is covered
and how it differs from PyTorch.

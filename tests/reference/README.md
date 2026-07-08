Phase 0 reference fixtures

These fixtures are deliberately tiny so Rust kernels can be compared against a
known baseline.

## Preferred: committed / pure-Python fixture

A checked-in `phase0_linear_rmsnorm.json` is generated with:

```bash
python3 tests/reference/generate_phase0_pure.py
```

This needs only the Python standard library (no PyTorch). The Rust Phase 0 test
will also attempt to run this generator automatically if the JSON is missing.

## Optional: regenerate from PyTorch

If you want binaries that match a PyTorch reference run:

```bash
pip install torch
python3 tests/reference/generate_phase0_reference.py
```

Schema fields:

- a 3D activation tensor `[B, S, H]`
- a linear weight tensor in HuggingFace layout `[out, in]`
- expected linear output after flattening `[B*S, H] @ W^T`
- expected RMSNorm output

The Rust side validates:

1. `Linear::forward` flatten/unflatten behavior
2. weight transpose-on-load convention
3. RMSNorm numerical correctness

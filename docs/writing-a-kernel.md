# Writing a kernel

This guide adds one element-wise op, **softplus** (`ln(1 + e^x)`), to the
`ferrotensor` crate: a WGSL case, a CPU reference, the public method with its
gradient, and the tests. Every step below was applied to the repo and checked
with the commands at the end.

Read [architecture.md](architecture.md) first for how an op reaches the GPU.

## The shape of a change

Element-wise unary ops share one kernel (`unary`) that switches on an op code.
A new op touches four files in `ferrotensor/src`:

| File | What to add |
|------|-------------|
| `backend/kernels.rs` | an op-code constant and a `case` in the WGSL `unary` kernel |
| `tensor/cpu.rs` | the same maths in `unary_scalar` (the reference implementation) |
| `tensor/ops.rs` | the public method, with its backward closure |
| `tensor/tests.rs` | one line in `cases()` |

Binary ops (`binary_op`), reductions (`reduce_op`) and one-off kernels
(`softmax`, `matmul`, ...) follow the same pattern; a brand-new kernel also
needs an entry in `KERNEL_NAMES` in `backend/compute.rs` and a `*_raw`
dispatch method in `tensor/raw.rs`.

## 1. Op code and WGSL case

In `backend/kernels.rs`, next to the other codes:

```rust
pub(crate) mod unary_op {
    // ...
    pub const LEAKY_RELU_GRAD: u32 = 25;
    pub const SOFTPLUS: u32 = 26;
}
```

and in the `unary` kernel's `switch`, using the same number:

```wgsl
case 25u: { y = select(a, 1.0, x > 0.0); }
case 26u: { y = max(x, 0.0) + log(1.0 + exp(-abs(x))); }
default: { y = x; }
```

`a` and `b` are two scalar parameters passed through the uniform (used by
`clamp`, `leaky_relu`, `add_scalar`, ...). Write numerically safe forms:
`max(x, 0) + ln(1 + e^-|x|)` never overflows, the naive `log(1 + exp(x))` does.

## 2. CPU reference

In `tensor/cpu.rs`, in `unary_scalar`:

```rust
unary_op::SOFTPLUS => x.max(0.0) + (-x.abs()).exp().ln_1p(),
```

The CPU version is the source of truth: the GPU kernel is tested against it,
and the CPU gradient check is the first line of defence.

## 3. Public method and gradient

In `tensor/ops.rs`:

```rust
/// Softplus: `ln(1 + e^x)`, computed without overflow for large `x`.
pub fn softplus(&self) -> Tensor {
    // d/dx softplus(x) = sigmoid(x)
    self.unary_with_grad("softplus", unary_op::SOFTPLUS, 0.0, 0.0, |g, x, _| {
        g.mul(&x.sigmoid())
    })
}
```

`unary_with_grad(name, op, a, b, backward)` runs the kernel and attaches the
backward closure. The closure receives the incoming gradient `g`, the input
`x` and the output `y` (both detached) and returns the gradient with respect
to the input. Write it with ordinary tensor ops so it runs on whatever device
the tensors live on. If the derivative needs its own kernel (as `silu` does),
add a second op code such as `SILU_GRAD`.

## 4. Tests

In `tensor/tests.rs`, add one line to `cases()`:

```rust
case("softplus", &[(&[6], U)], |x| x[0].softplus()),
```

`U` draws inputs uniformly from `[-2, 2)`; other generators are `POS`
(positive only), `NZ` (away from zero, for kinks) and `D` (distinct values).
That single line gives you:

- a central-finite-difference check of the gradient on the CPU
  (`gradients_match_finite_differences`), and
- a GPU-vs-CPU comparison of forward values and gradients
  (`gpu_matches_cpu`).

## 5. Run it

```bash
# WGSL still parses and validates (naga), no GPU needed
cargo test -p ferrotensor --lib kernels_pass_naga_validation

# CPU gradient checks
cargo test -p ferrotensor --lib gradients_match_finite_differences

# GPU vs CPU parity (needs a GPU; skipped without the env var)
FERROTENSOR_RUN_GPU_TESTS=1 cargo test -p ferrotensor --lib gpu_matches_cpu
```

The GPU tests are gated because CI runners have none: without
`FERROTENSOR_RUN_GPU_TESTS=1`, `gpu_matches_cpu` prints "skipping GPU parity" and
passes. With it set, a missing GPU is a hard failure.

## Gotchas

- **Op codes must match** in WGSL, `kernels.rs` and `cpu.rs`. An unknown code
  falls through to `default: y = x` on both sides, so a mismatch fails the
  parity test rather than crashing.
- **1-D dispatch** uses `flat_index(gid, nwg) = x + y * nwg.x * 256`; copy that
  helper instead of indexing with `gid.x` alone, or tensors with more than
  65535 workgroups will be wrong.
- **WGSL has no `pow` for negative bases** and some functions differ in edge
  cases from Rust's `f32`; the `pow_scalar` case in `kernels.rs` shows how to
  handle it.
- **Uniform layout**: uniform structs are padded to 16 bytes. `UnaryParams`
  is four 4-byte fields for that reason.
- **Do not add a kernel just for speed** without a benchmark in `benchmarks/`.

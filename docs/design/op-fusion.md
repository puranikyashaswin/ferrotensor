# Design: fusing chains of element-wise ops (issue #19)

Status: proposed. Implementation follows in separate PRs, measured against
the `tensor-api` benchmark and its checksums.

## Problem

A chain of 100 element-wise ops on a `[64]` tensor takes about 17x as long
on the GPU as on the CPU (`benchmarks/RESULTS.md`, "What is still slow").
Batched dispatch already removed the per-op submission cost; what is left is
per-dispatch GPU time (about 9-13 µs each on an Apple M2) and the host work
to build one bind group per op. For small tensors the arithmetic is
negligible: the cost is the number of dispatches.

## How element-wise ops run today

- `Tensor::unary_raw` and `Tensor::binary_raw` (`ferrotensor/src/tensor/raw.rs`)
  allocate an output buffer and call `GpuContext::dispatch("unary" | "binary", …)`.
- `GpuContext` (`ferrotensor/src/device.rs`) appends the dispatch to a pending
  list; `ComputeEngine::encode` records all pending dispatches into one compute
  pass at each sync point (`to_vec`, `item`, `synchronize`, or every
  `MAX_PENDING_OPS`).
- The `unary` kernel switches on an op code with two scalars `a`, `b`; the
  `binary` kernel handles broadcasting through a stride table.
- Autograd keeps intermediates alive: `unary_with_grad` captures `x.detach()`
  and `y.detach()` for the backward pass, so in training most intermediate
  outputs are still referenced after the next op runs.

## Constraints

1. **Every tensor that exists must hold its value.** An intermediate that
   autograd, or user code, still references has to be written to its buffer.
2. **Bit-identical results by default.** The benchmark checksums and the
   GPU-vs-CPU tests compare exact bits for element-wise ops. A fused kernel
   computes `x * a` and then `+ b` in one shader, which a GPU compiler may
   contract into one FMA instruction, changing the last bit. Fusion must keep
   each step's result rounded to f32 the way the separate kernels do, or be
   off by default where it cannot.
3. **Binding limits.** A dispatch can bind only
   `max_storage_buffers_per_shader_stage` storage buffers (8 in wgpu's
   defaults; much higher on Metal and on NVIDIA Vulkan drivers, and
   Ferrotensor requests the adapter's full limits). A fused group needs one
   binding per external input and per written output.
4. **No change to the public API** and no lazy evaluation visible to users:
   ops still return ordinary tensors.

## Proposal: peephole fusion in the dispatch queue

Fuse at the point where dispatches are queued, not in the tensor graph.

1. `unary_raw`, and `binary_raw` when neither side broadcasts, call a new
   `GpuContext::elementwise(step)` instead of `dispatch`. A step is
   `{ op_kind: unary | binary, op, a, b, inputs: [buffer; 1 or 2], output: buffer, n }`.
2. `GpuContext` keeps an open fusion group: a list of steps over the same
   element count `n`. A new step joins the open group when it has the same
   `n` and the group stays within the binding budget; otherwise the group is
   closed into one dispatch of a new `fused` kernel and a new group starts.
   Any non-element-wise dispatch, sync point or copy closes the group first,
   so ordering with other kernels is unchanged.
3. The `fused` kernel interprets a small program from a storage buffer: for
   each element, it runs the steps in order on registers, reading inputs from
   bound buffers and writing every step's output to its own bound buffer
   (constraint 1). One dispatch replaces up to `G` dispatches.
4. Group size `G` comes from the device's binding limit:
   `G = min(16, limit - external_inputs - 1)`, so it adapts to the adapter.

What this saves: `G - 1` dispatches and bind groups per group, and the
reads of intermediates (each value stays in a register for the next step).
What it does not save: the writes of intermediates, which constraint 1
requires while they are referenced.

### Later step: skipping dead intermediates

When a group is closed, an intermediate whose buffer is referenced only by
the queue itself (no tensor holds it any more, e.g. in `no_grad` code) does
not need to be written. Checking the buffer handle's reference count at
encode time lets the program skip that store and return the buffer to the
pool. This is a separate PR, measured on its own.

## Bit-exactness

WGSL has no way to forbid FMA contraction. Two options, to be decided by
measurement on Metal and Vulkan:

- **A.** Write each step's result to its output buffer and read it back as
  the next step's input inside the shader (the same invocation reads its own
  write, so no barrier is needed). The intent is to force rounding to f32 at
  each step, at the cost of the saved reads. A compiler may still forward the
  value through a register, so this must be verified per backend by the
  checksums, not assumed.
- **B.** Keep values in registers and accept that a fused chain may differ
  from the unfused one in the last bit. Then fusion is opt-in
  (`FERROTENSOR_FUSION=1`) until a tolerance-based test suite replaces the
  exact checksums for these ops.

Default plan: implement A first. If its speed-up is close to B's, keep A
and fusion on by default; otherwise ship B as opt-in.

## Plan as steps

| # | Step | Expected result | If yes | If no |
|---|---|---|---|---|
| 1 | Add the `fused` kernel and `GpuContext::elementwise`, behind `FERROTENSOR_FUSION` (off) | All tests pass with fusion on and off, on Metal and on lavapipe Vulkan in CI | → 2 | Fix; nothing is on by default |
| 2 | Measure `tensor-api` with fusion on (A) | The 100-op `[64]` chain drops several-fold, checksums unchanged | → 3 | If checksums change, try B; if no speed-up, stop and record why |
| 3 | Measure the MLP step and `char_transformer` | No slowdown; small speed-up where element-wise chains are long | → 4 | Restrict fusion to unary-only chains or shorter groups |
| 4 | Run the hardware report on an NVIDIA GPU through Vulkan | Same correctness and a speed-up there too | → 5 | Per-backend group size, or off on that backend |
| 5 | Turn fusion on by default; dead-intermediate skipping as a follow-up | 100-op chain closer to the CPU time | Done | Keep opt-in, document the gap |

## Tests

- Every existing GPU test runs with fusion on and off (CI matrix over
  `FERROTENSOR_FUSION`), including the lavapipe job.
- New tests: groups that hit the binding limit, groups interrupted by a
  sync, a matmul or a broadcasting binary, chains whose intermediates are
  dropped, and autograd through a fused chain (gradients equal unfused).
- The `tensor-api` checksums must match fusion off for option A.

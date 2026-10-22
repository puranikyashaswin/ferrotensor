# Good first issues

Concrete tasks found while writing the docs and running the examples. Each
one names the files involved and how you would know it is done. They are
drafts for GitHub issues; labels are suggestions. Difficulty: **easy** (an
hour or two), **medium** (a day), **harder** (needs design discussion).

Before starting, read [CONTRIBUTING.md](../CONTRIBUTING.md). For anything that
touches a GPU kernel, [writing-a-kernel.md](writing-a-kernel.md) shows the
workflow.

## 1. `cargo clippy` fails on a workspace lint-priority error

*Done.* `rust_2018_idioms` has `priority = -1` in `[workspace.lints.rust]`, and
`cargo clippy --workspace --all-targets --all-features -- -D warnings` exits
cleanly on Rust 1.99.1 (it is also a CI job).

## 2. Remove unused dependencies and empty features from the `ferrotensor` crate

*Done in 0.2.0.* `ferrotensor` now depends only on wgpu, pollster, bytemuck, safetensors, thiserror and rand, and the empty `metal`/`vulkan` features are gone.

*easy · `cleanup` · `good first issue`*

`ferrotensor/Cargo.toml` depends on `ferrotensor-backend`, `ferrotensor-autograd`, `ferrotensor-graph`,
`ferrotensor-trainer`, `ferrotensor-loader`, `serde`, `serde_json`, `memmap2` and `log`,
but nothing under `ferrotensor/src` appears to use them (a grep for each finds no
imports). It also declares `metal` and `vulkan` features that no code reads.
Dropping them shortens builds a lot. Verify each with `cargo build -p ferrotensor`
after removing it, and keep any that turn out to be needed (for example for
re-exports the docs promise).

Done when `cargo build -p ferrotensor --all-targets` and `cargo test -p ferrotensor` pass
with the smaller dependency list.

## 3. Fix rustdoc broken-link warnings

*Done.* `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` exits
cleanly on Rust 1.99.1 (it is also a CI job).

## 4. Softmax kernel does the row reduction in a single thread

*medium · `performance`, `kernels`*

In `ferrotensor/src/backend/kernels.rs` the `softmax` entry point assigns one thread
per row and loops over the whole axis three times (max, sum, normalise). For
long rows (the attention softmax over a sequence) most of the GPU idles.
Rewrite it as one workgroup per row with a shared-memory tree reduction for
the max and the sum. Keep the op codes (`softmax` and `log_softmax`) and the
CPU reference unchanged.

Done when `FERROTENSOR_RUN_GPU_TESTS=1 cargo test -p ferrotensor --lib` passes and
`examples/gpu_matmul_attention.rs` (attention at seq 512) is no slower; report
before/after numbers and the machine in the PR.

## 5. `Embedding::forward` reads ids back from the GPU and uploads them again

*Done.* `Embedding::forward` now gathers with the `gather_rows` kernel reading
the `f32` ids straight from the id tensor's GPU buffer; ids outside
`[0, vocab)` give zero rows. The tests in `ferrotensor/src/nn/embedding.rs`
check that `forward` and `forward_ids` agree bit for bit on the CPU and (with
`FERROTENSOR_RUN_GPU_TESTS=1`) on the GPU. The backward pass still scatters
the gradient on the host.

## 6. Make loss-function target types consistent

*Done in 0.2.0.* Losses and metrics take `impl Into<Targets>`; negative ids are skipped everywhere.

*easy · `api`*

In `ferrotensor/src/functional.rs`, `cross_entropy` takes targets as an `f32`
`Tensor` of class ids, `nll_loss` takes `&[i64]`, and `accuracy` and
`one_hot` take `&[usize]`. Going from one to another needs conversions (see
`examples/mlp_two_moons.rs`). Propose and implement one convention (for
example a `Targets` enum or an `impl Into` trait, with the old forms kept
working), update the doc examples, and note it in `CHANGELOG.md`.

## 7. Add `tril` / `triu` (and use them for the causal mask)

*Done in 0.2.0.* `Tensor::tril` / `Tensor::triu` exist and build the causal mask.

*easy · `tensor-ops`, `good first issue`*

There is no way to build a lower-triangular mask with the tensor API; the
causal attention path builds its mask internally. Add `Tensor::tril(diagonal)`
and `triu(diagonal)` using `masked_fill` or a small kernel, with gradient
checks via a new `case(...)` in `ferrotensor/src/tensor/tests.rs`. Follow
[writing-a-kernel.md](writing-a-kernel.md) if you choose the kernel route.

Done when the new ops pass the CPU gradient check and the GPU parity test.

## 8. KV cache for `nn::MultiHeadAttention`

*Done in 0.2.0.* `MultiHeadAttention::forward_cached` with `nn::KvCache`, tested against the full causal forward.

*harder · `nn`, `inference`*

`examples/char_transformer.rs` generates text by re-running the whole context
window for every new character, because `nn::MultiHeadAttention` has no
cache. Add an incremental forward that appends the new key/value rows to a
cache and attends from the new position only. The LLM stack already has a
`KVCache` in `ferrotensor-graph` to learn from.

Done when generating N tokens with the cache matches the uncached logits
within 1e-5 (a test), and the example uses it.

## 9. GPU parity tests for the `nn` layers and `serialize`

*Done in 0.2.0.* GPU parity tests cover every `nn` layer and safetensors round trips.

*easy · `tests`, `good first issue`*

`FERROTENSOR_RUN_GPU_TESTS=1` runs GPU-vs-CPU checks for tensor ops, functional ops
and the optimizers, but not for whole layers (`MultiHeadAttention`,
`LayerNorm`, `Embedding`, `Sequential`) or for save/load of a GPU module.
Add a test per layer that builds the layer, copies its weights to a GPU
device (`Tensor::to_device`), and compares forward and gradients with the CPU
result, gated the same way as `gpu_training_matches_cpu` in
`ferrotensor/src/optim/tests.rs`. Add a safetensors round trip from a GPU module.

## 10. Cut per-op overhead on the GPU path

*Done.* GPU ops are recorded into one command encoder and one compute pass
and submitted at each read back, with pooled output buffers and a shared
argument buffer. On this repo's M2, `char_transformer` trains 600 steps in
31.9 s on the GPU (105.0 s before; 42.5 s on the CPU) and `mlp_two_moons`
takes about 2 s on the GPU (12-17 s before). See
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md#ferrotensor-tensor-api-cpu-and-gpu-devices)
for the method and what is still slow.

## 11. Port the legacy `ferrotensor_backend` examples and verify they run

*Done.* The WGSL parse error is gone and every example runs on an Apple M2:
the four in `ferrotensor-backend/examples` (`init_wgpu`, `tensor_creation`,
`fused_op`, `ops_test`), plus `basic_tensor` (moved there from the
`ferrotensor` crate) and `flash_attention` (moved to
`ferrotensor-graph/examples`). Each has its command and real output in its
doc comment and checks its result against a CPU computation or hand-computed
values. `lora_finetune` was deleted: its loss table was a formula, not
training; the real LoRA path is the
[walkthrough](examples/lora-sft.md).

## 12. `DataLoader` clones every sample and has no prefetching

*Done in 0.2.0.* Batches are stacked without per-sample clones, and `DataLoader::epoch_prefetched` prefetches on a background thread.

*medium · `data`*

`Dataset::get` returns owned `Sample`s, so `DataLoader::epoch` clones each
sample's data once to read it and again into the batch (`ferrotensor/src/data.rs`).
Look at borrowing samples (for example `get(&self, i) -> (&Sample, &Sample)`
or a visitor), and optionally add a prefetch thread that prepares the next
batch while the current step runs.

Done when a benchmark with a large in-memory dataset shows fewer allocations or
less time per epoch, and the public behaviour (order, shuffling, `drop_last`)
is unchanged and tested.

## 13. `FlashAttention::memory_savings` prints "0MB vs 1MB"

*easy · `bug`, `docs`, `good first issue`*

`ferrotensor-graph/src/flash_attention.rs` formats the two sizes with integer
division by 1,000,000, so for sequence length 512 and head dim 64 it prints
`8.0x less memory (0MB vs 1MB for seq=512)`: the flash side rounds to zero.
Print KiB (or switch units by size), and say in the doc comment that the
sizes are per head and for the score matrix only.

Done when the string for `seq = 512` shows non-zero sizes and the existing
`test_memory_savings` is extended to check the numbers.

## 14. `FlashAttention::forward_multi_head` and `multi_head_forward` are stubs

*easy · `cleanup`, `good first issue`*

In `ferrotensor-graph/src/flash_attention.rs`, `forward_multi_head` has a
`TODO` and calls `forward`, and `multi_head_forward` calls `forward` with the
same tensor as Q, K and V and no projections, although its doc comment says it
"handles Q, K, V projection and output projection". `forward` already splits
heads (it views the `[batch, seq, heads, head_dim]` layout). Either delete
the two methods and update their callers, or make the docs say what they
do. Check with `rg "multi_head_forward|forward_multi_head"` first.

Done when no public method's doc comment promises something it does not do.

## 15. Run the reductions in `cross_entropy`, `argmax` and `cat` on the GPU

*medium · `performance`, `kernels`*

Since GPU ops are [batched](architecture.md#batched-gpu-dispatch), every op
that copies values to the host in the middle of its work splits the batch.
The ones that do (see "What is still slow" in
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md#ferrotensor-tensor-api-cpu-and-gpu-devices)):
`cross_entropy` with class-index targets, `argmax`/`argmin` and `cat`. Pick
one, write the kernel (see [writing-a-kernel.md](writing-a-kernel.md)) and
keep the CPU reference as the oracle.

Done when the op's GPU path no longer calls `to_vec` or `item`, the GPU parity
test passes, and `cargo run -p ferrotensor-benchmarks --release --bin
tensor-api` shows the case you touched no slower (report both numbers and the
machine).

## 16. Index backward passes (`index_select`, `Embedding`) scatter on the host

*medium · `performance`, `kernels`*

The backward pass of `index_select` and of `nn::Embedding` reads the ids and
the incoming gradient back to the host and scatters there, which forces a
sync on the GPU. Add a scatter-add kernel (atomics on `u32`-encoded floats or
a sorted-ids reduction; note WGSL has no float atomics) and use it from both.

Done when the gradient checks in `ferrotensor/src/tensor/tests.rs` and
`ferrotensor/src/nn/embedding.rs` pass on the CPU and, with
`FERROTENSOR_RUN_GPU_TESTS=1`, on the GPU, and neither backward calls `to_vec`.

## 17. Full reductions (`sum()`) use a single GPU thread

*medium · `performance`, `kernels`*

A reduction over a whole tensor (`x.sum()`) runs one GPU thread per output,
so for a scalar output the `reduce` kernel in
`ferrotensor/src/backend/kernels.rs` uses one thread to add every element.
Write a two-stage or workgroup-tree reduction for the full-reduction case and
leave the per-axis path alone.

Done when `sum()` over a `[4096, 4096]` tensor on the GPU matches the CPU
within f32 tolerance and is measurably faster than before (report numbers and
the machine).

## 18. `clip_grad_norm` reads one scalar back per parameter

*easy · `performance`, `good first issue`*

`optim::clip_grad_norm` computes `g.square().sum().item()` for every
parameter (`ferrotensor/src/optim/optimizer.rs`), so on the GPU each
parameter forces a sync. Accumulate the squared norms on the device (add the
per-parameter sums as tensors, then read the total once), keeping the result
and the clipping behaviour identical.

Done when the existing clipping tests pass unchanged and
`FERROTENSOR_RUN_GPU_TESTS=1 cargo test -p ferrotensor --lib` still passes.

## 19. Fuse chains of element-wise ops into one kernel

*harder · `performance`, `kernels`*

100 tiny element-wise ops on the GPU still take about 18x as long as on the
CPU, mostly GPU time per dispatch (about 9-13 µs each). Because ops are now
recorded before they run, consecutive `unary`/`binary` dispatches on same-sized
buffers could be merged into one kernel that applies the op codes in
sequence. This needs design discussion first: where the merge happens
(`GpuContext::dispatch` or `ComputeEngine::encode`), and how intermediates
that other tensors still reference stay correct.

Done when the 100-op case in `tensor-api` improves, every output stays
bit-identical (the benchmark prints checksums), and the CPU-vs-GPU parity tests
pass.

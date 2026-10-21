# Good first issues

Concrete tasks found while writing the docs and running the examples. Each
one names the files involved and how you would know it is done. They are
drafts for GitHub issues; labels are suggestions. Difficulty: **easy** (an
hour or two), **medium** (a day), **harder** (needs design discussion).

Before starting, read [CONTRIBUTING.md](../CONTRIBUTING.md). For anything that
touches a GPU kernel, [writing-a-kernel.md](writing-a-kernel.md) shows the
workflow.

## 1. `cargo clippy` fails on a workspace lint-priority error

*easy · `build` · `good first issue`*

On Rust 1.99.1, `cargo clippy --workspace --all-targets` stops with
`error: lint group 'rust_2018_idioms' has the same priority (0) as a lint`
pointing at `Cargo.toml:28`. Lint groups in `[workspace.lints.rust]` need a
lower priority than the individual lints. Give `rust_2018_idioms` a
`priority = -1` (and check the `[workspace.lints.clippy]` table for the same
problem), then fix whatever warnings clippy shows once it can run.

Done when `cargo clippy --workspace --all-targets` exits cleanly on stable.

## 2. Remove unused dependencies and empty features from the `rusty` crate

*easy · `cleanup` · `good first issue`*

`rusty/Cargo.toml` depends on `rusty-backend`, `rusty-autograd`, `rusty-graph`,
`rusty-trainer`, `rusty-loader`, `serde`, `serde_json`, `memmap2` and `log`,
but nothing under `rusty/src` appears to use them (a grep for each finds no
imports). It also declares `metal` and `vulkan` features that no code reads.
Dropping them shortens builds a lot. Verify each with `cargo build -p rusty`
after removing it, and keep any that turn out to be needed (for example for
re-exports the docs promise).

Done when `cargo build -p rusty --all-targets` and `cargo test -p rusty` pass
with the smaller dependency list.

## 3. Fix rustdoc broken-link warnings

*easy · `docs` · `good first issue`*

`cargo doc --workspace --no-deps` warns `unresolved link to 'target'` at
`rusty-backend/src/lib.rs:5087`: the text `[target]` in a doc comment is read
as an intra-doc link. Escape the brackets (`\[target\]`), then fix any other
warnings the command prints. CI builds the docs, so this keeps the output
clean.

## 4. Softmax kernel does the row reduction in a single thread

*medium · `performance`, `kernels`*

In `rusty/src/backend/kernels.rs` the `softmax` entry point assigns one thread
per row and loops over the whole axis three times (max, sum, normalise). For
long rows (the attention softmax over a sequence) most of the GPU idles.
Rewrite it as one workgroup per row with a shared-memory tree reduction for
the max and the sum. Keep the op codes (`softmax` and `log_softmax`) and the
CPU reference unchanged.

Done when `RUSTY_RUN_GPU_TESTS=1 cargo test -p rusty --lib` passes and
`examples/gpu_matmul_attention.rs` (attention at seq 512) is no slower; report
before/after numbers and the machine in the PR.

## 5. `Embedding::forward` reads ids back from the GPU and uploads them again

*easy · `performance`*

`rusty/src/nn/embedding.rs` implements `Module::forward` by calling
`x.to_vec()` to turn the id tensor into indices, then `functional::embedding`
uploads them again. On a GPU tensor that is a device-to-host copy plus a
host-to-device copy per call, and it forces a sync. Either document that
`forward` expects host-friendly ids and steer users to `forward_ids`, or add a
gather path that keeps the ids on the device (the `gather_rows` kernel is the
starting point).

Done when a test shows both paths give identical output on CPU, and (with
`RUSTY_RUN_GPU_TESTS=1`) on the GPU.

## 6. Make loss-function target types consistent

*easy · `api`*

In `rusty/src/functional.rs`, `cross_entropy` takes targets as an `f32`
`Tensor` of class ids, `nll_loss` takes `&[i64]`, and `accuracy` and
`one_hot` take `&[usize]`. Going from one to another needs conversions (see
`examples/mlp_two_moons.rs`). Propose and implement one convention (for
example a `Targets` enum or an `impl Into` trait, with the old forms kept
working), update the doc examples, and note it in `CHANGELOG.md`.

## 7. Add `tril` / `triu` (and use them for the causal mask)

*easy · `tensor-ops`, `good first issue`*

There is no way to build a lower-triangular mask with the tensor API; the
causal attention path builds its mask internally. Add `Tensor::tril(diagonal)`
and `triu(diagonal)` using `masked_fill` or a small kernel, with gradient
checks via a new `case(...)` in `rusty/src/tensor/tests.rs`. Follow
[writing-a-kernel.md](writing-a-kernel.md) if you choose the kernel route.

Done when the new ops pass the CPU gradient check and the GPU parity test.

## 8. KV cache for `nn::MultiHeadAttention`

*harder · `nn`, `inference`*

`examples/char_transformer.rs` generates text by re-running the whole context
window for every new character, because `nn::MultiHeadAttention` has no
cache. Add an incremental forward that appends the new key/value rows to a
cache and attends from the new position only. The LLM stack already has a
`KVCache` in `rusty-graph` to learn from.

Done when generating N tokens with the cache matches the uncached logits
within 1e-5 (a test), and the example uses it.

## 9. GPU parity tests for the `nn` layers and `serialize`

*easy · `tests`, `good first issue`*

`RUSTY_RUN_GPU_TESTS=1` runs GPU-vs-CPU checks for tensor ops, functional ops
and the optimizers, but not for whole layers (`MultiHeadAttention`,
`LayerNorm`, `Embedding`, `Sequential`) or for save/load of a GPU module.
Add a test per layer that builds the layer, copies its weights to a GPU
device (`Tensor::to_device`), and compares forward and gradients with the CPU
result, gated the same way as `gpu_training_matches_cpu` in
`rusty/src/optim/tests.rs`. Add a safetensors round trip from a GPU module.

## 10. Cut per-op overhead on the GPU path

*harder · `performance`*

Each op creates a uniform buffer, a bind group and a command encoder, and
submits it on its own (`ComputeEngine::run` in
`rusty/src/backend/compute.rs`, called from `rusty/src/tensor/raw.rs`). On
this repo's M2, the 109k-parameter transformer in
`examples/char_transformer.rs` trains 600 steps in about 47 s on the CPU and
141 s on the GPU, and the 1.2k-parameter MLP takes about 0.5 s on the CPU and
tens of seconds on the GPU. Options: cache bind groups and pipelines'
layouts, pool the small uniform buffers, and record several ops into one
encoder so they submit together. Needs a design note before a large PR.

Done when the GPU run of `char_transformer` is measurably faster with
identical results, with before/after numbers in the PR.

## 11. Port the legacy `rusty_backend` examples and verify they run

*medium · `examples`, `docs`*

`rusty/examples/basic_tensor.rs`, `flash_attention.rs` and `lora_finetune.rs`
are written against the low-level `rusty_backend` API, not the `rusty` tensor
API the rest of the docs use, and at the time of writing `basic_tensor`
panics in `ComputeEngine::new` with a WGSL parse error. Once the backend fix
lands, run each, fix what is broken, and put the real output in the example's
doc comment and in `rusty/examples/README.md`. Delete or clearly mark any
example that cannot be made to work.

## 12. `DataLoader` clones every sample and has no prefetching

*medium · `data`*

`Dataset::get` returns owned `Sample`s, so `DataLoader::epoch` clones each
sample's data once to read it and again into the batch (`rusty/src/data.rs`).
Look at borrowing samples (for example `get(&self, i) -> (&Sample, &Sample)`
or a visitor), and optionally add a prefetch thread that prepares the next
batch while the current step runs.

Done when a benchmark with a large in-memory dataset shows fewer allocations or
less time per epoch, and the public behaviour (order, shuffling, `drop_last`)
is unchanged and tested.

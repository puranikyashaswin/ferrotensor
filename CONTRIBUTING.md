# Contributing to Ferrotensor

Thanks for your interest. Ferrotensor is a young, one-maintainer project, so small,
focused pull requests that come with tests are the easiest to review and merge.

## Setup

You need Rust 1.88 or newer (stable). A GPU is optional: the default test
suite runs on the CPU device.

```bash
git clone https://github.com/puranikyashaswin/ferrotensor.git
cd ferrotensor
cargo build --workspace
cargo test --workspace
```

Then try an example to check your GPU works:

```bash
cargo run --release -p ferrotensor --example quickstart
```

Building the whole workspace needs a few GB of RAM; on small machines use
`CARGO_BUILD_JOBS=2` and build one crate at a time (`-p ferrotensor`).

## Where things are

| You want to... | Read |
|----------------|------|
| understand how a tensor op becomes a GPU kernel | [docs/architecture.md](docs/architecture.md) |
| add a GPU op, with tests | [docs/writing-a-kernel.md](docs/writing-a-kernel.md) |
| see what works today | [docs/use-cases.md](docs/use-cases.md) |
| find something to work on | [docs/good-first-issues.md](docs/good-first-issues.md) |

The crates: `ferrotensor` is the tensor API (tensor, autograd, `nn`, `optim`, `data`,
`serialize`); `ferrotensor-backend`, `ferrotensor-autograd`, `ferrotensor-graph`, `ferrotensor-loader`,
`ferrotensor-trainer`, `ferrotensor-cli`, `ferrotensor-monitor` and `ferrotensor-hub` are the LLM
fine-tuning stack; `benchmarks` holds the GPU benchmarks. Each crate has a
README.

## Tests

```bash
cargo test --workspace                      # everything; no GPU needed
cargo test -p ferrotensor                         # one crate
cargo test -p ferrotensor --lib tensor::tests     # one module
```

### GPU tests are opt-in

CI runners have no GPU, so tests that need one are skipped unless you set
`FERROTENSOR_RUN_GPU_TESTS=1`:

```bash
FERROTENSOR_RUN_GPU_TESTS=1 cargo test -p ferrotensor --lib
```

With the variable set, the op test suite also runs every case on the GPU and
requires its forward values and gradients to match the CPU result, and a
missing GPU is a failure. Without it, those tests print "skipping GPU parity"
and pass. If you change a kernel or a backward formula, run the GPU tests
locally before opening the PR and say so in the description.

Test types you will meet:

- **Gradient checks** (`ferrotensor/src/tensor/tests.rs`): add a `case(...)` line
  for a new op; it is checked against finite differences.
- **Shader validation** (`kernels_pass_naga_validation`): parses and validates
  the WGSL with naga, so shader typos fail on any machine.
- **Doc tests**: public items carry runnable examples; `cargo test` runs them.

## Style and checks

Run these before pushing; they are what CI runs:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo doc --workspace --no-deps
```

> Known issue: on recent stable toolchains `cargo clippy` currently stops at a
> lint-priority error in the workspace `Cargo.toml`
> ([issue 1](docs/good-first-issues.md#1-cargo-clippy-fails-on-a-workspace-lint-priority-error)).
> Fixing it is a good first PR.

- Document every public item (`missing_docs` is a warning).
- No `unsafe` (the workspace forbids it).
- Avoid `unwrap()`, `expect()` and `panic!` in library code except for
  documented precondition panics (the lints warn on them).
- Match the surrounding code: naming, comment density, error handling.
- Do not add benchmark or performance claims to docs without a measurement you
  can reproduce, and say which machine produced it.

## Pull requests

1. Open or comment on an issue first for anything larger than a small fix.
2. Branch from `main`, keep the change focused, and add tests.
3. Make sure fmt, clippy, tests and docs pass as above.
4. Write a clear commit message: a short imperative subject (about 50
   characters), and a body saying why when it is not obvious. Conventional
   prefixes (`fix:`, `feat:`, `docs:`) are fine but not required.
5. Open the PR with what changed, how you tested it (including whether you ran
   the GPU tests, and on what hardware), and anything you are unsure about.
6. Expect review comments; push fixes as new commits and they will be squashed
   or rebased on merge.

Docs follow the same rule as code: every command and number in them must be
something you ran.

## Good first issues

[docs/good-first-issues.md](docs/good-first-issues.md) lists concrete, small
tasks with the files involved. Typical entry points:

- add a missing element-wise op or loss (follow
  [writing-a-kernel.md](docs/writing-a-kernel.md); the test is one line);
- add a GPU/CPU parity test for a layer that only has CPU tests;
- fix a documented API inconsistency;
- improve a doc page or add a runnable example (it must run, and its output
  goes in the page).

If you want to take one, comment on the issue so work is not duplicated.

## Reporting issues

- **Bugs:** include `rustc --version`, OS, GPU (`Device::best().name()`), and
  the smallest program that reproduces it. Say whether it also fails on
  `Device::cpu()`, which tells us if the bug is in a kernel or in the op logic.
- **Features:** describe the use case first, then the proposed API.
- **Security issues:** see [SECURITY.md](SECURITY.md); do not open a public issue.

## License

By contributing, you agree that your contributions are dual licensed under
MIT OR Apache-2.0, the same terms as the project, without any additional
terms or conditions.

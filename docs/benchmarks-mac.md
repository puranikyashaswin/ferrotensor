# Comparing Ferrotensor, mlx-lm and PyTorch on a Mac

`scripts/bench_mac.sh` trains the same LoRA recipe on SmolLM2-135M with three
stacks on an Apple-silicon Mac, one after another, and prints one Markdown
table. A JSON file keeps every raw number, the versions of everything involved
and the exact commands.

| Stack | What runs | GPU path |
|---|---|---|
| Ferrotensor | the `ferrotensor train` CLI | Metal through wgpu |
| mlx-lm | `mlx_lm.lora` (the trainer unchanged, plus a recording callback) | MLX on Metal |
| PyTorch + PEFT | [`benchmarks/baselines/pytorch_lora.py`](../benchmarks/baselines/pytorch_lora.py) with `--device mps` | MPS |

This page has no results. Numbers go into a document only after someone has
run the script, with the output of `results.json` next to them. Nothing here
says which stack is faster.

## Run it

```bash
scripts/bench_mac.sh
```

On first use this creates a virtual environment outside the repository
(`~/.cache/ferrotensor-bench-mac/venv`, or `$FERROTENSOR_BENCH_VENV`), installs
the packages pinned in
[`scripts/bench_mac_requirements.txt`](../scripts/bench_mac_requirements.txt),
builds the release `ferrotensor` binary, downloads the model (about 270 MB)
and the Alpaca data, and then runs the three stacks. It needs Python 3.11 or
newer (set `PYTHON=` to choose one), a Rust toolchain, and an Apple-silicon Mac.

| Option | Meaning |
|---|---|
| `--steps N` | optimizer steps per run (default 250, which is one epoch of the 2,000 training examples) |
| `--only ferrotensor\|mlx\|pytorch` | run one stack, or a comma-separated subset |
| `--out DIR` | where caches, logs and results go (default `data/bench-mac`, which is gitignored) |
| `--skip-install` | reuse the virtual environment as it is (no `pip install`) |
| `--repeats N` | runs per stack; the table then shows the median with the minimum and maximum |
| `--discard-steps N` | steps dropped from the timing as warm-up (default 3) |
| `--final-window N` | steps averaged for the final training loss (default 10) |
| `--base-dtype f32\|native` | frozen-weight precision, see below (default `f32`) |
| `--cooldown S` | seconds between runs (default 30) |
| `--dry-run` | print the three commands and the mlx-lm config, then exit; nothing is downloaded or run |

A short pass to check the setup before a full run: `scripts/bench_mac.sh --steps 20`.
(With 20 steps or fewer the learning-rate warmup is shortened to fit, so the
losses are not those of the full recipe.)

Outputs, in the `--out` directory:

- `results.md`: the table below, also printed on stdout;
- `results.json`: every raw number (each step's seconds and loss for every run,
  the `/usr/bin/time -l` reports), the settings, the machine (OS, chip, RAM),
  the git commit, the versions of Ferrotensor, rustc, mlx, mlx-lm, PyTorch,
  PEFT, transformers and the other libraries, the exact command of every run
  and the list of known differences;
- `runs/<stack>-r<k>/`: each run's log, its `/usr/bin/time` report and the
  files the stack wrote (for mlx-lm, the generated `mlx_config.yaml`).

If any number is missing or malformed, or any stack exits with an error, the
script stops with a message and prints no table. The raw numbers of the runs
that finished are kept in `results.partial.json`.

## The recipe

The recipe is the one in the [LoRA walkthrough](examples/lora-sft.md), built the
same way as `scripts/lora-run.sh` and the PyTorch baseline:

- model [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M);
- data [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned),
  prompt plus response at most 500 characters, seed-0 shuffle, the first 64
  held out and the next 2,000 for training, wrapped in ChatML with the loss on
  the response tokens only;
- LoRA rank 8, alpha 16 on `q_proj`, `k_proj`, `v_proj` and `o_proj` of every
  layer (921,600 trainable parameters), f32 adapters;
- AdamW (0.9, 0.999, 1e-8), no weight decay, lr 5e-4, 20 linear warmup steps,
  cosine decay to lr/10, global gradient-norm clipping at 1.0;
- batch size 1, 8 accumulation steps (8 sequences per optimizer step),
  sequences cut at 256 tokens, 250 steps.

The three stacks read the same Alpaca files (the mlx-lm copy is the same rows
as `prompt` and `completion`) and are evaluated on the same 64 held-out
examples.

### Commands

`scripts/bench_mac.py --dry-run` prints the exact commands. Each is run under
`/usr/bin/time -l`. Ferrotensor:

```bash
ferrotensor train --model data/bench-mac/cache/models/SmolLM2-135M --f32-weights --device gpu \
  --data .../alpaca/train.jsonl --eval-data .../alpaca/val.jsonl --eval-samples 64 --eval-every 0 \
  --out RUN_DIR --seq-len 256 --lr 0.0005 --warmup-steps 20 --batch-size 1 --grad-accum 8 \
  --max-steps 250 --grad-clip 1.0 --weight-decay 0.0 --lora-rank 8 --lora-alpha 16.0 \
  --lora-targets q_proj,k_proj,v_proj,o_proj --seed 0 --log-every 1 --save-every 0
```

PyTorch + PEFT:

```bash
python benchmarks/baselines/pytorch_lora.py HuggingFaceTB/SmolLM2-135M --device mps --dtype f32 \
  --steps 250 --seq-len 256 --batch-size 1 --grad-accum 8 --warmup-steps 20 --seed 0 \
  --data-root data/bench-mac/cache --summary RUN_DIR/summary.json --record-steps
```

mlx-lm is configured with a YAML file (`mlx_lm.lora` has no flags for the
optimizer settings or the LoRA keys). The run is
`python scripts/bench_mac.py _mlx-worker METRICS.json --config RUN_DIR/mlx_config.yaml`,
a small wrapper that calls `mlx_lm.lora`'s own `main()` and records the
training and evaluation reports from its callback, because mlx-lm prints
progress through a terminal UI that cannot be parsed reliably. The trainer is
not modified. The config:

```yaml
fine_tune_type: "lora"
train: true
test: true
mask_prompt: true
num_layers: 30
batch_size: 1
grad_accumulation_steps: 8
iters: 2000                 # micro-batches: 250 optimizer steps x 8
max_seq_length: 257
learning_rate: 0.0005
lr_schedule: {name: cosine_decay, warmup: 20, warmup_init: 0.0, arguments: [0.0005, 230, 0.00005]}
optimizer: "adamw"
optimizer_config: {adamw: {betas: [0.9, 0.999], eps: 1e-8, weight_decay: 0.0, bias_correction: true}}
lora_parameters: {rank: 8, scale: 2.0, dropout: 0.0, keys: [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj]}
steps_per_report: 8
val_batches: -1
test_batches: -1
```

(The real file also sets `model`, `data`, `seed`, `adapter_path` and the
eval/save intervals, and writes the numbers in decimal.) It was written against
**mlx-lm 0.32.2** (`mlx_lm.lora` as of that release; the flags and config keys
were read from that version's `lora.py`, `tuner/trainer.py` and
`tuner/datasets.py`) and is pinned to it. A different mlx-lm release may need
changes here.

## What the table reports

| Row | How it is obtained |
|---|---|
| Seconds per optimizer step | each stack's own per-step timer; the first 3 steps are dropped (shader and kernel compilation, allocator warm-up) and the median of the remaining steps is reported |
| Peak resident set size | `/usr/bin/time -l` ("maximum resident set size") around the whole command |
| Peak memory footprint | `/usr/bin/time -l` ("peak memory footprint") |
| GPU memory reported by the library | mlx-lm: `mlx.core.get_peak_memory()` at the end of the process. PyTorch: the highest `torch.mps.driver_allocated_memory()` sampled after each step and after the final evaluation (PyTorch has no peak counter for MPS). Ferrotensor: not exposed by the CLI, shown as `n/a` |
| Final training loss | mean of the logged loss of the last 10 steps (80 sequences) |
| Held-out loss before / after | mean over every response token of the 64 held-out examples, evaluated in the same process before training and after the last step |

### How timing is measured

- Ferrotensor: `--log-every 1` makes `metrics.jsonl` hold one `seconds` value
  per optimizer step.
- mlx-lm: the callback receives `iterations_per_second` for every report, and
  `steps_per_report` is 8, so one report is one optimizer step and its time is
  8 / `iterations_per_second`. mlx-lm's own timer excludes its evaluations and
  batch assembly.
- PyTorch: `time.perf_counter()` around each optimizer step with
  `torch.mps.synchronize()` before the clock starts and before it stops.

The timers are not identical; the exact scope of each is listed in the
output under "Known differences".

### How memory is measured

On Apple silicon the CPU and the GPU share one pool of memory, so "GPU
memory" and "process memory" are not separate numbers. Two kinds of number are
therefore reported and labelled:

1. `/usr/bin/time -l` around the whole process, the same way for all three.
   The maximum resident set size is the largest resident size among the
   processes the command waited for (the command and the children it
   reaped), not a sum. It includes model loading, both held-out evaluations
   and, for mlx-lm and PyTorch, the Python interpreter and the libraries they
   import. It does not count memory the GPU driver holds outside the
   process's resident pages, which is why the peak memory footprint (the
   kernel's accounting of what the process is responsible for, also from
   `time -l`) is shown next to it. The footprint is only printed by recent
   macOS releases; if it is missing the row shows `n/a`.
2. What the library itself says about GPU memory, where it says anything. The
   mlx figure is a true peak; the PyTorch figure is a maximum over samples and
   can miss a short spike between steps; Ferrotensor has none.

Neither is "the GPU's memory" in an exact sense, and the two kinds are not
interchangeable. Compare like with like across stacks only on the first.

## Known differences between the stacks

The output prints this list with every run. mlx-lm and PyTorch cannot be made
identical to Ferrotensor with the options they offer; where they differ, the
difference is stated and not hidden.

- **Base-weight precision.** With `--base-dtype f32` (the default) all three
  keep the frozen weights in f32: `ferrotensor train --f32-weights`; the model
  converted with `mlx_lm convert --dtype float32` (not timed); PyTorch
  float32. Ferrotensor's CLI default is f16 frozen weights with f32 compute;
  `--base-dtype native` runs each stack's own default, which makes mlx-lm
  compute in bfloat16 (the checkpoint's dtype) while PyTorch stays float32
  because the baseline script has no other mode on MPS. Adapters are f32 in
  every case.
- **Data order.** Each stack shuffles with its own seeded generator, so the
  8 sequences in a step differ between stacks. The final training loss is over
  different examples. The held-out loss is on the same examples.
- **Gradient clipping.** Ferrotensor and PyTorch clip the global gradient norm
  at 1.0. mlx-lm's trainer has no gradient clipping, so its run is unclipped.
- **Learning-rate schedule.** The mlx-lm schedule is a warmup followed by
  `cosine_decay` and starts at 0 for the first update, so it lags the other
  two by one optimizer step.
- **Logged training loss.** Ferrotensor weights each sequence of a step by its
  response-token count; mlx-lm and PyTorch report the plain mean of the 8
  per-sequence mean losses.
- **Padding.** mlx-lm pads each batch to 1 + a multiple of 32 tokens;
  `max_seq_length` is 257 so that sequences are cut to the same 256 input
  tokens as in the other two.
- **Optimizer.** All three use AdamW with the same betas and epsilon and no
  weight decay. mlx-lm's AdamW does not apply bias correction unless told to,
  so `bias_correction: true` is set.
- **Chat template.** The `SmolLM2-135M` base checkpoint has no chat template.
  mlx-lm is given a local copy of the model whose `tokenizer_config.json` has a
  ChatML template, and `mask_prompt` restricts the loss to the response. The
  script compares the resulting tokens and prompt lengths with the baseline's
  formatting for every training and held-out example and reports the number of
  mismatches.
- **Timers** cover slightly different things (see above), and mlx-lm compiles
  its training step per input shape with `mx.compile`, so a new padded length
  after the warm-up can produce one slow step. The median ignores it; the JSON
  has every step.

A sanity check is built in: the held-out loss before training should be the
same in all three, to about 0.01, because the forward passes are the same. If
it is not, the output says so; look at `--base-dtype` and the data first.

## Caveats

- **First run.** Metal shader and kernel compilation, MLX graph compilation
  and PyTorch's MPS graph caching all happen on the first steps. The warm-up
  steps are excluded from the timing; they are still in the wall time and in
  the per-step data in `results.json`. The first run on a machine also
  downloads the model and the data, and the first mlx-lm run converts the
  model; none of that is timed.
- **Thermal state.** A fanless MacBook Air slows down when it is hot. Runs are
  interleaved (Ferrotensor, mlx-lm, PyTorch, then the next repeat) with a
  cool-down between them so that no stack always runs last, and `results.json`
  records the power source and `pmset -g therm` output before each run. A run
  on battery or in Low Power Mode is not comparable with one on power.
- **Other apps.** Close them. Memory pressure on an 8 GB machine changes both
  speed and the resident size.
- **One run is not a distribution.** Use `--repeats 3` or more for anything
  you plan to quote; the table then gives the median and the minimum and
  maximum across repeats.
- **Order of stacks.** They run strictly one after another, never in parallel.
- **Versions.** Results belong to one commit of Ferrotensor and one set of
  library versions; both are in `results.json`. Quote them together.
- **What was verified.** The harness is checked for syntax and its `--help` and
  `--dry-run` output. The first full run on a Mac is the first test of the
  parsing of each stack's output; the script is written to stop with an error
  when something does not parse.

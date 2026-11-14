#!/usr/bin/env python3
"""Train one LoRA recipe with three stacks on an Apple-silicon Mac and compare them.

The recipe is the one in docs/examples/lora-sft.md: SmolLM2-135M, 2,000 Alpaca
training examples and 64 held out, LoRA rank 8 / alpha 16 on q/k/v/o of every
layer, AdamW at lr 5e-4 with 20 warmup steps and cosine decay to lr/10, 8
sequences per optimizer step (batch 1 x 8 accumulation), sequences cut at 256
tokens, f32 adapters.

The stacks run one after another, never in parallel:

  ferrotensor  the `ferrotensor train` CLI (Metal through wgpu)
  mlx          `mlx_lm.lora` (the unmodified trainer, with a recording callback)
  pytorch      benchmarks/baselines/pytorch_lora.py --device mps (PyTorch + PEFT)

Each run is wrapped in `/usr/bin/time -l` for peak resident memory. Per-step
seconds, losses and library-reported GPU memory come from each stack's own
output. Any missing or malformed number stops the run with an error; no partial
table is printed.

Normally started through scripts/bench_mac.sh, which sets up the environment.
Nothing is downloaded or run with --dry-run, which only prints the commands.
See docs/benchmarks-mac.md.
"""

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "benchmarks" / "baselines" / "pytorch_lora.py"
STACKS = ("ferrotensor", "mlx", "pytorch")
LABELS = {"ferrotensor": "Ferrotensor", "mlx": "mlx-lm", "pytorch": "PyTorch + PEFT (MPS)"}

# The recipe. These are fixed so that every run of the harness trains the same thing.
SEQ_LEN = 256
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 5e-4
LR_WARMUP = 20
LORA_RANK = 8
LORA_ALPHA = 16.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.0
ADAM_BETAS = [0.9, 0.999]
ADAM_EPS = 1e-8
SEED = 0
HELDOUT = 64
TRAIN_EXAMPLES = 2000

MLX_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' }}"
    "{% if not loop.last or add_generation_prompt %}{{ '\\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


class BenchError(Exception):
    """A failure the harness reports and stops on."""


def say(message=""):
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# environment


def sh(cmd, check=False):
    """Run a short command; return its stdout, or None if it failed (unless check)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        if check:
            raise BenchError(f"{shlex.join(map(str, cmd))} failed: {e}") from e
        return None
    if out.returncode != 0:
        if check:
            raise BenchError(f"{shlex.join(map(str, cmd))} exited {out.returncode}: {out.stderr.strip()}")
        return None
    return out.stdout.strip()


def require_apple_silicon():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise BenchError(
            f"this benchmark needs an Apple-silicon Mac (macOS, arm64); this is "
            f"{platform.system()} {platform.machine()}"
        )
    if not Path("/usr/bin/time").exists():
        raise BenchError("/usr/bin/time not found; it is needed for peak memory")


def lib_versions():
    names = ["mlx", "mlx-lm", "torch", "peft", "transformers", "tokenizers",
             "huggingface_hub", "datasets", "numpy", "safetensors"]
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def environment(ferrotensor_bin):
    commit = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = sh(["git", "-C", str(REPO), "status", "--porcelain"])
    ram = sh(["sysctl", "-n", "hw.memsize"])
    env = {
        "os": f"macOS {sh(['sw_vers', '-productVersion'])} ({sh(['sw_vers', '-buildVersion'])})",
        "kernel": platform.release(),
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_bytes": int(ram) if ram and ram.isdigit() else None,
        "python": platform.python_version(),
        "git_commit": commit,
        "git_dirty": bool(dirty) if commit else None,
        "rustc": sh(["rustc", "--version"]),
        "ferrotensor": sh([str(ferrotensor_bin), "--version"]) if Path(ferrotensor_bin).exists() else None,
        "libraries": lib_versions(),
    }
    return env


def thermal_snapshot():
    """Power source and thermal-warning text, kept next to each run as context."""
    therm = sh(["pmset", "-g", "therm"])
    batt = sh(["pmset", "-g", "batt"])
    return {"pmset_therm": therm, "pmset_batt": (batt or "").splitlines()[0] if batt else None}


# --------------------------------------------------------------------------
# data and models


def load_baseline():
    """Import benchmarks/baselines/pytorch_lora.py for its data helpers."""
    try:
        spec = importlib.util.spec_from_file_location("pytorch_lora_baseline", BASELINE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        raise BenchError(
            f"cannot import {BASELINE.name} ({e}); run this through scripts/bench_mac.sh, "
            f"which installs the pinned packages"
        ) from e
    return module


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def prepare_data(bl, cache, model_repo, want_mlx, base_dtype):
    """Download the model and data, write the shared files; return a description."""
    name = model_repo.split("/")[-1]
    model_dir = Path(bl.snapshot_download(model_repo, allow_patterns=["*.json", "*.safetensors"],
                                          local_dir=cache / "models" / name))
    train_rows, val_rows = bl.alpaca_split(cache)
    if len(train_rows) != TRAIN_EXAMPLES or len(val_rows) != HELDOUT:
        raise BenchError(f"expected {TRAIN_EXAMPLES} train and {HELDOUT} held-out examples, "
                         f"got {len(train_rows)} and {len(val_rows)}")
    # The files `ferrotensor train` reads: the same ones scripts/lora-run.sh writes.
    write_jsonl(cache / "alpaca" / "train.jsonl", train_rows)
    write_jsonl(cache / "alpaca" / "val.jsonl", val_rows)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    eos = tok.token_to_id("<|im_end|>")
    kept_train = [r for r in train_rows if bl.build_example(tok, eos, r["prompt"], r["response"], SEQ_LEN)]
    kept_val = [r for r in val_rows if bl.build_example(tok, eos, r["prompt"], r["response"], SEQ_LEN)]
    info = {
        "model_dir": str(model_dir),
        "train_file": str(cache / "alpaca" / "train.jsonl"),
        "val_file": str(cache / "alpaca" / "val.jsonl"),
        "train_examples": len(train_rows),
        "heldout_examples": len(val_rows),
        "train_examples_usable": len(kept_train),
        "heldout_examples_usable": len(kept_val),
        "mlx": None,
    }
    if want_mlx:
        info["mlx"] = prepare_mlx(bl, cache, name, model_dir, tok, eos, kept_train, kept_val, base_dtype)
    return info


def prepare_mlx(bl, cache, name, model_dir, tok, eos, kept_train, kept_val, base_dtype):
    """A ChatML copy of the model for mlx-lm, optionally converted to f32, and its data."""
    chatml = cache / "models" / f"{name}-chatml"
    chatml.mkdir(parents=True, exist_ok=True)
    for f in model_dir.iterdir():
        if f.suffix == ".json" and f.name != "tokenizer_config.json":
            shutil.copyfile(f, chatml / f.name)
        elif f.suffix == ".safetensors":
            link = chatml / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    cfg_path = model_dir / "tokenizer_config.json"
    tok_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    tok_cfg["chat_template"] = MLX_CHAT_TEMPLATE
    (chatml / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2) + "\n")

    mlx_model = chatml
    if base_dtype == "f32":
        mlx_model = cache / "models" / f"{name}-mlx-f32"
        if not (mlx_model / "config.json").exists():
            if mlx_model.exists():
                shutil.rmtree(mlx_model)
            say(f"== converting {name} to float32 for mlx-lm (not timed)")
            cmd = [sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(chatml),
                   "--mlx-path", str(mlx_model), "--dtype", "float32"]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise BenchError(f"mlx_lm convert failed: {shlex.join(cmd)}")

    data = cache / "mlx_data"
    mlx_rows_train = [{"prompt": r["prompt"], "completion": r["response"]} for r in kept_train]
    mlx_rows_val = [{"prompt": r["prompt"], "completion": r["response"]} for r in kept_val]
    write_jsonl(data / "train.jsonl", mlx_rows_train)
    write_jsonl(data / "valid.jsonl", mlx_rows_val)
    write_jsonl(data / "test.jsonl", mlx_rows_val)

    check = check_chat_template(bl, mlx_model, tok, eos, kept_train + kept_val)
    return {"model_dir": str(mlx_model), "data_dir": str(data), "template_check": check}


def check_chat_template(bl, mlx_model, tok, eos, rows):
    """Compare mlx-lm's tokenization with the baseline's, example by example."""
    from transformers import AutoTokenizer
    hf = AutoTokenizer.from_pretrained(str(mlx_model))
    mismatches = []
    for i, r in enumerate(rows):
        base = bl.build_example(tok, eos, r["prompt"], r["response"], SEQ_LEN)
        inputs, targets, mask = base
        base_full = inputs + targets[-1:]
        base_offset = mask.index(1) + 1
        messages = [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["response"]}]
        full = hf.apply_chat_template(messages, return_dict=False)[: SEQ_LEN + 1]
        offset = len(hf.apply_chat_template(messages[:1], add_generation_prompt=True, return_dict=False))
        if full != base_full or offset != base_offset:
            mismatches.append({"index": i, "tokens_equal": full == base_full,
                               "prompt_length_mlx": offset, "prompt_length_baseline": base_offset})
    return {"examples_checked": len(rows), "mismatches": len(mismatches), "first_mismatches": mismatches[:3]}


# --------------------------------------------------------------------------
# commands


def warmup_for(steps):
    return min(LR_WARMUP, steps - 1)


def ferrotensor_command(binary, model_dir, train_file, val_file, run_dir, steps, base_dtype):
    cmd = [str(binary), "train", "--model", str(model_dir)]
    if base_dtype == "f32":
        cmd.append("--f32-weights")
    cmd += [
        "--device", "gpu",
        "--data", str(train_file), "--eval-data", str(val_file),
        "--eval-samples", str(HELDOUT), "--eval-every", "0",
        "--out", str(run_dir), "--seq-len", str(SEQ_LEN),
        "--lr", repr(LR), "--warmup-steps", str(warmup_for(steps)),
        "--batch-size", str(BATCH_SIZE), "--grad-accum", str(GRAD_ACCUM),
        "--max-steps", str(steps),
        "--grad-clip", str(GRAD_CLIP), "--weight-decay", str(WEIGHT_DECAY),
        "--lora-rank", str(LORA_RANK), "--lora-alpha", str(LORA_ALPHA),
        "--lora-targets", ",".join(TARGETS),
        "--seed", str(SEED), "--log-every", "1", "--save-every", "0",
    ]
    return cmd


def pytorch_command(model_repo, cache, run_dir, steps):
    return [
        sys.executable, str(BASELINE), model_repo,
        "--device", "mps", "--dtype", "f32",
        "--steps", str(steps), "--seq-len", str(SEQ_LEN),
        "--batch-size", str(BATCH_SIZE), "--grad-accum", str(GRAD_ACCUM),
        "--warmup-steps", str(warmup_for(steps)), "--seed", str(SEED),
        "--data-root", str(cache), "--summary", str(run_dir / "summary.json"), "--record-steps",
    ]


def yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = repr(v)
        if "e" in s or "E" in s:
            s = f"{v:.12f}".rstrip("0")
            if s.endswith("."):
                s += "0"
        return s
    if isinstance(v, int):
        return str(v)
    return json.dumps(v)  # a JSON string is a valid YAML double-quoted scalar


def to_yaml(obj, indent=0):
    lines = []
    pad = "  " * indent
    for k, v in obj.items():
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(to_yaml(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{pad}{k}: [{', '.join(yaml_scalar(x) for x in v)}]")
        else:
            lines.append(f"{pad}{k}: {yaml_scalar(v)}")
    return "\n".join(lines)


def mlx_config(model_dir, data_dir, run_dir, steps, num_layers):
    iters = steps * GRAD_ACCUM  # mlx-lm counts micro-batches; BATCH_SIZE is 1
    warmup = warmup_for(steps)
    schedule = {"name": "cosine_decay"}
    if warmup > 0:
        schedule.update({"warmup": warmup, "warmup_init": 0.0})
    schedule["arguments"] = [LR, steps - warmup, LR * 0.1]
    return {
        "model": str(model_dir),
        "data": str(data_dir),
        "train": True,
        "test": True,
        "fine_tune_type": "lora",
        "mask_prompt": True,
        "num_layers": num_layers,
        "batch_size": BATCH_SIZE,
        "grad_accumulation_steps": GRAD_ACCUM,
        "iters": iters,
        "max_seq_length": SEQ_LEN + 1,  # mlx-lm cuts at max_seq_length tokens; the others cut inputs at 256
        "learning_rate": LR,
        "lr_schedule": schedule,
        "optimizer": "adamw",
        "optimizer_config": {"adamw": {"betas": ADAM_BETAS, "eps": ADAM_EPS,
                                       "weight_decay": WEIGHT_DECAY, "bias_correction": True}},
        "lora_parameters": {"rank": LORA_RANK, "scale": LORA_ALPHA / LORA_RANK, "dropout": 0.0,
                            "keys": [f"self_attn.{t}" for t in TARGETS]},
        "steps_per_report": GRAD_ACCUM,  # one report per optimizer step
        "steps_per_eval": iters + 1,  # only the evaluation at iteration 1 (and the last)
        "val_batches": -1,
        "test_batches": -1,
        "save_every": iters + 1,
        "seed": SEED,
        "adapter_path": str(run_dir / "adapters"),
    }


def mlx_command(config_path, metrics_path):
    return [sys.executable, str(Path(__file__).resolve()), "_mlx-worker", str(metrics_path),
            "--config", str(config_path)]


# --------------------------------------------------------------------------
# running and parsing


def run_timed(cmd, log_path, tag, env=None):
    """Run cmd under /usr/bin/time -l, tee its output to log_path; return the time report."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    time_path = log_path.with_suffix(".time")
    full = ["/usr/bin/time", "-l", "-o", str(time_path), *cmd]
    say(f"$ {shlex.join(cmd)}")
    tail = []
    start = time.perf_counter()
    with open(log_path, "w") as log:
        p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace", env=env, cwd=REPO)
        for line in p.stdout:
            log.write(line)
            tail.append(line)
            del tail[:-30]
            sys.stderr.write(f"[{tag}] {line}")
        code = p.wait()
    wall = time.perf_counter() - start
    if code != 0:
        raise BenchError(f"{tag} exited with status {code}; last output:\n" + "".join(tail) +
                         f"(full log: {log_path})")
    if not time_path.exists():
        raise BenchError(f"{tag}: /usr/bin/time wrote no report to {time_path}")
    return time_path.read_text(), wall


def parse_time_l(text, tag):
    """Maximum resident set size and peak memory footprint, in bytes, from `time -l`."""
    rss = re.search(r"^\s*(\d+)\s+maximum resident set size", text, re.M)
    if not rss:
        raise BenchError(f"{tag}: no 'maximum resident set size' in the /usr/bin/time -l report:\n{text}")
    foot = re.search(r"^\s*(\d+)\s+peak memory footprint", text, re.M)
    return int(rss.group(1)), int(foot.group(1)) if foot else None


def finite_list(values, n, what, tag):
    if not isinstance(values, list) or len(values) != n:
        got = len(values) if isinstance(values, list) else type(values).__name__
        raise BenchError(f"{tag}: expected {n} values for {what}, found {got}")
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise BenchError(f"{tag}: non-finite value in {what}: {v!r}")
    return [float(v) for v in values]


def finite(value, what, tag):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BenchError(f"{tag}: missing or non-finite {what}: {value!r}")
    return float(value)


def parse_ferrotensor(run_dir, log_text, steps, tag):
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise BenchError(f"{tag}: {path} was not written")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    train = [r for r in rows if "loss" in r and "seconds" in r]
    evals = [r["eval_loss"] for r in rows if "eval_loss" in r]
    if len(evals) != 2:
        raise BenchError(f"{tag}: expected 2 held-out evaluations (before, after), found {len(evals)}")
    params = re.search(r"\((\d+) trainable params\)", log_text)
    device = re.search(r"^device:\s*(.+)$", log_text, re.M)
    return {
        "step_seconds": finite_list([r["seconds"] for r in train], steps, "step seconds", tag),
        "step_losses": finite_list([r["loss"] for r in train], steps, "step losses", tag),
        "heldout_before": finite(evals[0], "held-out loss before", tag),
        "heldout_after": finite(evals[1], "held-out loss after", tag),
        "trainable_params": int(params.group(1)) if params else None,
        "device": device.group(1).strip() if device else None,
        "gpu_peak_bytes": None,
        "gpu_peak_label": "not exposed by the ferrotensor CLI",
    }


def parse_mlx(metrics_path, steps, tag):
    if not metrics_path.exists():
        raise BenchError(f"{tag}: the mlx worker wrote no metrics file {metrics_path}")
    rec = json.loads(metrics_path.read_text())
    reports = sorted(rec["train"], key=lambda r: r["iteration"])
    if len(reports) != steps:
        raise BenchError(f"{tag}: expected {steps} training reports (one per optimizer step), "
                         f"found {len(reports)}")
    for r in reports:
        if not (r["iterations_per_second"] > 0):
            raise BenchError(f"{tag}: bad iterations_per_second in {r}")
    first_val = [v for v in rec["val"] if v["iteration"] == 0]
    if not first_val:
        raise BenchError(f"{tag}: mlx-lm did not report the held-out loss at iteration 0")
    return {
        # One report covers GRAD_ACCUM micro-iterations, which is one optimizer step.
        "step_seconds": finite_list([GRAD_ACCUM / r["iterations_per_second"] for r in reports],
                                    steps, "step seconds", tag),
        "step_losses": finite_list([r["train_loss"] for r in reports], steps, "step losses", tag),
        "heldout_before": finite(first_val[0]["val_loss"], "held-out loss before", tag),
        "heldout_after": finite(rec.get("test_loss"), "held-out loss after (mlx-lm --test)", tag),
        "trainable_params": rec.get("trainable_params"),
        "device": "Apple GPU (MLX)",
        "gpu_peak_bytes": int(finite(rec.get("peak_memory_bytes"), "mx.get_peak_memory()", tag)),
        "gpu_peak_label": "mlx.core.get_peak_memory(), whole process",
    }


def parse_pytorch(summary_path, steps, tag):
    if not summary_path.exists():
        raise BenchError(f"{tag}: {summary_path} was not written")
    s = json.loads(summary_path.read_text())
    driver = finite(s.get("mps_driver_allocated_max_gb"), "mps_driver_allocated_max_gb", tag)
    return {
        "step_seconds": finite_list(s.get("step_seconds"), steps, "step seconds", tag),
        "step_losses": finite_list(s.get("step_losses"), steps, "step losses", tag),
        "heldout_before": finite(s.get("heldout_loss_before"), "held-out loss before", tag),
        "heldout_after": finite(s.get("heldout_loss_after"), "held-out loss after", tag),
        "trainable_params": s.get("trainable_params"),
        "device": s.get("device"),
        "gpu_peak_bytes": int(driver * 1e9),
        "gpu_peak_label": "torch.mps.driver_allocated_memory(), highest value sampled after each step",
    }


def run_stack(stack, repeat, args, prep, out, steps):
    tag = f"{stack}#{repeat}"
    run_dir = out / "runs" / f"{stack}-r{repeat}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    cache = out / "cache"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    thermal = thermal_snapshot()
    if stack == "ferrotensor":
        cmd = ferrotensor_command(args.ferrotensor_bin, prep["model_dir"], prep["train_file"],
                                  prep["val_file"], run_dir, steps, args.base_dtype)
        report, wall = run_timed(cmd, run_dir / "run.log", tag, env)
        parsed = parse_ferrotensor(run_dir, (run_dir / "run.log").read_text(), steps, tag)
        extra = {}
    elif stack == "mlx":
        layers = json.loads((Path(prep["mlx"]["model_dir"]) / "config.json").read_text())["num_hidden_layers"]
        cfg = mlx_config(prep["mlx"]["model_dir"], prep["mlx"]["data_dir"], run_dir, steps, layers)
        cfg_path = run_dir / "mlx_config.yaml"
        cfg_path.write_text(to_yaml(cfg) + "\n")
        metrics_path = run_dir / "mlx_metrics.json"
        cmd = mlx_command(cfg_path, metrics_path)
        report, wall = run_timed(cmd, run_dir / "run.log", tag, env)
        parsed = parse_mlx(metrics_path, steps, tag)
        extra = {"config_file": str(cfg_path), "config": cfg}
    else:
        cmd = pytorch_command(args.model, cache, run_dir, steps)
        report, wall = run_timed(cmd, run_dir / "run.log", tag, env)
        parsed = parse_pytorch(run_dir / "summary.json", steps, tag)
        extra = {}
    rss, foot = parse_time_l(report, tag)
    usable = parsed["step_seconds"][args.discard_steps:]
    window = parsed["step_losses"][-args.final_window:]
    parsed.update({
        "stack": stack,
        "repeat": repeat,
        "command": shlex.join(cmd),
        "wall_seconds": wall,
        "max_rss_bytes": rss,
        "peak_footprint_bytes": foot,
        "seconds_per_step_median": statistics.median(usable),
        "seconds_per_step_steps_used": len(usable),
        "final_train_loss": statistics.fmean(window),
        "final_train_loss_steps": len(window),
        "thermal": thermal,
        "time_l_report": report,
        "log": str(run_dir / "run.log"),
        **extra,
    })
    return parsed


# --------------------------------------------------------------------------
# aggregation and output


def aggregate(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values}


def summarize(runs):
    out = {}
    for stack in STACKS:
        mine = [r for r in runs if r["stack"] == stack]
        if not mine:
            continue
        s = {}
        for key in ("seconds_per_step_median", "max_rss_bytes", "final_train_loss",
                    "heldout_before", "heldout_after", "wall_seconds"):
            s[key] = aggregate([r[key] for r in mine])
        foot = [r["peak_footprint_bytes"] for r in mine]
        s["peak_footprint_bytes"] = aggregate(foot) if all(v is not None for v in foot) else None
        gpu = [r["gpu_peak_bytes"] for r in mine]
        s["gpu_peak_bytes"] = aggregate(gpu) if all(v is not None for v in gpu) else None
        s["gpu_peak_label"] = mine[0]["gpu_peak_label"]
        s["trainable_params"] = mine[0]["trainable_params"]
        out[stack] = s
    return out


def cell(agg, fmt, scale=1.0, repeats=1):
    if agg is None:
        return "n/a"
    f = lambda v: format(v / scale, fmt)
    if repeats == 1:
        return f(agg["median"])
    return f"{f(agg['median'])} ({f(agg['min'])} to {f(agg['max'])})"


def known_differences(args, prep, summary):
    d = []
    if args.base_dtype == "f32":
        d.append("Base-weight precision: f32 in all three (ferrotensor `--f32-weights`; the model converted "
                 "to float32 with `mlx_lm convert --dtype float32`; PyTorch float32). Ferrotensor's CLI "
                 "default stores frozen weights as f16; `--base-dtype native` runs that default.")
    else:
        d.append("Base-weight precision differs by stack (`--base-dtype native`): ferrotensor stores frozen "
                 "weights as f16 and computes in f32; mlx-lm runs the checkpoint's own dtype (bfloat16, "
                 "computing in bfloat16); PyTorch stays float32 because the baseline script has no "
                 "other mode on MPS. Do not compare losses or memory across stacks without that in mind.")
    d.append("LoRA adapters are f32 in all three. Initialization is A ~ U(+-1/sqrt(in)), B = 0 in all three, "
             "drawn from each stack's own random generator.")
    d.append("Data order: each stack shuffles with its own seeded generator, so the 8 sequences in a given step "
             "differ between stacks. The final training loss is therefore over different examples and is "
             "only roughly comparable. The held-out set is the same 64 examples for all three.")
    d.append("Gradient clipping: ferrotensor and PyTorch clip the global gradient norm at 1.0. mlx-lm's trainer "
             "has no gradient clipping, so the mlx run is unclipped.")
    d.append("Learning-rate schedule: mlx-lm's schedule is built from `warmup` + `cosine_decay` and starts at "
             "lr 0 for the first update, so it lags the other two by one optimizer step.")
    d.append("Logged training loss: ferrotensor weights each of the 8 sequences by its response-token count; "
             "mlx-lm and PyTorch report the plain mean of the 8 per-sequence mean losses. Held-out loss is the "
             "mean over all held-out response tokens in all three.")
    d.append("Padding: mlx-lm pads every batch to 1 + a multiple of 32 tokens (its own rule); ferrotensor and "
             "PyTorch run each sequence at its own length at batch 1. `max_seq_length` is 257 for mlx-lm so "
             "that sequences are cut to 256 input tokens like the others.")
    d.append("Optimizer: AdamW with betas (0.9, 0.999), eps 1e-8, no weight decay in all three; for mlx-lm "
             "`bias_correction: true` is set because its default is off.")
    d.append("mlx-lm is given a local copy of the model whose tokenizer_config.json has a ChatML chat template "
             "(the base checkpoint ships none), and the loss is restricted to the response with `mask_prompt`.")
    tc = (prep.get("mlx") or {}).get("template_check")
    if tc:
        d.append(f"Token-level check of that template against the baseline's formatting: "
                 f"{tc['mismatches']} mismatching examples out of {tc['examples_checked']}.")
    d.append("What each step timer covers: ferrotensor, the wall time between its log lines (one optimizer "
             "step, including batch assembly); mlx-lm, its own training timer over 8 micro-iterations "
             "(excludes batch assembly and its evaluations); PyTorch, perf_counter around the step with "
             "`torch.mps.synchronize()` (includes batch assembly).")
    d.append("mlx-lm compiles its step per input shape (`mx.compile`), and batches are padded to a few "
             "different lengths, so an occasional slow step can appear after the warm-up. The median ignores "
             "it; every step's time is in the JSON file.")
    return d


def render(args, steps, env, prep, runs, summary):
    stacks = [s for s in STACKS if s in summary]
    n = args.repeats
    used = steps - args.discard_steps
    lines = []
    lines.append(f"## Apple-silicon LoRA benchmark: {args.model}, {steps} optimizer steps, "
                 f"{n} repeat{'s' if n != 1 else ''}")
    lines.append("")
    ram = f"{env['ram_bytes'] / 2**30:.0f} GiB" if env["ram_bytes"] else "unknown RAM"
    lines.append(f"{env['chip'] or 'unknown chip'}, {ram}, {env['os']}; ferrotensor commit "
                 f"{(env['git_commit'] or 'unknown')[:10]}{' (dirty tree)' if env['git_dirty'] else ''}. "
                 f"{'Median over repeats, with the range in parentheses.' if n > 1 else 'One run per stack; one run is not a distribution.'}")
    lines.append("")
    lines.append("| | " + " | ".join(LABELS[s] for s in stacks) + " |")
    lines.append("|---|" + "---:|" * len(stacks))

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(summary[s]) for s in stacks) + " |")

    row(f"Seconds per optimizer step (median of {used} steps after {args.discard_steps} warm-up)",
        lambda s: cell(s["seconds_per_step_median"], ".3f", 1, n))
    row("Peak resident set size, GB (`/usr/bin/time -l`, whole process)",
        lambda s: cell(s["max_rss_bytes"], ".2f", 1e9, n))
    row("Peak memory footprint, GB (`/usr/bin/time -l`)",
        lambda s: cell(s["peak_footprint_bytes"], ".2f", 1e9, n))
    row("GPU memory reported by the library, GB (see Measurement)",
        lambda s: cell(s["gpu_peak_bytes"], ".2f", 1e9, n))
    row(f"Final training loss (mean of last {args.final_window} steps)",
        lambda s: cell(s["final_train_loss"], ".4f", 1, n))
    row("Held-out loss before training",
        lambda s: cell(s["heldout_before"], ".4f", 1, n))
    row("Held-out loss after training",
        lambda s: cell(s["heldout_after"], ".4f", 1, n))
    row("Wall time of the whole run, s (includes loading and evaluations)",
        lambda s: cell(s["wall_seconds"], ".1f", 1, n))
    lines.append("")

    befores = [summary[s]["heldout_before"]["median"] for s in stacks]
    if len(befores) > 1:
        spread = max(befores) - min(befores)
        verdict = "as expected for identical forward passes" if spread < 0.01 else \
            "LARGER than expected for identical forward passes: check the base-weight precision and the data"
        lines.append(f"Held-out loss before training differs by {spread:.4f} across stacks, {verdict}.")
        lines.append("")
    params = {LABELS[s]: summary[s]["trainable_params"] for s in stacks if summary[s]["trainable_params"]}
    if params:
        lines.append("Trainable parameters: " + ", ".join(f"{k} {v:,}" for k, v in params.items()) + ".")
        lines.append("")

    lines.append("### Recipe")
    lines.append("")
    lines.append(f"- Model {args.model}; {prep['train_examples_usable']} training and "
                 f"{prep['heldout_examples_usable']} held-out examples (yahma/alpaca-cleaned, seed-0 shuffle, "
                 f"ChatML, response-only loss).")
    lines.append(f"- LoRA rank {LORA_RANK:g}, alpha {LORA_ALPHA:g} on {', '.join(TARGETS)} of every layer, f32 adapters.")
    lines.append(f"- AdamW lr {LR:g}, {warmup_for(steps)} warmup steps then cosine to lr/10, weight decay 0; "
                 f"batch {BATCH_SIZE} x {GRAD_ACCUM} accumulation = {BATCH_SIZE * GRAD_ACCUM} sequences per step; "
                 f"sequence length {SEQ_LEN}; {steps} steps; base weights {args.base_dtype}.")
    lines.append("")
    lines.append("### Measurement")
    lines.append("")
    lines.append(f"- Seconds per step: each stack's own per-step timing; the first {args.discard_steps} steps "
                 f"(shader and kernel compilation, allocator warm-up) are dropped, the median of the remaining "
                 f"{used} is reported.")
    lines.append("- Peak resident set size and peak memory footprint: `/usr/bin/time -l` around the whole "
                 "command, so they include model loading, the held-out evaluations and, for mlx-lm and "
                 "PyTorch, the Python interpreter and its libraries. The value is the largest among the "
                 "processes the command waited for, not a sum. On unified memory the footprint is the better "
                 "guide to what the run costs the machine.")
    for s in stacks:
        lines.append(f"- GPU memory, {LABELS[s]}: {summary[s]['gpu_peak_label']}.")
    lines.append(f"- Final training loss: mean of the logged loss of the last {args.final_window} steps "
                 f"({args.final_window * BATCH_SIZE * GRAD_ACCUM} sequences).")
    lines.append("")
    lines.append("### Known differences between the stacks")
    lines.append("")
    for d in known_differences(args, prep, summary):
        lines.append(f"- {d}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# mlx worker: runs mlx_lm.lora with a recording callback


def mlx_worker(metrics_path, argv):
    """Run `mlx_lm.lora` unchanged and record its reports at full precision.

    mlx-lm prints progress through a terminal UI, so its numbers are taken from
    the training callback instead. The trainer itself is not modified.
    """
    import mlx.core as mx
    import mlx_lm.lora as lora
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.callbacks import TrainingCallback

    rec = {"train": [], "val": [], "test_loss": None, "trainable_params": None}

    class Recorder(TrainingCallback):
        def on_train_loss_report(self, info):
            rec["train"].append({
                "iteration": int(info["iteration"]),
                "train_loss": float(info["train_loss"]),
                "iterations_per_second": float(info["iterations_per_second"]),
                "peak_memory_gb": float(info["peak_memory"]),
            })

        def on_val_loss_report(self, info):
            rec["val"].append({"iteration": int(info["iteration"]), "val_loss": float(info["val_loss"])})

    lora.get_reporting_callbacks = lambda *a, **k: Recorder()
    real_evaluate = lora.evaluate

    def evaluate(*a, **k):
        model = k.get("model")
        if model is not None:
            rec["trainable_params"] = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        loss = real_evaluate(*a, **k)
        rec["test_loss"] = float(loss)
        return loss

    lora.evaluate = evaluate  # only evaluate_model (the --test pass) looks this name up
    sys.argv = ["mlx_lm.lora", *argv]
    lora.main()
    rec["peak_memory_bytes"] = int(mx.get_peak_memory())
    Path(metrics_path).write_text(json.dumps(rec))


# --------------------------------------------------------------------------


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Compare Ferrotensor, mlx-lm and PyTorch + PEFT (MPS) on one LoRA recipe. "
                    "Usually started through scripts/bench_mac.sh.",
        epilog="Fixed recipe (see docs/benchmarks-mac.md): SmolLM2-135M, rank 8, alpha 16, q/k/v/o, "
               "lr 5e-4, 20 warmup steps, batch 1 x 8 accumulation, sequence length 256.")
    ap.add_argument("--steps", type=int, default=250, help="optimizer steps per run (default 250, one epoch)")
    ap.add_argument("--only", default="ferrotensor,mlx,pytorch",
                    help="comma-separated subset of ferrotensor,mlx,pytorch (default: all three)")
    ap.add_argument("--out", default=str(REPO / "data" / "bench-mac"),
                    help="output directory for caches, logs, results.json and results.md "
                         "(default: data/bench-mac, which is gitignored)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per stack; repeats are interleaved (default 1)")
    ap.add_argument("--discard-steps", type=int, default=3,
                    help="steps dropped from the timing as warm-up (default 3)")
    ap.add_argument("--final-window", type=int, default=10,
                    help="steps averaged for the final training loss (default 10)")
    ap.add_argument("--base-dtype", choices=["f32", "native"], default="f32",
                    help="f32: frozen weights f32 in every stack (default); native: each stack's own default")
    ap.add_argument("--cooldown", type=int, default=30,
                    help="seconds to wait between runs so the chip can cool (default 30)")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="Hugging Face repo id")
    ap.add_argument("--ferrotensor-bin", default=str(REPO / "target" / "release" / "ferrotensor"),
                    help="path of the release `ferrotensor` binary")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the three commands and the mlx-lm config, then exit; nothing is "
                         "downloaded or run")
    args = ap.parse_args(argv)
    args.only = [s.strip() for s in args.only.split(",") if s.strip()]
    for s in args.only:
        if s not in STACKS:
            ap.error(f"--only: unknown stack {s!r}; choose from {', '.join(STACKS)}")
    if args.steps < 5 or args.repeats < 1 or args.discard_steps < 0 or args.final_window < 1:
        ap.error("--steps must be at least 5, --repeats at least 1, --discard-steps at least 0, "
                 "--final-window at least 1")
    if args.discard_steps >= args.steps:
        ap.error("--discard-steps must be smaller than --steps")
    if args.final_window > args.steps:
        args.final_window = args.steps
    return args


def dry_run(args):
    out = Path(args.out)
    cache = out / "cache"
    name = args.model.split("/")[-1]
    model_dir = cache / "models" / name
    run = out / "runs" / "<stack>-r<k>"
    steps = args.steps
    print("# commands (each is run under /usr/bin/time -l)\n")
    print("ferrotensor:\n  " + shlex.join(ferrotensor_command(
        args.ferrotensor_bin, model_dir, cache / "alpaca" / "train.jsonl", cache / "alpaca" / "val.jsonl",
        run, steps, args.base_dtype)) + "\n")
    print("pytorch:\n  " + shlex.join(pytorch_command(args.model, cache, run, steps)) + "\n")
    mlx_model = cache / "models" / (f"{name}-mlx-f32" if args.base_dtype == "f32" else f"{name}-chatml")
    cfg = mlx_config(mlx_model, cache / "mlx_data", run, steps, 30)
    print("mlx (config.yaml, then the command; mlx_lm.lora is run through a recording wrapper):\n")
    print(to_yaml(cfg))
    print("\n  " + shlex.join(mlx_command(run / "mlx_config.yaml", run / "mlx_metrics.json")))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_mlx-worker":
        mlx_worker(argv[1], argv[2:])
        return 0
    args = parse_args(argv)
    if args.dry_run:
        dry_run(args)
        return 0
    require_apple_silicon()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    steps = args.steps
    args.ferrotensor_bin = str(Path(args.ferrotensor_bin).resolve())
    if "ferrotensor" in args.only and not Path(args.ferrotensor_bin).exists():
        raise BenchError(f"{args.ferrotensor_bin} not found; build it with "
                         f"`cargo build --release -p ferrotensor-cli` (scripts/bench_mac.sh does)")

    say("== preparing the model and data (downloads on first use; not timed)")
    bl = load_baseline()
    prep = prepare_data(bl, cache, args.model, "mlx" in args.only, args.base_dtype)
    env = environment(args.ferrotensor_bin)
    tc = (prep["mlx"] or {}).get("template_check")
    if tc and tc["mismatches"]:
        say(f"warning: the mlx ChatML template differs from the baseline's formatting on "
            f"{tc['mismatches']} of {tc['examples_checked']} examples; this is reported in the output")

    runs = []
    partial = out / "results.partial.json"
    try:
        for repeat in range(1, args.repeats + 1):
            for stack in args.only:
                if runs:
                    say(f"== cooling down for {args.cooldown} s")
                    time.sleep(args.cooldown)
                say(f"== {LABELS[stack]}, repeat {repeat} of {args.repeats}")
                runs.append(run_stack(stack, repeat, args, prep, out, steps))
    except BenchError:
        if runs:
            partial.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
            say(f"(raw numbers of the runs that finished: {partial})")
        raise

    summary = summarize(runs)
    markdown = render(args, steps, env, prep, runs, summary)
    result = {
        "schema": 1,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command_line": [sys.executable, *sys.argv],
        "settings": {
            "model": args.model, "steps": steps, "repeats": args.repeats, "only": args.only,
            "discard_steps": args.discard_steps, "final_window": args.final_window,
            "base_dtype": args.base_dtype, "cooldown_seconds": args.cooldown,
        },
        "recipe": {
            "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "lr": LR,
            "lr_warmup_steps": warmup_for(steps), "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
            "lora_targets": TARGETS, "grad_clip": GRAD_CLIP, "weight_decay": WEIGHT_DECAY,
            "adam_betas": ADAM_BETAS, "adam_eps": ADAM_EPS, "seed": SEED,
            "train_examples": prep["train_examples"], "heldout_examples": prep["heldout_examples"],
            "train_examples_usable": prep["train_examples_usable"],
            "heldout_examples_usable": prep["heldout_examples_usable"],
        },
        "environment": env,
        "mlx_template_check": tc,
        "commands": {r["stack"] + f"-r{r['repeat']}": r["command"] for r in runs},
        "summary": summary,
        "runs": runs,
        "known_differences": known_differences(args, prep, summary),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "results.md").write_text(markdown + "\n")
    if partial.exists():
        partial.unlink()
    print(markdown)
    say(f"raw numbers: {out / 'results.json'}")
    say(f"table:       {out / 'results.md'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BenchError as e:
        say(f"error: {e}")
        sys.exit(1)

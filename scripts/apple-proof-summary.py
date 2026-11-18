#!/usr/bin/env python3
"""Collect the Apple silicon LoRA proof (.github/workflows/apple-proof.yml).

    apple-proof-summary.py PROOF_DIR          write PROOF_DIR/summary.json, print Markdown
    apple-proof-summary.py --check MODEL_DIR  exit 0 when that model's run passed

Reads what the workflow left in PROOF_DIR: hardware.json, info.json,
tried.txt, and per model plan.json, finetune.json, the run's metrics.jsonl
and memory_report.json, eval.json, generate.json, each command's .exit
file and its stderr with the /usr/bin/time -l report.
"""

import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

MIB = 1024 * 1024


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def read_text(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def exit_code(path):
    text = read_text(path)
    try:
        return int(text.strip()) if text else None
    except ValueError:
        return None


def time_report(stderr):
    """The numbers of the /usr/bin/time -l report at the end of `stderr`."""
    if not stderr:
        return None
    out = {}
    m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", stderr)
    if m:
        out["real_seconds"] = float(m.group(1))
        out["user_seconds"] = float(m.group(2))
        out["sys_seconds"] = float(m.group(3))
    for key, label in (
        ("max_rss_bytes", "maximum resident set size"),
        ("peak_memory_footprint_bytes", "peak memory footprint"),
        ("page_reclaims", "page reclaims"),
        ("page_faults", "page faults"),
        ("swaps", "swaps"),
        ("involuntary_context_switches", "involuntary context switches"),
    ):
        m = re.search(r"(\d+)\s+" + re.escape(label), stderr)
        if m:
            out[key] = int(m.group(1))
    for key in ("max_rss_bytes", "peak_memory_footprint_bytes"):
        if key in out:
            out[key.replace("_bytes", "_mib")] = round(out[key] / MIB)
    out["terminated_abnormally"] = "terminated abnormally" in stderr or "Alarm clock" in stderr
    out["panicked"] = "panicked" in stderr
    out["gpu_device_lost_hint"] = "GPU device was lost" in stderr
    return out


def memory_report(path):
    """Peak GPU memory by category over the recorded phases."""
    data = read_json(path)
    if not data or not data.get("reports"):
        return None
    last = data["reports"][-1]
    peak = last.get("overall_peak", {})
    categories = {
        name: round(c.get("allocated_bytes", 0) / MIB, 1)
        for name, c in peak.get("categories", {}).items()
        if c.get("allocated_bytes")
    }
    return {
        "phases": [r.get("label") for r in data["reports"]],
        "overall_peak_total_mib": round(last.get("overall_peak_total_bytes", 0) / MIB, 1),
        "overall_peak_live_mib": round(peak.get("live_bytes", 0) / MIB, 1),
        "overall_peak_held_mib": round(last.get("overall_peak_held_bytes", 0) / MIB, 1),
        "overall_peak_awaiting_gpu_mib": round(
            last.get("overall_peak_awaiting_gpu_bytes", 0) / MIB, 1
        ),
        "categories_at_peak_mib": categories,
    }


def metrics(path):
    text = read_text(path)
    if not text:
        return []
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "loss" in row and "seconds" in row:
            rows.append(row)
    return rows


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def stage_reached(stderr, steps_done):
    """The last stage a training run reached, from its stderr."""
    if not stderr:
        return "not started"
    if steps_done:
        return f"optimizer step {steps_done}"
    if "eval before" in stderr:
        return "evaluation before training"
    if "after model load" in stderr:
        return "after model load"
    return "model load"


def training(run_dir, tag):
    """One `finetune` run: `tag` is finetune or finetune-force."""
    code = exit_code(run_dir / f"{tag}.exit")
    if code is None:
        return None
    stdout = read_json(run_dir / f"{tag}.json") or {}
    stderr = read_text(run_dir / f"{tag}.stderr") or ""
    out_dir = run_dir / ("out-force" if tag == "finetune-force" else "out")
    rows = metrics(out_dir / "metrics.jsonl")
    seconds = [r["seconds"] for r in rows]
    return {
        "command": tag,
        "exit": code,
        "error": stdout.get("error"),
        "chosen": stdout.get("chosen"),
        "grad_accum": stdout.get("grad_accum"),
        "budget_mib": stdout.get("budget_mib"),
        "estimated_peak_mib": stdout.get("estimated_peak_mib"),
        "fits": stdout.get("fits"),
        "steps_completed": len(rows),
        "stage_reached": stage_reached(stderr, len(rows)),
        "losses": [round(r["loss"], 4) for r in rows],
        "first_step_seconds": round(seconds[0], 2) if seconds else None,
        "median_step_seconds": round(statistics.median(seconds), 2) if seconds else None,
        "median_step_seconds_after_first": (
            round(statistics.median(seconds[1:]), 2) if len(seconds) > 1 else None
        ),
        "eval_loss_before": stdout.get("eval_loss_before"),
        "eval_loss_after": stdout.get("eval_loss_after"),
        "measured_peak_mib": stdout.get("measured_peak_mib"),
        "train_seconds": stdout.get("seconds"),
        "gpu_memory_report": memory_report(out_dir / "memory_report.json"),
        "time": time_report(stderr),
        "stderr_tail": stderr.splitlines()[-12:] if code != 0 else None,
    }


def model_result(run_dir, steps):
    run_dir = Path(run_dir)
    plan = read_json(run_dir / "plan.json") or {}
    planned = training(run_dir, "finetune")
    forced = training(run_dir, "finetune-force")
    run = forced or planned
    ev = read_json(run_dir / "eval.json") or {}
    gen = read_json(run_dir / "generate.json") or {}
    eval_code = exit_code(run_dir / "eval.exit")
    gen_code = exit_code(run_dir / "generate.exit")
    checks = {
        "finetune_exit_0": bool(run) and run["exit"] == 0,
        "all_steps_completed": bool(run) and run["steps_completed"] == steps,
        "losses_finite": bool(run) and bool(run["losses"]) and all(map(finite, run["losses"])),
        "eval_lora_exit_0": eval_code == 0,
        "heldout_loss_finite": finite(ev.get("loss")),
        "adapter_reloaded": gen_code == 0 and bool(gen.get("text")),
    }
    return {
        "model": run_dir.name,
        "plan": {
            "exit": exit_code(run_dir / "plan.exit"),
            "adapter": plan.get("adapter"),
            "backend": plan.get("backend"),
            "device_type": plan.get("device_type"),
            "budget_mib": plan.get("budget_mib"),
            "budget_source": plan.get("budget_source"),
            "chosen": plan.get("chosen"),
            "estimated_peak_mib": plan.get("estimated_peak_mib"),
            "fits": plan.get("fits"),
            "error": plan.get("error"),
        },
        "planner_refused": bool(forced),
        "finetune": planned,
        "finetune_force": forced,
        "eval_lora": {
            "exit": eval_code,
            "loss": ev.get("loss"),
            "perplexity": ev.get("perplexity"),
            "examples": ev.get("examples"),
            "seconds": ev.get("seconds"),
            "error": ev.get("error"),
            "time": time_report(read_text(run_dir / "eval.stderr")),
        },
        "reload": {
            "exit": gen_code,
            "text": gen.get("text"),
            "new_tokens": gen.get("new_tokens"),
            "tokens_per_second": gen.get("tokens_per_second"),
            "error": gen.get("error"),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def selected_adapter(info):
    if not info:
        return None
    for a in info.get("adapters", []):
        if a.get("selected"):
            return {k: a.get(k) for k in ("name", "vendor", "device_type", "backend", "driver", "shader_f16")}
    return None


def fmt(x, unit=""):
    return "n/a" if x is None else f"{x}{unit}"


def markdown(summary):
    hw = summary["hardware"] or {}
    ad = summary["adapter"] or {}
    lines = [
        "## Apple silicon LoRA proof",
        "",
        f"Runner: {hw.get('model')}, {hw.get('cpu')}, {hw.get('ncpu')} CPUs, "
        f"{round(hw.get('memsize_bytes', 0) / 2**30, 2)} GiB RAM, macOS {hw.get('macos')}.",
        f"Adapter: {ad.get('name')} ({ad.get('backend')}, {ad.get('device_type')}). "
        f"Kernel self-check passed: {summary['self_check_passed']}.",
        "The GPU is paravirtualised; times say nothing about speed on a physical Mac.",
        "",
        "| Model | Planner budget | Planner choice | Predicted peak | Fits (plan) | "
        "Steps | Median s/step | Loss first → last | GPU peak (report) | Max RSS | "
        "Peak footprint | Held-out loss with adapter | Reload | Passed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["models"]:
        p = r["plan"]
        run = r["finetune_force"] or r["finetune"] or {}
        mem = run.get("gpu_memory_report") or {}
        t = run.get("time") or {}
        losses = run.get("losses") or []
        chosen = p.get("chosen") or {}
        choice = (
            f"batch {chosen.get('batch_size')}, seq {chosen.get('seq_len')}, "
            f"ckpt {'on' if chosen.get('checkpoint_blocks') else 'off'}"
            if chosen
            else "n/a"
        )
        lines.append(
            f"| {r['model']}{' (--force)' if r['planner_refused'] else ''} "
            f"| {fmt(p.get('budget_mib'), ' MiB')} | {choice} "
            f"| {fmt(p.get('estimated_peak_mib'), ' MiB')} | {fmt(p.get('fits'))} "
            f"| {run.get('steps_completed', 0)}/{summary['steps']} "
            f"| {fmt(run.get('median_step_seconds'))} "
            f"| {losses[0] if losses else 'n/a'} → {losses[-1] if losses else 'n/a'} "
            f"| {fmt(mem.get('overall_peak_total_mib'), ' MiB')} "
            f"| {fmt(t.get('max_rss_mib'), ' MiB')} "
            f"| {fmt(t.get('peak_memory_footprint_mib'), ' MiB')} "
            f"| {fmt(r['eval_lora'].get('loss'))} | {r['checks']['adapter_reloaded']} "
            f"| {r['passed']} |"
        )
    lines += ["", "```json", json.dumps(summary, indent=2), "```"]
    return "\n".join(lines)


def main(argv):
    steps = int(os.environ.get("STEPS", "20"))
    if len(argv) == 3 and argv[1] == "--check":
        return 0 if model_result(argv[2], steps)["passed"] else 1
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    proof = Path(argv[1])
    info = read_json(proof / "info.json")
    tried = (read_text(proof / "tried.txt") or "").split()
    models = [model_result(proof / repo.split("/")[-1], steps) for repo in tried]
    checks = (info or {}).get("checks") or []
    summary = {
        "steps": steps,
        "commit": os.environ.get("GITHUB_SHA"),
        "run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
            if "GITHUB_RUN_ID" in os.environ
            else None
        ),
        "hardware": read_json(proof / "hardware.json"),
        "adapter": selected_adapter(info),
        "self_check_passed": bool(checks) and all(c.get("passed") for c in checks),
        "gpu_note": "paravirtualised Apple GPU on a GitHub macOS arm64 runner; "
        "times say nothing about speed on a physical Mac",
        "tried": tried,
        "largest_passing_model": next((m["model"] for m in models if m["passed"]), None),
        "models": models,
        "passed": any(m["passed"] for m in models),
    }
    (proof / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

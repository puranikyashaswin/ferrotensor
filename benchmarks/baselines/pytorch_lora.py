#!/usr/bin/env python3
"""The LoRA walkthrough's recipe in PyTorch + PEFT, as a reference point.

Reproduces scripts/lora-run.sh as closely as two frameworks allow: the same
alpaca-cleaned split (2,000 train, 64 held out, seed-0 shuffle), the same
ChatML formatting and response-only loss, sequences cut at 256 tokens, LoRA
rank 8 / alpha 16 on q/k/v/o (A ~ U(+-1/sqrt(in)), B = 0), AdamW (0.9, 0.999,
1e-8, no weight decay), lr 5e-4 with 20 linear warmup steps then cosine decay
to lr/10, global gradient clipping at 1.0, and 8 sequences per optimizer step.
Each sequence's loss is its mean over its own response tokens; a step
averages the sequences. Held-out loss is the mean over all 2,744 held-out
response tokens, as `ferrotensor eval` computes it.

The data order differs (each framework has its own seeded shuffle), so the
loss after training matches only approximately; the loss before training
should match exactly.

    pip install torch transformers peft tokenizers huggingface_hub
    python benchmarks/baselines/pytorch_lora.py HuggingFaceTB/SmolLM2-135M --batch-size 8 --grad-accum 1

Writes data/runs/<model>-pytorch-<dtype>-b<batch>/summary.json.
"""

import argparse
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM


def alpaca_split(root: Path):
    """The walkthrough's split, built exactly as scripts/lora-run.sh does."""
    src = hf_hub_download("yahma/alpaca-cleaned", "alpaca_data_cleaned.json",
                          repo_type="dataset", local_dir=root / "alpaca-cleaned")
    rows = json.load(open(src))
    out = []
    for r in rows:
        p = r["instruction"].strip()
        if r.get("input", "").strip():
            p += "\n\n" + r["input"].strip()
        resp = r["output"].strip()
        if len(p) + len(resp) <= 500 and resp:
            out.append({"prompt": p, "response": resp})
    random.Random(0).shuffle(out)
    return out[64:2064], out[:64]


def build_example(tok: Tokenizer, eos: int, prompt: str, response: str, max_len: int):
    """Same as ferrotensor_trainer::build_sft_example with the chat template."""
    chat = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    p = tok.encode(chat, add_special_tokens=True).ids
    r = tok.encode(response, add_special_tokens=False).ids + [eos]
    if not p or len(p) > max_len:
        return None
    full = p + r
    inputs, targets = full[:-1], full[1:]
    mask = [0] * len(targets)
    for i in range(len(p) - 1, len(targets)):
        mask[i] = 1
    inputs, targets, mask = inputs[:max_len], targets[:max_len], mask[:max_len]
    if len(inputs) < 2 or 1 not in mask:
        return None
    return inputs, targets, mask


def lr_at(step: int, lr: float, warmup: int, total: int) -> float:
    """ferrotensor's WarmupCosineScheduler: optimizer step k (1-based) uses lr_at(k)."""
    if step <= warmup:
        return lr * step / max(warmup, 1)
    floor = lr * 0.1
    progress = (step - warmup) / max(total - warmup, 1)
    return floor + (lr - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def batch_tensors(examples, device):
    """Right-pad to the longest; padded targets are ignored through the mask."""
    n = max(len(e[0]) for e in examples)
    ids = torch.zeros(len(examples), n, dtype=torch.long)
    tgt = torch.zeros(len(examples), n, dtype=torch.long)
    msk = torch.zeros(len(examples), n)
    for i, (x, y, m) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x)
        tgt[i, :len(y)] = torch.tensor(y)
        msk[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids.to(device), tgt.to(device), msk.to(device)


def per_sequence_loss(model, examples, device, dtype):
    """Sum over sequences of each sequence's mean response-token loss, and token stats."""
    ids, tgt, msk = batch_tensors(examples, device)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        logits = model(input_ids=ids).logits.float()
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
    tokens = msk.sum(dim=1)
    seq_loss = (nll * msk).sum(dim=1) / tokens.clamp(min=1)
    return seq_loss, (nll * msk).sum(), tokens.sum()


@torch.no_grad()
def heldout(model, examples, device, dtype):
    total, tokens = 0.0, 0.0
    for e in examples:
        _, s, t = per_sequence_loss(model, [e], device, dtype)
        total += s.item()
        tokens += t.item()
    return total / tokens, int(tokens)


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return torch.cuda.get_device_name(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = "cuda"
    root = Path("data")
    name = args.model.split("/")[-1]
    run = root / "runs" / f"{name}-pytorch-{args.dtype}-b{args.batch_size}"
    run.mkdir(parents=True, exist_ok=True)

    model_dir = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"],
                                  local_dir=root / "models" / name)
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    eos = tok.token_to_id("<|im_end|>")
    train_rows, val_rows = alpaca_split(root)
    train = [e for r in train_rows if (e := build_example(tok, eos, r["prompt"], r["response"], args.seq_len))]
    val = [e for r in val_rows if (e := build_example(tok, eos, r["prompt"], r["response"], args.seq_len))]

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(device)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    before, held_tokens = heldout(model, val, device, dtype)
    print(f"eval before: loss {before:.4f} ({held_tokens} tokens)", flush=True)

    order = list(range(len(train)))
    random.Random(args.seed).shuffle(order)
    per_step = args.batch_size * args.grad_accum
    torch.cuda.reset_peak_memory_stats()
    step_times, curve, cursor = [], [], 0
    model.train()
    for step in range(1, args.steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses = []
        for _ in range(args.grad_accum):
            idx = [order[(cursor + i) % len(order)] for i in range(args.batch_size)]
            cursor += args.batch_size
            seq_loss, _, _ = per_sequence_loss(model, [train[i] for i in idx], device, dtype)
            (seq_loss.sum() / per_step).backward()
            losses.extend(seq_loss.detach().tolist())
        for g in opt.param_groups:
            g["lr"] = lr_at(step, 5e-4, 20, args.steps)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        if step % 10 == 0:
            curve.append({"step": step, "loss": statistics.fmean(losses)})
            print(f"{step:6d} | {statistics.fmean(losses):.4f} | {step_times[-1]:.2f} s", flush=True)
    model.eval()
    after, _ = heldout(model, val, device, dtype)
    print(f"eval after: loss {after:.4f}", flush=True)

    summary = {
        "framework": f"pytorch {torch.__version__} + peft, CUDA {torch.version.cuda}",
        "model": args.model, "device": gpu_name(), "dtype": args.dtype,
        "steps": args.steps, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "trainable_params": trainable,
        "heldout_loss_before": round(before, 4), "heldout_loss_after": round(after, 4),
        "heldout_ppl_after": round(math.exp(after), 3), "heldout_tokens": held_tokens,
        "seconds_per_step_median": statistics.median(step_times),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "train_loss_curve": curve,
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"summary: {run / 'summary.json'}")
    print(f"held-out loss {before:.4f} -> {after:.4f}, {summary['seconds_per_step_median']:.3f} s/step, "
          f"peak GPU {summary['peak_gpu_memory_gb']} GB")


if __name__ == "__main__":
    main()

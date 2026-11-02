#!/usr/bin/env bash
# Measure this machine the same way every other Ferrotensor result was measured,
# and write one JSON report you can attach to an issue.
#
#   scripts/hardware-report.sh            # full kernel and Tensor API benchmarks
#   scripts/hardware-report.sh --quick    # smaller kernel sizes, one Tensor API case
#
# Writes reports/<host>-<backend>-<commit>.json plus the raw logs next to it.
# Respects WGPU_BACKEND (e.g. WGPU_BACKEND=vulkan) like the rest of Ferrotensor.
set -euo pipefail

quick=0
for arg in "$@"; do
  case "$arg" in
    --quick) quick=1 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
out_dir="$root/reports"
mkdir -p "$out_dir"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "== building (release)"
cargo build --release -q -p ferrotensor-cli -p ferrotensor-benchmarks

echo "== ferrotensor info --check"
# A failed check still produces a report: that is what we want to see.
set +e
./target/release/ferrotensor info --check --json > "$work/info.json"
check_status=$?
set -e

echo "== kernel benchmarks"
kernel_args=()
[ "$quick" = 1 ] && kernel_args+=(--quick)
./target/release/bench ${kernel_args[@]+"${kernel_args[@]}"} | tee "$work/kernels.md"

echo "== Tensor API benchmarks"
tensor_args=()
[ "$quick" = 1 ] && tensor_args+=(--filter "100 element-wise")
./target/release/tensor-api ${tensor_args[@]+"${tensor_args[@]}"} | tee "$work/tensor_api.md"

{
  rustc --version
  uname -a
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|driverInfo|apiVersion' | head -8
  true
} > "$work/system.txt" 2>&1

python3 - "$work" "$out_dir" "$quick" "$check_status" "$(git rev-parse --short HEAD)" <<'PY'
import json, re, socket, sys
from pathlib import Path

work, out_dir, quick, check_status, commit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4]), sys.argv[5]

def table(path):
    """Rows of the first Markdown table in a file, as dicts keyed by header."""
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(header, (c.strip() for c in l.strip("|").split("|")))) for l in lines[2:]]

def ms(text):
    m = re.match(r"([\d.]+) ms", text or "")
    return float(m.group(1)) if m else None

info = json.loads((work / "info.json").read_text())
gpu = next((a for a in info.get("adapters", []) if a.get("selected")), None)
kernels = [
    {"kernel": r["kernel"], "shape": r["shape"], "median_ms": ms(r["median time"]), "throughput": r["throughput"]}
    for r in table(work / "kernels.md")
]
tensor_api = [
    {"case": r["case"], "device": r["device"], "median_ms": ms(r["median"]), "checksum": r["checksum"]}
    for r in table(work / "tensor_api.md")
]
report = {
    "commit": commit,
    "host": socket.gethostname(),
    "mode": "quick" if quick else "full",
    "gpu": gpu,
    "kernel_check_passed": check_status == 0,
    "info": info,
    "system": (work / "system.txt").read_text().strip().splitlines(),
    "kernels": kernels,
    "tensor_api": tensor_api,
}
backend = (gpu or {}).get("backend", "none").lower()
name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{report['host']}-{backend}-{commit}")
path = out_dir / f"{name}.json"
path.write_text(json.dumps(report, indent=2) + "\n")
for raw in ("kernels.md", "tensor_api.md", "system.txt"):
    (out_dir / f"{name}.{raw}").write_text((work / raw).read_text())
print(f"\nreport: {path}")
print(f"GPU: {(gpu or {}).get('name', 'none')} ({backend}), kernel check {'passed' if check_status == 0 else 'FAILED'}, "
      f"{len(kernels)} kernel and {len(tensor_api)} Tensor API results")
PY

exit "$check_status"

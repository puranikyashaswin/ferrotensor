#!/usr/bin/env bash
# Compare Ferrotensor, mlx-lm and PyTorch + PEFT (MPS) on the same LoRA recipe
# on an Apple-silicon Mac, and print one Markdown table. See
# docs/benchmarks-mac.md for the recipe, how memory is measured and the caveats.
#
#   scripts/bench_mac.sh                       # all three stacks, 250 steps
#   scripts/bench_mac.sh --steps 20            # a short pass to check the setup
#   scripts/bench_mac.sh --only mlx            # one stack: ferrotensor | mlx | pytorch
#   scripts/bench_mac.sh --repeats 3           # three runs per stack, median and range
#   scripts/bench_mac.sh --out DIR             # default: data/bench-mac (gitignored)
#   scripts/bench_mac.sh --skip-install        # reuse the virtual environment as it is
#
# Any other option is passed to scripts/bench_mac.py (see its --help), for
# example --repeats, --base-dtype, --cooldown or --dry-run.
#
# The stacks run one after another, never in parallel. Close other apps first.
# The virtual environment lives outside the repository, in
# $FERROTENSOR_BENCH_VENV (default ~/.cache/ferrotensor-bench-mac/venv); the
# pinned packages are in scripts/bench_mac_requirements.txt. Needs Python 3.11
# or newer (override the interpreter with PYTHON=...) and, for the ferrotensor
# run, a Rust toolchain.
set -euo pipefail

usage() { sed -n '2,21p' "$0"; exit "${1:-0}"; }

root="$(cd "$(dirname "$0")/.." && pwd)"
only="ferrotensor,mlx,pytorch"
skip_install=0
dry_run=0
py_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --skip-install) skip_install=1; shift ;;
    --dry-run) dry_run=1; py_args+=("$1"); shift ;;
    --only) [ $# -ge 2 ] || { echo "--only needs a value" >&2; exit 2; }
            only="$2"; py_args+=("$1" "$2"); shift 2 ;;
    --steps|--out) [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }
                   py_args+=("$1" "$2"); shift 2 ;;
    *) py_args+=("$1"); shift ;;
  esac
done

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "error: this benchmark needs an Apple-silicon Mac (macOS, arm64); this is $(uname -s) $(uname -m)" >&2
  exit 1
fi

venv="${FERROTENSOR_BENCH_VENV:-$HOME/.cache/ferrotensor-bench-mac/venv}"
case "$venv" in
  "$root"|"$root"/*) echo "error: the virtual environment must be outside the repository: $venv" >&2; exit 1 ;;
esac

if [ "$dry_run" = 1 ]; then
  exec "${PYTHON:-python3}" "$root/scripts/bench_mac.py" ${py_args[@]+"${py_args[@]}"}
fi

if [ "$skip_install" = 0 ]; then
  py="${PYTHON:-python3}"
  command -v "$py" >/dev/null || { echo "error: $py not found; set PYTHON to a Python 3.11+ interpreter" >&2; exit 1; }
  "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' ||
    { echo "error: $py is older than Python 3.11 (torch 2.15 needs 3.11+); set PYTHON" >&2; exit 1; }
  if [ ! -x "$venv/bin/python" ]; then
    echo "== creating $venv"
    mkdir -p "$(dirname "$venv")"
    "$py" -m venv "$venv"
  fi
  echo "== installing the pinned packages into $venv"
  "$venv/bin/python" -m pip install -q --upgrade pip
  "$venv/bin/python" -m pip install -q -r "$root/scripts/bench_mac_requirements.txt"
elif [ ! -x "$venv/bin/python" ]; then
  echo "error: --skip-install, but there is no environment at $venv; run once without it" >&2
  exit 1
fi

case ",$only," in
  *,ferrotensor,*)
    echo "== building the ferrotensor CLI (release)"
    command -v cargo >/dev/null || { echo "error: cargo not found; install Rust (https://rustup.rs)" >&2; exit 1; }
    (cd "$root" && cargo build --release -q -p ferrotensor-cli)
    ;;
esac

exec "$venv/bin/python" "$root/scripts/bench_mac.py" ${py_args[@]+"${py_args[@]}"}

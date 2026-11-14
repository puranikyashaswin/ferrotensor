#!/usr/bin/env bash
# Run every Ferrotensor example and fail when one fails.
#
#   scripts/run-examples.sh          # every example (needs a GPU adapter)
#   scripts/run-examples.sh --cpu    # only the examples that need no GPU
#   scripts/run-examples.sh --gpu    # the examples that need a GPU, plus the
#                                    # ones that pick the GPU up when there is one
#
# Prints one PASS or FAIL line per example, then the output of the failed ones,
# and exits non-zero if any failed. CI runs --cpu on the ubuntu test job and
# --gpu on the software Vulkan job. The examples check their own results with
# assert!, so a wrong result fails here too.
#
# The Python example (ferrotensor-py/examples/mlp.py) runs in
# .github/workflows/python.yml, against the built wheel.
set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# ADDING AN EXAMPLE? Add it to EXAMPLES below. This script fails if an
# `examples/*.rs` file is neither listed there nor in SKIPPED, so a new example
# cannot be forgotten.
#
# One entry per run, fields separated by "|":
#   mode | label | crate | example | environment | arguments
#
#   mode         cpu   runs with --cpu (needs no GPU)
#                gpu   runs with --gpu (needs a GPU adapter)
#                both  runs with --cpu and with --gpu
#   label        name shown in the PASS/FAIL line
#   environment  space-separated NAME=value pairs, may be empty
#   arguments    space-separated; @WORK@ is replaced by a scratch directory
#
# Entries run in order, and gen_dummy must come before the examples that read
# the file it writes.
# ---------------------------------------------------------------------------
EXAMPLES=(
  # The Tensor API examples. They run on the CPU unless told otherwise;
  # quickstart and similarity_search use the GPU when there is one.
  "both|quickstart|ferrotensor|quickstart||"
  "cpu|mlp_two_moons|ferrotensor|mlp_two_moons||"
  "gpu|mlp_two_moons (gpu)|ferrotensor|mlp_two_moons|FERROTENSOR_DEVICE=gpu|"
  # 600 steps is the default; 150 is enough for the loss check and keeps CI short.
  "cpu|char_transformer|ferrotensor|char_transformer|FERROTENSOR_STEPS=150|"
  "gpu|char_transformer (gpu)|ferrotensor|char_transformer|FERROTENSOR_DEVICE=gpu FERROTENSOR_STEPS=100|"
  "both|similarity_search|ferrotensor|similarity_search||"
  "gpu|gpu_matmul_attention|ferrotensor|gpu_matmul_attention|FERROTENSOR_REQUIRE_GPU=1 FERROTENSOR_EXAMPLE_QUICK=1|"

  # The low-level examples. All of them open a GPU context.
  "gpu|init_wgpu|ferrotensor-backend|init_wgpu||"
  "gpu|tensor_creation|ferrotensor-backend|tensor_creation||"
  "gpu|basic_tensor|ferrotensor-backend|basic_tensor||"
  "gpu|fused_op|ferrotensor-backend|fused_op||"
  "gpu|ops_test|ferrotensor-backend|ops_test||"
  "gpu|flash_attention|ferrotensor-graph|flash_attention||"
  "gpu|llama_block|ferrotensor-graph|llama_block||"

  # gen_dummy writes a safetensors file that read_tensor and load_and_run read.
  "both|gen_dummy|ferrotensor-loader|gen_dummy||@WORK@/dummy.safetensors"
  "cpu|read_tensor|ferrotensor-loader|read_tensor||@WORK@/dummy.safetensors"
  "gpu|load_and_run|ferrotensor-loader|load_and_run||@WORK@/dummy.safetensors"
)

# Examples that are not run, with the reason. Format: "crate/example|reason".
SKIPPED=(
  "ferrotensor-monitor/tui_demo|interactive terminal UI"
)

# Crates whose examples are not checked here. The trainer examples run in the
# "Trainer examples" step of the GPU job in .github/workflows/ci.yml, and
# peft_fixture runs in its "PEFT export against PyTorch + PEFT" step.
IGNORED_CRATES=(ferrotensor-trainer)

want_cpu=0
want_gpu=0
for arg in "$@"; do
  case "$arg" in
    --cpu) want_cpu=1 ;;
    --gpu) want_gpu=1 ;;
    -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (use --cpu, --gpu or --help)" >&2; exit 2 ;;
  esac
done
if [ "$want_cpu" = 0 ] && [ "$want_gpu" = 0 ]; then
  want_cpu=1
  want_gpu=1
fi

# --- Every examples/*.rs file must be listed or skipped, and every entry must exist.
listed=" "
for entry in "${EXAMPLES[@]}"; do
  IFS='|' read -r _ _ crate example _ _ <<<"$entry"
  listed+="$crate/$example "
done
for entry in "${SKIPPED[@]}"; do
  listed+="${entry%%|*} "
done

missing=0
for file in */examples/*.rs; do
  crate="${file%%/*}"
  name="$(basename "$file" .rs)"
  ignored=0
  for c in "${IGNORED_CRATES[@]}"; do
    [ "$c" = "$crate" ] && ignored=1
  done
  [ "$ignored" = 1 ] && continue
  case "$listed" in
    *" $crate/$name "*) ;;
    *)
      echo "FAIL  $crate/$name is not listed in scripts/run-examples.sh (add it to EXAMPLES or SKIPPED)" >&2
      missing=1
      ;;
  esac
done
for item in $listed; do
  if [ ! -f "${item%%/*}/examples/${item##*/}.rs" ]; then
    echo "FAIL  $item is listed in scripts/run-examples.sh but the file does not exist" >&2
    missing=1
  fi
done
if [ "$missing" = 1 ]; then
  exit 1
fi

# --- Run the examples.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

passed=0
failed=0
failed_labels=()

for entry in "${EXAMPLES[@]}"; do
  IFS='|' read -r mode label crate example envs args <<<"$entry"
  case "$mode" in
    cpu) [ "$want_cpu" = 1 ] || continue ;;
    gpu) [ "$want_gpu" = 1 ] || continue ;;
    both) ;;
    *) echo "bad mode '$mode' for $label in scripts/run-examples.sh" >&2; exit 1 ;;
  esac

  env_args=()
  read -ra env_args <<<"$envs"
  cargo_args=()
  read -ra cargo_args <<<"${args//@WORK@/$work}"
  cmd=(cargo run --release -p "$crate" --example "$example")
  if [ "${#cargo_args[@]}" -gt 0 ]; then
    cmd+=(-- "${cargo_args[@]}")
  fi

  log="$work/${label//[^A-Za-z0-9_]/_}.log"
  started=$SECONDS
  if env ${env_args[@]+"${env_args[@]}"} "${cmd[@]}" >"$log" 2>&1; then
    printf 'PASS  %-28s %4ds\n' "$label" $((SECONDS - started))
    passed=$((passed + 1))
  else
    printf 'FAIL  %-28s %4ds\n' "$label" $((SECONDS - started))
    echo "----- last lines of the output of $label -----"
    tail -n 60 "$log"
    echo "----- end -----"
    failed=$((failed + 1))
    failed_labels+=("$label")
  fi
done

echo
echo "$passed passed, $failed failed"
if [ "$failed" -gt 0 ]; then
  printf 'failed: %s\n' "${failed_labels[@]}"
  exit 1
fi

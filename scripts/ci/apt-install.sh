#!/usr/bin/env bash
# Install apt packages on a hosted Ubuntu runner and survive mirror outages.
#
# Usage: scripts/ci/apt-install.sh PACKAGE...
#
# The runner images point apt at `mirror+file:/etc/apt/apt-mirrors.txt`, whose
# first entry is the Azure mirror. When that mirror is down, `apt-get update`
# or `apt-get install` fails (exit 100) and the job fails before any test runs.
# This script runs both commands up to ATTEMPTS times with growing waits. If
# the default mirror list keeps failing, it points the apt sources at the
# canonical archive.ubuntu.com / security.ubuntu.com hosts and tries again.
# It exits non-zero if every attempt fails.
#
# Environment overrides (used to test this script off a runner):
#   APT_DIR       apt configuration directory (default /etc/apt)
#   ATTEMPTS      attempts per mirror setting (default 3)
#   WAIT_SECONDS  base wait; attempt n waits n times this (default 15)
set -u

APT_DIR="${APT_DIR:-/etc/apt}"
ATTEMPTS="${ATTEMPTS:-3}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 PACKAGE..." >&2
  exit 2
fi

sudo_cmd=()
if [ "$(id -u)" -ne 0 ]; then
  sudo_cmd=(sudo)
fi

apt_opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30)

try_install() {
  "${sudo_cmd[@]}" apt-get update -q "${apt_opts[@]}" &&
    "${sudo_cmd[@]}" apt-get install -y -q "${apt_opts[@]}" "$@"
}

# Run try_install up to ATTEMPTS times, waiting 1x, 2x, ... WAIT_SECONDS between tries.
run_attempts() {
  local label="$1"
  shift
  local n
  for n in $(seq 1 "$ATTEMPTS"); do
    echo "apt-install: attempt ${n}/${ATTEMPTS} (${label})"
    if try_install "$@"; then
      echo "apt-install: installed on attempt ${n} (${label})"
      return 0
    fi
    if [ "$n" -lt "$ATTEMPTS" ]; then
      echo "apt-install: attempt ${n} failed, waiting $((n * WAIT_SECONDS))s"
      sleep $((n * WAIT_SECONDS))
    fi
  done
  return 1
}

# Rewrite the runner's mirror+file: and azure.archive.ubuntu.com sources to the
# canonical hosts (security.ubuntu.com for the *-security suites). Returns 1
# without touching anything when no source file has either.
use_canonical_mirrors() {
  # Bracket classes, not backslashes: awk -v would eat the backslashes.
  local pattern='mirror[+]file:/etc/apt/apt-mirrors[.]txt|https?://azure[.]archive[.]ubuntu[.]com/ubuntu/?'
  local files=() f changed=0 tmp
  for f in "$APT_DIR/sources.list" "$APT_DIR"/sources.list.d/*.sources "$APT_DIR"/sources.list.d/*.list; do
    if [ -f "$f" ] && grep -Eq "$pattern" "$f"; then
      files+=("$f")
    fi
  done
  if [ "${#files[@]}" -eq 0 ]; then
    echo "apt-install: no mirror+file: or azure sources under $APT_DIR, leaving apt sources unchanged"
    return 1
  fi

  tmp="$(mktemp)"
  for f in "${files[@]}"; do
    # Paragraph mode: a deb822 stanza is one paragraph; a one-line-style .list
    # file is one paragraph whose lines are each handled on their own.
    awk -v pat="$pattern" '
      BEGIN { RS = "" }
      {
        n = split($0, lines, "\n")
        stanza_sec = ($0 ~ /(^|\n)Suites:[^\n]*-security/)
        out = ""
        for (i = 1; i <= n; i++) {
          line = lines[i]
          if (line ~ /^URIs:/) {
            sub(pat, stanza_sec ? "http://security.ubuntu.com/ubuntu/" : "http://archive.ubuntu.com/ubuntu/", line)
          } else if (line ~ /^deb/) {
            sub(pat, (line ~ /-security/) ? "http://security.ubuntu.com/ubuntu/" : "http://archive.ubuntu.com/ubuntu/", line)
          }
          out = out (i > 1 ? "\n" : "") line
        }
        printf "%s%s", (NR > 1 ? "\n\n" : ""), out
      }
      END { print "" }' "$f" >"$tmp"
    if ! cmp -s "$f" "$tmp"; then
      echo "apt-install: rewriting $f"
      diff -u "$f" "$tmp" | sed 's/^/apt-install:   /'
      "${sudo_cmd[@]}" cp "$tmp" "$f"
      changed=1
    fi
  done
  rm -f "$tmp"
  if [ "$changed" -eq 0 ]; then
    echo "apt-install: apt sources already canonical, nothing to rewrite"
    return 1
  fi
  return 0
}

if run_attempts "default mirrors" "$@"; then
  exit 0
fi

echo "apt-install: default mirrors failed ${ATTEMPTS} times, falling back to archive.ubuntu.com / security.ubuntu.com"
if use_canonical_mirrors && run_attempts "canonical mirrors" "$@"; then
  exit 0
fi

echo "apt-install: giving up, could not install: $*" >&2
exit 1

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
# The fallback only helps when one host or the mirror list is down. It cannot
# help when the runner resolves no Ubuntu host at all; callers that must survive
# that cache the .deb files (see install-lavapipe.sh).
#
# The source rewrite is a CI-only action: it needs CI=true (GitHub Actions sets
# it), and the original files are put back when the script exits.
#
# Environment:
#   APT_ARCHIVE_DIR  download the .deb files into this directory and keep them
#                    (apt option Dir::Cache::archives); default is apt's own
#   APT_DIR          apt configuration directory (default /etc/apt)
#   ATTEMPTS         attempts per mirror setting (default 3)
#   WAIT_SECONDS     base wait; attempt n waits n times this (default 15)
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
if [ -n "${APT_ARCHIVE_DIR:-}" ]; then
  apt_opts+=(-o "Dir::Cache::archives=$APT_ARCHIVE_DIR" -o APT::Keep-Downloaded-Packages=true)
fi

# Source files rewritten by use_canonical_mirrors are saved first and put back
# on exit. backup_list holds "original<TAB>backup" entries.
backup_dir=""
backup_list=()

# shellcheck disable=SC2329  # run by the EXIT trap below
restore_sources() {
  local entry orig bak
  for entry in ${backup_list[@]+"${backup_list[@]}"}; do
    orig="${entry%%$'\t'*}"
    bak="${entry#*$'\t'}"
    if ${sudo_cmd[@]+"${sudo_cmd[@]}"} cp -p "$bak" "$orig"; then
      echo "apt-install: restored $orig"
    else
      echo "apt-install: could not restore $orig; the original is in $bak" >&2
      return 0
    fi
  done
  if [ -n "$backup_dir" ]; then
    rm -rf "$backup_dir"
  fi
}
trap restore_sources EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

try_install() {
  ${sudo_cmd[@]+"${sudo_cmd[@]}"} apt-get update -q "${apt_opts[@]}" &&
    ${sudo_cmd[@]+"${sudo_cmd[@]}"} apt-get install -y -q "${apt_opts[@]}" "$@"
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
# without touching anything when CI is not "true", when no source file has
# either, or when no file could be backed up and written. Every file it
# rewrites is saved first and put back by restore_sources on exit.
use_canonical_mirrors() {
  if [ "${CI:-}" != "true" ]; then
    echo "apt-install: CI is not 'true', refusing to rewrite apt sources"
    return 1
  fi
  # Bracket classes, not backslashes: awk -v would eat the backslashes.
  local pattern='mirror[+]file:/etc/apt/apt-mirrors[.]txt|https?://azure[.]archive[.]ubuntu[.]com/ubuntu/?'
  local files=() f changed=0 tmp bak
  for f in "$APT_DIR/sources.list" "$APT_DIR"/sources.list.d/*.sources "$APT_DIR"/sources.list.d/*.list; do
    if [ -f "$f" ] && grep -Eq "$pattern" "$f"; then
      files+=("$f")
    fi
  done
  if [ "${#files[@]}" -eq 0 ]; then
    echo "apt-install: no mirror+file: or azure sources under $APT_DIR, leaving apt sources unchanged"
    return 1
  fi

  if ! tmp="$(mktemp)"; then
    echo "apt-install: mktemp failed, leaving apt sources unchanged" >&2
    return 1
  fi
  if [ -z "$backup_dir" ] && ! backup_dir="$(mktemp -d)"; then
    echo "apt-install: mktemp -d failed, leaving apt sources unchanged" >&2
    backup_dir=""
    rm -f "$tmp"
    return 1
  fi

  for f in "${files[@]}"; do
    # Line by line, so blank lines and everything else stay as they were. A
    # deb822 stanza is the run of lines between blank lines; its Suites: line
    # decides whether URIs: points at security.ubuntu.com. A one-line-style
    # .list file is one such run whose deb lines are each handled on their own.
    if ! awk -v pat="$pattern" \
      -v main_url="http://archive.ubuntu.com/ubuntu/" \
      -v sec_url="http://security.ubuntu.com/ubuntu/" '
      function flush(   i, sec, line) {
        sec = 0
        for (i = 1; i <= n; i++) if (buf[i] ~ /^Suites:.*-security/) sec = 1
        for (i = 1; i <= n; i++) {
          line = buf[i]
          if (line ~ /^URIs:/) {
            sub(pat, sec ? sec_url : main_url, line)
          } else if (line ~ /^deb/) {
            sub(pat, (line ~ /-security/) ? sec_url : main_url, line)
          }
          print line
        }
        n = 0
      }
      /^[[:space:]]*$/ { flush(); print; next }
      { buf[++n] = $0 }
      END { flush() }' "$f" >"$tmp"; then
      echo "apt-install: awk failed on $f, leaving it unchanged" >&2
      continue
    fi
    if cmp -s "$f" "$tmp"; then
      continue
    fi
    echo "apt-install: rewriting $f"
    diff -u "$f" "$tmp" | sed 's/^/apt-install:   /'
    bak="$backup_dir/${#backup_list[@]}"
    if ! ${sudo_cmd[@]+"${sudo_cmd[@]}"} cp -p "$f" "$bak"; then
      echo "apt-install: could not back up $f, leaving it unchanged" >&2
      continue
    fi
    backup_list+=("$f"$'\t'"$bak")
    if ${sudo_cmd[@]+"${sudo_cmd[@]}"} cp "$tmp" "$f"; then
      changed=1
    else
      echo "apt-install: could not write $f" >&2
    fi
  done
  rm -f "$tmp"
  if [ "$changed" -eq 0 ]; then
    echo "apt-install: no apt source was rewritten"
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

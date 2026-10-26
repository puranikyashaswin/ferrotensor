#!/usr/bin/env bash
# Build the website into one folder: the landing page at the root and the
# mdBook guide under docs/, fill in the release (release.py), then finish it
# for search engines (seo.py).
# Usage: site/build.sh [output-dir] (default site/dist).
set -euo pipefail

site="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$site/dist}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

case "$out" in /|"$HOME") echo "refusing to clear $out" >&2; exit 1;; esac
rm -rf "${out:?}"/*
cp -R "$site/landing/." "$out/"
rm -f "$out/og.svg"
mdbook build "$site/book" --dest-dir "$out/docs"
python3 "$site/release.py" "$out"
python3 "$site/seo.py" "$out"
echo "site built in $out"

#!/usr/bin/env bash
# Install Mesa lavapipe (software Vulkan) on a hosted Ubuntu runner, downloading
# nothing when it can be avoided.
#
# Usage: scripts/ci/install-lavapipe.sh CACHE_DIR
#
# CACHE_DIR is a directory the workflow restores and saves with actions/cache.
#   a. A lavapipe ICD is already installed: do nothing.
#   b. CACHE_DIR holds .deb files: install them with dpkg, offline. If that
#      fails (for example a missing dependency) or leaves no ICD, go to c.
#   c. Install with scripts/ci/apt-install.sh (retries, mirror fallback) and
#      copy every .deb apt downloaded, the packages and their dependencies,
#      into CACHE_DIR so the workflow can save it.
#
# Environment overrides (used to test this script off a runner):
#   ICD_DIR      where the Vulkan ICD manifests live (default /usr/share/vulkan/icd.d)
#   APT_INSTALL  command used for step c (default apt-install.sh next to this file)
set -u

if [ "$#" -ne 1 ]; then
  echo "usage: $0 CACHE_DIR" >&2
  exit 2
fi
CACHE_DIR="$1"
ICD_DIR="${ICD_DIR:-/usr/share/vulkan/icd.d}"
APT_INSTALL="${APT_INSTALL:-$(dirname "$0")/apt-install.sh}"
PACKAGES=(mesa-vulkan-drivers vulkan-tools)

sudo_cmd=()
if [ "$(id -u)" -ne 0 ]; then
  sudo_cmd=(sudo)
fi

icd_present() {
  compgen -G "$ICD_DIR/lvp_icd*.json" >/dev/null
}

if icd_present; then
  echo "install-lavapipe: path a, lavapipe ICD already present, installing nothing:"
  ls "$ICD_DIR"/lvp_icd*.json
  exit 0
fi

shopt -s nullglob
debs=("$CACHE_DIR"/*.deb)
if [ "${#debs[@]}" -gt 0 ]; then
  echo "install-lavapipe: path b, installing ${#debs[@]} cached .deb files offline with dpkg"
  if ${sudo_cmd[@]+"${sudo_cmd[@]}"} dpkg -i "${debs[@]}" && icd_present; then
    echo "install-lavapipe: path b done, lavapipe ICD installed from the cache"
    exit 0
  fi
  echo "install-lavapipe: offline dpkg install failed or left no ICD, falling through to apt" >&2
fi

echo "install-lavapipe: path c, installing ${PACKAGES[*]} with apt"
stage="$(mktemp -d)" || exit 1
# apt downloads as the _apt user; it must be able to enter the directory.
chmod 755 "$stage"
if ! APT_ARCHIVE_DIR="$stage" "$APT_INSTALL" "${PACKAGES[@]}"; then
  ${sudo_cmd[@]+"${sudo_cmd[@]}"} rm -rf "$stage"
  exit 1
fi

# Keep exactly what this install downloaded: a stale or partial set would make
# the next offline install fail.
mkdir -p "$CACHE_DIR" && rm -f "$CACHE_DIR"/*.deb
staged=("$stage"/*.deb)
if [ "${#staged[@]}" -eq 0 ]; then
  echo "install-lavapipe: apt downloaded no .deb files, nothing to cache"
elif cp "${staged[@]}" "$CACHE_DIR"/; then
  echo "install-lavapipe: cached ${#staged[@]} .deb files in $CACHE_DIR:"
  ls -l "$CACHE_DIR"
else
  echo "install-lavapipe: could not copy the .deb files, not caching them" >&2
  rm -f "$CACHE_DIR"/*.deb
fi
${sudo_cmd[@]+"${sudo_cmd[@]}"} rm -rf "$stage"
exit 0

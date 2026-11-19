#!/bin/sh
# Fail when a tracked file sits at the repository root without being listed in
# tools/root_allowlist.txt. Keeps scratch notes, CSVs, images and logs out of
# the root (put them under docs/, tests/fixtures/ or benchmark_results/).
#
# usage: tools/check_root_files.sh [FILE_LIST]
#   FILE_LIST: newline-separated tracked paths (default: `git ls-files`); used by the test.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
allow="$here/root_allowlist.txt"
if [ "$#" -ge 1 ]; then list="$(cat "$1")"; else list="$(git -C "$here/.." ls-files)"; fi
bad=0
for f in $(printf '%s\n' "$list" | grep -v '/' || true); do
    if ! grep -qxF -- "$f" "$allow"; then
        echo "unexpected file at the repository root: $f" >&2
        bad=1
    fi
done
if [ "$bad" -ne 0 ]; then
    echo "Move it under docs/, tests/fixtures/ or benchmark_results/, or add it to tools/root_allowlist.txt if it truly belongs at the root." >&2
    exit 1
fi
echo "root files ok"

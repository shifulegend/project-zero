#!/bin/sh
# Static analysis ratchet: run clang-tidy (bugprone, cert, clang-analyzer subset) over src/*.c and
# fail when the number of warnings exceeds tools/clang_tidy_baseline.txt. Lower the baseline
# as findings are fixed (--update rewrites it); it must never go up.
#
# usage: tools/check_clang_tidy.sh [--update]      env: CLANG_TIDY=clang-tidy-18
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
tidy="${CLANG_TIDY:-clang-tidy-18}"
baseline_file="$root/tools/clang_tidy_baseline.txt"
checks='-*,bugprone-*,clang-analyzer-*,cert-*,readability-identifier-naming,-bugprone-easily-swappable-parameters,-bugprone-reserved-identifier,-cert-dcl37-c,-cert-dcl51-cpp,-clang-analyzer-optin.performance.Padding,-bugprone-narrowing-conversions,-cert-err33-c,-bugprone-implicit-widening-of-multiplication-result'
out="$(mktemp)"; trap 'rm -f "$out"' EXIT
cd "$root"
find src -name '*.c' | sort | xargs -P "$(nproc 2>/dev/null || echo 2)" -I{} \
    "$tidy" --quiet -checks="$checks" --header-filter='^$' {} -- \
    -std=c99 -march=native -Iinclude -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -DPZ_VERSION=v \
    > "$out" 2>/dev/null || true
count="$(grep -c 'warning:' "$out" || true)"
if [ "${1:-}" = "--update" ]; then
    echo "$count" > "$baseline_file"; echo "baseline written: $count"; exit 0
fi
base="$(cat "$baseline_file")"
echo "clang-tidy warnings: $count (baseline $base)"
if [ "$count" -gt "$base" ]; then
    echo "NEW clang-tidy findings (count went up). Fix them; first lines:" >&2
    grep 'warning:' "$out" | sed "s|$root/||" | sort | head -40 >&2
    exit 1
fi
[ "$count" -lt "$base" ] && echo "note: fewer findings than the baseline; run tools/check_clang_tidy.sh --update to lock it in"
exit 0

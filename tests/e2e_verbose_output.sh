#!/usr/bin/env bash
# e2e_verbose_output.sh — --verbose prints each startup line exactly once (the attention-kernel
# line was once printed twice after a refactor copied its block).
#
# Usage: tests/e2e_verbose_output.sh <model.gguf>
# Requires a real downloaded model; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_verbose_output.sh <model.gguf>}"
BIN="${PZ:-./pz}"
FAIL=0
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

"$BIN" --model "$MODEL" --prompt "Hi" --max-tokens 2 --temperature 0 --threads 2 --verbose \
    >"$OUT" 2>&1

once() { # once <label> <pattern>: the pattern matches exactly one line
    local n
    n=$(grep -c -- "$2" "$OUT")
    if [ "$n" -eq 1 ]; then
        echo "  [PASS] $1 printed once"
    else
        echo "  [FAIL] $1 printed $n times"
        FAIL=$((FAIL + 1))
    fi
}

once "attention line" "Attention: "
once "KV cache line" "KV cache: "

[ "$FAIL" -eq 0 ] && echo "e2e_verbose_output: all passed" && exit 0
echo "e2e_verbose_output: $FAIL failed"
exit 1

#!/usr/bin/env bash
# e2e_cli_exit_status.sh — a run that prints an error must exit non-zero, and a run that
# succeeds must exit 0. Covers the modes that load a model and then fail on their own
# arguments (and a truncated model file): --plan-layers, --rpc-worker, --remote-layers, --perplexity, --server.
#
# Usage: tests/e2e_cli_exit_status.sh <model.gguf>
# Requires a real downloaded model; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_cli_exit_status.sh <model.gguf>}"
BIN="./pz"
FAIL=0
COMMON=(--model "$MODEL" --no-calibrate --threads 2)

expect() { # expect <wanted status> <label> -- <args...>
    local want=$1 label=$2
    shift 3
    "$BIN" "${COMMON[@]}" "$@" >/dev/null 2>&1 </dev/null
    local got=$?
    if [ "$got" -eq "$want" ]; then
        echo "  [PASS] $label (exit $got)"
    else
        echo "  [FAIL] $label: wanted exit $want, got $got"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== CLI exit status ==="
expect 0 "--plan-layers 4,4 prints a plan" -- --plan-layers 4,4
expect 1 "--plan-layers with no node sizes" -- --plan-layers ,
expect 1 "--rpc-worker without --layers" -- --rpc-worker 19999
expect 1 "--remote-layers with a malformed node" -- --remote-layers garbage --prompt hi
expect 1 "--perplexity on a missing file" -- --perplexity /nonexistent.txt

# A taken port must fail the server start, not exit 0.
PORT="${E2E_PORT:-8097}"
python3 - "$PORT" <<'PY' &
import socket, sys, time
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen(1)
time.sleep(60)
PY
HOLDER=$!
trap 'kill $HOLDER 2>/dev/null' EXIT
sleep 1
expect 1 "--server on a port that is already taken" -- --server --port "$PORT"

# A truncated download must be refused with an error, not crash the loader (exit 139 before).
TRUNC="$(mktemp)"
head -c 5000000 "$MODEL" >"$TRUNC"
"$BIN" --model "$TRUNC" --no-calibrate --threads 2 --prompt hi >/dev/null 2>&1 </dev/null
got=$?
rm -f "$TRUNC"
if [ "$got" -eq 1 ]; then
    echo "  [PASS] a truncated model file is refused (exit $got)"
else
    echo "  [FAIL] a truncated model file: wanted exit 1, got $got"
    FAIL=$((FAIL + 1))
fi

echo "=== $((7 - FAIL)) passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]

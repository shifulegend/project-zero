#!/usr/bin/env bash
# e2e_log_format.sh — with --log-format json every line on stderr is one JSON object
# {ts, level, component, msg}, stdout carries only program output, the text format is unchanged,
# --log-level filters, and a server started with an API key never writes the key to its log.
#
# Usage: tests/e2e_log_format.sh <model.gguf>
# Requires a real downloaded model; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_log_format.sh <model.gguf>}"
BIN="./pz"
PORT="${E2E_PORT:-8095}"
FAIL=0
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"; kill $SRV 2>/dev/null' EXIT
SRV=0

check() { # check <label> <command...>: passes when the command succeeds
    local label=$1
    shift
    if "$@"; then
        echo "  [PASS] $label"
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL + 1))
    fi
}

negate() { ! "$@"; }
empty() { [ ! -s "$1" ]; }

all_json() { # all_json <file>: every non-empty line parses as an object with the four keys
    python3 - "$1" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line)
    assert set(d) == {"ts", "level", "component", "msg"}, d
    assert d["level"] in ("debug", "info", "warn", "error"), d
    n += 1
sys.exit(0 if n else 1)
PY
}

COMMON=(--model "$MODEL" --no-calibrate --threads 2 --prompt hi --max-tokens 4 --temperature 0)

echo "=== structured logging ==="
"$BIN" "${COMMON[@]}" --log-format json >"$TMP/out" 2>"$TMP/err" </dev/null
check "a normal run writes only JSON lines to stderr" all_json "$TMP/err"
check "components are named (gguf_tok)" grep -q '"component":"gguf_tok"' "$TMP/err"
check "status lines moved off stdout in JSON mode" negate grep -q '^\[Calibration\]\|^Active:' "$TMP/out"

"$BIN" "${COMMON[@]}" >"$TMP/out_t" 2>"$TMP/err_t" </dev/null
text_plain() { grep -q '^\[1/4\] Opening model file' "$TMP/err_t" && grep -q '^\[Calibration\] Skipped' "$TMP/out_t"; }
check "text mode keeps its plain lines (progress on stderr, status on stdout)" text_plain

"$BIN" --model /nonexistent.gguf --prompt hi --log-format json 2>"$TMP/err_e" >/dev/null </dev/null
error_line() { all_json "$TMP/err_e" && grep -q '"level":"error"' "$TMP/err_e"; }
check "an error is one JSON object with level error" error_line

"$BIN" "${COMMON[@]}" --log-format json --log-level error >"$TMP/o" 2>"$TMP/err_l" </dev/null
check "--log-level error hides info lines" empty "$TMP/err_l"

"$BIN" --log-format xml >/dev/null 2>"$TMP/err_x" </dev/null
xml_status=$?
refused() { [ "$xml_status" -eq 1 ] && grep -q "must be one of: text, json" "$TMP/err_x"; }
check "an unknown --log-format is refused with exit 1" refused

# The server must not put the API key in its log, however verbose.
SECRET="sk-e2e-secret-4f9a1c7d"
"$BIN" --model "$MODEL" --server --port "$PORT" --no-calibrate --threads 2 --api-key "$SECRET" \
    --verbose --log-format json >/dev/null 2>"$TMP/err_s" &
SRV=$!
for _ in $(seq 1 120); do curl -s "localhost:$PORT/health" >/dev/null && break; sleep 1; done
curl -s -H "Authorization: Bearer $SECRET" "localhost:$PORT/v1/models" >/dev/null
curl -s -H "Authorization: Bearer wrong-$SECRET" "localhost:$PORT/v1/models" >/dev/null
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
check "the server's verbose log is JSON lines too" all_json "$TMP/err_s"
check "the API key never appears in the log" negate grep -q "$SECRET" "$TMP/err_s"

echo "=== $((9 - FAIL)) passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]

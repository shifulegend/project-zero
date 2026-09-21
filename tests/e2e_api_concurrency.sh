#!/usr/bin/env bash
# e2e_api_concurrency.sh — TS-2.6: verify the API server's concurrency
# contract. src/api/http_server.c serializes generation via
# ctx->generation_mutex, acquired with pthread_mutex_trylock() -- a second
# concurrent /v1/chat/completions request must get 429, never block or
# queue. Also verifies no grammar (FSMState) contamination across requests:
# FSMState is a stack-local per generate_with_callback() call
# (src/transformer/generate.c), so back-to-back JSON-mode requests with
# different structures must each independently produce valid JSON.
#
# Usage: tests/e2e_api_concurrency.sh <model.gguf>
# Requires a real downloaded model; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_api_concurrency.sh <model.gguf>}"
BIN="./adaptive_ai_engine"
PORT="${E2E_PORT:-8098}"

n_pass=0
n_fail=0
pass() { echo "  [PASS] $1"; n_pass=$((n_pass + 1)); }
fail() { echo "  [FAIL] $1"; n_fail=$((n_fail + 1)); }

is_valid_json() {
    python3 -c "import sys,json; json.loads(sys.stdin.read())" 2>/dev/null
}

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN not found or not executable (run 'make release' first)" >&2
    exit 1
fi

SERVER_LOG=$(mktemp)
$BIN --model "$MODEL" --port "$PORT" --server --web-ui off > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -TERM $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

ready=0
for _ in $(seq 1 120); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null; then ready=1; break; fi
    sleep 0.5
done
if [ "$ready" != "1" ]; then
    fail "server never became ready on port $PORT (see $SERVER_LOG)"
    echo "=== $n_pass passed, $n_fail failed ==="
    exit 1
fi
pass "server ready on port $PORT"

# --- Concurrency: fire a slow generation in the background, then a second
# request almost immediately -- the second must get 429, not block. ------
REQ_A_OUT=$(mktemp)
(curl -s -o "$REQ_A_OUT" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Count slowly from one to ten, one number per sentence, with a short explanation for each."}],"max_tokens":150,"temperature":0,"stream":false,"response_format":{"type":"json_object"}}' \
    > "${REQ_A_OUT}.code") &
REQ_A_PID=$!

sleep 0.5   # let request A actually acquire generation_mutex and start generating

code_b=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16,"temperature":0,"stream":false}')

if [ "$code_b" = "429" ]; then
    pass "second concurrent request got 429 while the first was in flight"
else
    fail "second concurrent request got HTTP $code_b, expected 429 (serialization not holding)"
fi

wait "$REQ_A_PID"
code_a=$(cat "${REQ_A_OUT}.code" 2>/dev/null)
if [ "$code_a" = "200" ]; then
    pass "the first (in-flight) request itself completed successfully (HTTP 200)"
else
    fail "the first request did not complete with 200 (got: $code_a)"
fi
rm -f "$REQ_A_OUT" "${REQ_A_OUT}.code"

# --- No state leak: sequential JSON-mode requests with different shapes,
# each independently valid, plus a non-JSON request in between. ----------
content1=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Return a JSON array of the numbers 1, 2, 3."}],"max_tokens":60,"temperature":0,"stream":false,"response_format":{"type":"json_object"}}' \
    | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['choices'][0]['message']['content'])" 2>/dev/null)
if echo "$content1" | is_valid_json; then
    pass "sequential JSON request #1 (array shape) is independently valid JSON"
else
    fail "sequential JSON request #1 invalid: $content1"
fi

content_mid=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16,"temperature":0,"stream":false}' \
    | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['choices'][0]['message']['content'])" 2>/dev/null)
if echo "$content_mid" | grep -qF "Paris"; then
    pass "non-JSON request sandwiched between two JSON-mode requests is unaffected"
else
    fail "sandwiched non-JSON request regressed: $content_mid"
fi

content2=$(curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Return a JSON object with keys a and b, both booleans."}],"max_tokens":60,"temperature":0,"stream":false,"response_format":{"type":"json_object"}}' \
    | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d['choices'][0]['message']['content'])" 2>/dev/null)
if echo "$content2" | is_valid_json; then
    pass "sequential JSON request #2 (different object shape) is independently valid JSON, no leak from #1"
else
    fail "sequential JSON request #2 invalid (possible grammar-state leak): $content2"
fi

kill -TERM "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
trap - EXIT
rm -f "$SERVER_LOG"

echo ""
echo "=== $n_pass passed, $n_fail failed ==="
[ "$n_fail" -eq 0 ]

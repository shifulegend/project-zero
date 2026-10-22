#!/usr/bin/env bash
# e2e_parallel.sh — Phase 23 stage 2b: the --parallel N server over real
# sockets. Checks that N concurrent chat requests all run (no 429), give
# byte-for-byte the text a sequential server gives, that a whole-engine
# request (best_of) waits its turn among them, that the cancel endpoint
# stops only the request it names, that /v1/config and /metrics publish the
# slots, and that a model the batched forward cannot run falls back to
# sequential with a reason.
#
# Usage: tests/e2e_parallel.sh <dense-model.gguf> [recurrent-model.gguf]
# Requires real downloaded models; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_parallel.sh <dense-model.gguf> [recurrent-model.gguf]}"
FALLBACK_MODEL="${2:-}"
BIN="./pz"
PORT="${E2E_PORT:-8099}"
N=4

n_pass=0
n_fail=0
pass() { echo "  [PASS] $1"; n_pass=$((n_pass + 1)); }
fail() { echo "  [FAIL] $1"; n_fail=$((n_fail + 1)); }

[ -x "$BIN" ] || { echo "FAIL: $BIN not found (run 'make release' first)" >&2; exit 1; }
TMP=$(mktemp -d)
SERVER_PID=
stop_server() {
    [ -n "$SERVER_PID" ] && { kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
    SERVER_PID=
}
trap 'stop_server; rm -rf "$TMP"' EXIT

start_server() {   # model, extra args...
    local model="$1"; shift
    $BIN --model "$model" --port "$PORT" --server --web-ui off --metrics "$@" > "$TMP/server.log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 120); do
        curl -s -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null && return 0
        sleep 0.5
    done
    fail "server never became ready (see $TMP/server.log)"
    return 1
}

body() {   # i [extra json]
    printf '{"messages":[{"role":"user","content":"Count from 1 to %d in words, then name a colour."}],"max_tokens":40,"temperature":0%s}' \
        "$(( $1 + 3 ))" "${2:-}"
}

content() {   # file -> message content
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"], end="")' "$1" 2>/dev/null
}

# --- Reference: the same requests on a sequential server. ---------------
start_server "$MODEL" || exit 1
for i in $(seq 1 $N); do
    curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
        -d "$(body "$i")" > "$TMP/seq$i.json"
done
stop_server

# --- Parallel server. -----------------------------------------------------
start_server "$MODEL" --parallel $N --queue-size 8 || exit 1
slots=$(curl -s "http://127.0.0.1:$PORT/v1/config" | python3 -c 'import json,sys; print(json.load(sys.stdin)["features"]["parallel"]["slots"])')
[ "$slots" = "$N" ] && pass "/v1/config features.parallel.slots = $N" || fail "/v1/config parallel slots = $slots, want $N"
curl -s "http://127.0.0.1:$PORT/metrics" | grep -q "^project_zero_generation_slots $N$" \
    && pass "/metrics project_zero_generation_slots $N" || fail "/metrics does not report $N slots"

for i in $(seq 1 $N); do
    curl -s -o "$TMP/par$i.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$(body "$i")" > "$TMP/par$i.code" &
    PIDS="${PIDS:-} $!"
done
wait $PIDS   # not a bare wait: that would also wait for the server
ok=1; same=1
for i in $(seq 1 $N); do
    [ "$(cat "$TMP/par$i.code")" = "200" ] || ok=0
    [ "$(content "$TMP/par$i.json")" = "$(content "$TMP/seq$i.json")" ] || {
        same=0
        echo "    request $i: sequential=$(content "$TMP/seq$i.json" | head -c 80)"
        echo "    request $i: parallel  =$(content "$TMP/par$i.json" | head -c 80)"
    }
done
[ $ok = 1 ] && pass "$N concurrent requests all answered 200 (none rejected)" || fail "a concurrent request was not 200"
[ $same = 1 ] && pass "parallel text is identical to the sequential server's" || fail "parallel text differs from sequential"

# Whole-engine request (best_of) among batched ones: it waits, nobody fails.
for i in 1 2; do
    curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" -d "$(body "$i")" > "$TMP/mix$i.code" &
    MIX="${MIX:-} $!"
done
curl -s -o "$TMP/bo.json" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$(body 3 ',"best_of":2')" > "$TMP/bo.code" &
wait $MIX $!
if [ "$(cat "$TMP/mix1.code")$(cat "$TMP/mix2.code")$(cat "$TMP/bo.code")" = "200200200" ] &&
   python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["best_of"]["n"] == 2' "$TMP/bo.json" 2>/dev/null; then
    pass "best_of request runs between batched ones (all 200)"
else
    fail "mixed best_of + batched requests: codes $(cat "$TMP/mix1.code") $(cat "$TMP/mix2.code") $(cat "$TMP/bo.code")"
fi

# Cancel one streaming request by id; the other finishes normally.
curl -s -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Tell a long story about a lighthouse."}],"max_tokens":400,"temperature":0,"stream":true}' \
    > "$TMP/c1.sse" &
P1=$!
curl -s -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Tell a long story about a forest."}],"max_tokens":60,"temperature":0}' \
    > "$TMP/c2.json" &
P2=$!
id=
for _ in $(seq 1 100); do
    id=$(grep -o '"id":"chatcmpl-[^"]*"' "$TMP/c1.sse" 2>/dev/null | head -1 | cut -d'"' -f4)
    [ -n "$id" ] && break
    sleep 0.1
done
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions/cancel" \
    -H "Content-Type: application/json" -d "{\"id\":\"$id\"}")
wait $P1 $P2
if [ "$code" = "200" ] && grep -q '"finish_reason":"cancelled"' "$TMP/c1.sse" &&
   python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["choices"][0]["finish_reason"] in ("length","stop")' "$TMP/c2.json" 2>/dev/null; then
    pass "cancel stops only the named request"
else
    fail "cancel: endpoint $code, stream $(grep -o '"finish_reason":"[a-z]*"' "$TMP/c1.sse" | tail -1)"
fi
stop_server

# --- Fallback: a model the batched forward cannot run stays sequential. ---
if [ -n "$FALLBACK_MODEL" ]; then
    start_server "$FALLBACK_MODEL" --parallel $N || exit 1
    out=$(curl -s "http://127.0.0.1:$PORT/v1/config" | python3 -c 'import json,sys; p=json.load(sys.stdin)["features"]["parallel"]; print(p["slots"], p["requested"], bool(p["note"]))')
    [ "$out" = "1 $N True" ] && pass "unsupported model: 1 slot, requested $N, reason published" \
                             || fail "unsupported model fallback: got '$out'"
    stop_server
fi

echo
echo "=== $n_pass passed, $n_fail failed ==="
[ $n_fail -eq 0 ]

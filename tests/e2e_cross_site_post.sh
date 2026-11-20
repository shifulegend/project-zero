#!/usr/bin/env bash
# e2e_cross_site_post.sh — every state-changing POST route refuses a request that carries a foreign
# Origin (what a browser sends from another site) even when the server has no API key, and still
# serves one without an Origin (curl, SDKs).
#
# Usage: tests/e2e_cross_site_post.sh <model.gguf>
# Requires a real downloaded model; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_cross_site_post.sh <model.gguf>}"
BIN="./pz"
PORT="${E2E_PORT:-8096}"
FAIL=0

"$BIN" --model "$MODEL" --server --port "$PORT" --no-calibrate --threads 2 >/tmp/pz_e2e_xsite.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -s "localhost:$PORT/health" >/dev/null && break; sleep 1; done

status() { # status <path> [extra curl args...]
    local path=$1
    shift
    curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' \
        "$@" "localhost:$PORT$path"
}

echo "=== cross-site POST guard ==="
for route in /v1/prompt_cache/clear /v1/update/check /v1/chat/completions/cancel \
    /v1/chat/completions /v1/watermark/detect /v1/selfcheck; do
    got=$(status "$route" -H 'Origin: http://evil.example')
    if [ "$got" = 403 ]; then
        echo "  [PASS] $route with a foreign Origin: 403"
    else
        echo "  [FAIL] $route with a foreign Origin: wanted 403, got $got"
        FAIL=$((FAIL + 1))
    fi
done
got=$(status /v1/prompt_cache/clear)
if [ "$got" = 200 ]; then
    echo "  [PASS] /v1/prompt_cache/clear without an Origin: 200"
else
    echo "  [FAIL] /v1/prompt_cache/clear without an Origin: wanted 200, got $got"
    FAIL=$((FAIL + 1))
fi

echo "=== $((7 - FAIL)) passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]

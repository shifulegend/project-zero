#!/usr/bin/env bash
# e2e_prompt_cache.sh — Phase 21.5: KV prefix reuse must never change output.
# Runs the same 3-turn greedy conversation (each turn re-sends the full
# history, including the model's own earlier replies) with cache_prompt
# on and off, and requires byte-identical replies plus a non-zero
# usage.prompt_tokens_details.cached_tokens on the cached turns.
#
# Usage: tests/e2e_prompt_cache.sh <model.gguf>
# Requires a real downloaded model; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_prompt_cache.sh <model.gguf>}"
BIN="./adaptive_ai_engine"
PORT="${E2E_PORT:-8099}"

"$BIN" --model "$MODEL" --server --port "$PORT" --threads 4 >/tmp/pz_e2e_cache.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -s "localhost:$PORT/health" >/dev/null && break; sleep 1; done

python3 - "$PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
SYSTEM = "You are a careful assistant. " + " ".join(f"Rule {i}: be precise." for i in range(30))
TURNS = ["Explain what a CPU cache is.", "Why is it faster than RAM?", "Name one cache level."]

def chat(msgs, cache):
    body = json.dumps({"messages": msgs, "temperature": 0, "max_tokens": 48,
                       "cache_prompt": cache}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    r = json.load(urllib.request.urlopen(req))
    return r["choices"][0]["message"]["content"], r["usage"]["prompt_tokens_details"]["cached_tokens"]

def run(cache):
    msgs = [{"role": "system", "content": SYSTEM}]
    replies, cached = [], []
    for t in TURNS:
        msgs.append({"role": "user", "content": t})
        a, c = chat(msgs, cache)
        replies.append(a); cached.append(c)
        msgs.append({"role": "assistant", "content": a})
    return replies, cached

on, cached_on = run(True)
off, cached_off = run(False)
fails = 0
for i, (a, b) in enumerate(zip(on, off)):
    ok = a == b
    print(f"  [{'PASS' if ok else 'FAIL'}] turn {i+1} identical with/without cache")
    fails += not ok
ok = all(c > 0 for c in cached_on[1:]) and all(c == 0 for c in cached_off)
print(f"  [{'PASS' if ok else 'FAIL'}] cached_tokens on={cached_on} off={cached_off}")
fails += not ok
sys.exit(1 if fails else 0)
PY

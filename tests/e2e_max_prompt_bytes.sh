#!/bin/sh
# --max-prompt-bytes: a chat request whose message text is longer than the cap gets a clean 400
# (never a silent cut), one within the cap is answered, 0 turns the cap off, and GET /v1/config
# reports the value. Needs models/smollm2-q8_0.gguf; skips without it.
set -u
PZ=${PZ:-./pz}
MODEL=${MODEL:-models/smollm2-q8_0.gguf}
[ -x "$PZ" ] && [ -f "$MODEL" ] || { echo "skip: needs $PZ and $MODEL"; exit 0; }
fail=0
run_case() { # cap port long_chars expect_status
    "$PZ" --model "$MODEL" --server --port "$2" --no-config --threads 2 --max-prompt-bytes "$1" >/dev/null 2>&1 &
    pid=$!
    i=0
    until curl -sf "localhost:$2/health" >/dev/null 2>&1; do
        i=$((i + 1)); [ "$i" -gt 60 ] && { echo "FAIL: server did not start"; kill "$pid"; return 1; }
        sleep 0.5
    done
    text=$(head -c "$3" /dev/zero | tr '\0' 'a')
    code=$(curl -s -o /tmp/pz-mpb.out -w '%{http_code}' -H 'Content-Type: application/json' \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$text\"}],\"max_tokens\":2,\"temperature\":0}" \
        "localhost:$2/v1/chat/completions")
    cfg=$(curl -s "localhost:$2/v1/config" | grep -o '"max_prompt_bytes":[0-9]*')
    kill "$pid"; wait "$pid" 2>/dev/null
    [ "$code" = "$4" ] || { echo "FAIL: cap $1, $3 chars: HTTP $code, expected $4"; return 1; }
    [ "$cfg" = "\"max_prompt_bytes\":$1" ] || { echo "FAIL: /v1/config says '$cfg', expected cap $1"; return 1; }
    if [ "$4" = 400 ]; then grep -q 'max-prompt-bytes' /tmp/pz-mpb.out || { echo "FAIL: error text"; return 1; }; fi
    echo "ok: cap $1, $3 chars -> $code"
}
run_case 200 8911 100 200 || fail=1
run_case 200 8912 300 400 || fail=1
run_case 0 8913 300 200 || fail=1
exit $fail

#!/usr/bin/env bash
# e2e_vision_long_prompt.sh — a chat message with an image and a long text is not cut short, and two
# image requests at once both finish (they take every generation unit, so the second waits in the
# queue instead of sharing the first one's buffers).
#
# Usage: tests/e2e_vision_long_prompt.sh <model.gguf> <vision.bin> <projector.bin>
# Needs the two files tools/extract_multimodal.py writes (for example from SmolVLM-256M-Instruct with a
# SmolLM2-135M model of the same width) and python3; not part of `make test`.
set -u

MODEL="${1:?usage: e2e_vision_long_prompt.sh <model.gguf> <vision.bin> <projector.bin>}"
VISION="${2:?vision.bin}"
PROJ="${3:?projector.bin}"
BIN="./pz"
PORT="${E2E_PORT:-8097}"
FAIL=0

"$BIN" --model "$MODEL" --vision "$VISION" --proj "$PROJ" --server --port "$PORT" --parallel 2 \
    --queue-size 4 --web-ui off --no-calibrate --no-update-check --threads 2 \
    >/tmp/pz_e2e_vision.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for _ in $(seq 1 120); do curl -s "localhost:$PORT/health" >/dev/null && break; sleep 1; done

python3 - "$PORT" <<'PY'
import base64, json, sys, threading, urllib.request

port = sys.argv[1]
img = "data:image/jpeg;base64," + base64.b64encode(open("tests/fixtures/strawberry.jpg", "rb").read()).decode()


def ask(text, out, i):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": img}}]}], "max_tokens": 8, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        out[i] = json.load(urllib.request.urlopen(req, timeout=300))["usage"]["prompt_tokens"]
    except Exception as e:  # report, do not raise: the checks below name the failure
        out[i] = f"ERR {e}"


fail = 0
long_text = "word " * 3000  # about 15000 characters: more than the old 8 KiB buffer held
out = {}
ask(long_text, out, 0)
ok = isinstance(out[0], int) and out[0] > 2800
print(f"  [{'PASS' if ok else 'FAIL'}] a 15000-character message with an image keeps its text (prompt tokens: {out[0]})")
fail += not ok

out = {}
threads = [threading.Thread(target=ask, args=("What is this?", out, i)) for i in range(2)]
[t.start() for t in threads]
[t.join() for t in threads]
ok = all(isinstance(v, int) for v in out.values())
print(f"  [{'PASS' if ok else 'FAIL'}] two image requests at once both finish ({out})")
fail += not ok
sys.exit(1 if fail else 0)
PY
FAIL=$?
echo "=== vision long prompt: $([ "$FAIL" = 0 ] && echo passed || echo FAILED) ==="
exit "$FAIL"

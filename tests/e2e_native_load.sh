#!/usr/bin/env bash
# e2e_native_load.sh — the native ternary .bin loader (load_native in src/cli/session_model.c)
# and the legacy .bin tokenizer, on a tiny random model that tests/make_tiny_native.py writes: a
# normal load and generation, and each way a file can be refused (bad magic, too short, truncated
# weights, a MoE file that is too small or has no valid MoE header, a missing file).
#
# Usage: tests/e2e_native_load.sh        (needs ./pz and python3; not part of `make test`)
set -u

BIN="./pz"
DIR="$(mktemp -d)"
trap 'rm -rf "$DIR"' EXIT
FAIL=0

python3 tests/make_tiny_native.py "$DIR" || exit 1
python3 - "$DIR" <<'PY'
import struct, sys
d = sys.argv[1]
m = open(d + "/model.bin", "rb").read()
def w(name, data): open(d + "/" + name, "wb").write(data)
w("truncated.bin", m[:-100])
w("badmagic.bin", b"XXXX" + m[4:])
moe = bytearray(m)
struct.pack_into("<i", moe, 8 + 36, 2)  # scale_mode = 2 (MoE) without a MoE header
w("moe_small.bin", bytes(moe[:100]))
w("moe_zero_hdr.bin", bytes(moe))
w("tiny.bin", m[:40])
PY

run() { # <model file> [extra flags...]
    local model="$1"; shift
    HOME="$DIR" "$BIN" --model "$DIR/$model" --tokenizer "$DIR/tokenizer.bin" \
        --prompt hello --max-tokens 8 --temperature 0 --seed 1 --no-update-check \
        --no-calibrate "$@" </dev/null 2>&1
}

check() { # <name> <expected substring> <output>
    if grep -qF -- "$2" <<<"$3"; then
        echo "  [PASS] $1"
    else
        echo "  [FAIL] $1 (no \"$2\")"
        FAIL=1
    fi
}

out=$(run model.bin)
check "dense native model loads" "Model format: native ternary" "$out"
check "its configuration is printed" "kv_dim:     32" "$out"
check "generation runs to the end" "[gen]" "$out"
out=$(run model.bin --kv-type i8 --classifier int8 --threads 2 --ctx-size 64)
check "quantised KV and INT8 classifier load too" \
    "Classifier: INT8 (user-selected)" "$out"
check "bad magic is refused" "Invalid model configuration." "$(run badmagic.bin)"
check "a file shorter than the header is refused" \
    "Invalid model configuration." "$(run tiny.bin)"
check "truncated weights are refused" \
    "Failed to map weight structures from file." "$(run truncated.bin)"
check "a MoE file that is too small is refused" \
    "MoE model file too small." "$(run moe_small.bin)"
check "a MoE file without a MoE header is refused" \
    "Failed to read MoE config header." "$(run moe_zero_hdr.bin)"
check "a missing file is refused" "Failed to open model file" "$(run does_not_exist.bin)"

if [ "$FAIL" = 0 ]; then
    echo "=== native load: passed ==="
else
    echo "=== native load: FAILED ==="
fi
exit "$FAIL"

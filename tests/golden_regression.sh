#!/usr/bin/env bash
# golden_regression.sh — TS-5.1: golden-output regression + determinism matrix.
#
# For a fixed model + prompt + --temperature 0, every (--simd, --threads)
# combination must produce byte-identical output. Divergence across SIMD
# backends or thread counts at temperature 0 is a real correctness bug (the
# whole point of greedy decoding is that it's deterministic regardless of how
# the dot products are computed). Also checks that the expected substring
# shows up (a coarse "did it stay coherent" signal) and that --json mode
# produces syntactically valid JSON.
#
# Usage: tests/golden_regression.sh <path-to-gguf-model>
set -u

MODEL="${1:?usage: golden_regression.sh <model.gguf>}"
BIN="./adaptive_ai_engine"
THREADS_LIST="${GOLDEN_THREADS:-1 2 4}"
SIMD_LIST="${GOLDEN_SIMD:-scalar avx2 avx512f vnni}"

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN not found or not executable (run 'make release' first)" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "FAIL: model not found: $MODEL" >&2
    exit 1
fi

n_pass=0
n_fail=0

run_once() {
    # Prints the generated text (which may span multiple lines, e.g. --json
    # output): everything on stdout after the first blank line (which always
    # separates the startup banner/config dump from the generated text),
    # with trailing blank lines trimmed. "[gen] N tok/s (...)" goes to
    # stderr, already excluded via 2>/dev/null.
    local simd="$1" threads="$2" prompt="$3" extra="$4"
    $BIN --model "$MODEL" --prompt "$prompt" --max-tokens 16 --temperature 0 \
         --seed 42 --simd "$simd" --threads "$threads" $extra 2>/dev/null \
         | awk 'started{buf = buf == "" ? $0 : buf "\n" $0} NF==0 && !started{started=1} END{sub(/\n+$/, "", buf); print buf}'
}

check_contains() {
    local label="$1" haystack="$2" needle="$3"
    if echo "$haystack" | grep -qF "$needle"; then
        echo "  [PASS] $label (contains \"$needle\")"
        n_pass=$((n_pass + 1))
    else
        echo "  [FAIL] $label -- expected to contain \"$needle\", got: $haystack"
        n_fail=$((n_fail + 1))
    fi
}

echo "=== TS-5.1 golden-output regression: $MODEL ==="
echo "--- Determinism matrix: SIMD x threads, prompt 'The capital of France is' ---"

reference=""
for simd in $SIMD_LIST; do
    for threads in $THREADS_LIST; do
        out=$(run_once "$simd" "$threads" "The capital of France is" "")
        label="simd=$simd threads=$threads"
        if [ -z "$reference" ]; then
            reference="$out"
            echo "  [REF]  $label -> \"$out\""
        fi
        if [ "$out" = "$reference" ]; then
            echo "  [PASS] $label matches reference"
            n_pass=$((n_pass + 1))
        else
            echo "  [FAIL] $label DIVERGED: \"$out\" != reference \"$reference\""
            n_fail=$((n_fail + 1))
        fi
        check_contains "$label content check" "$out" "Paris"
    done
done

echo "--- Factual-completion prompt ---"
# The plan's suggested "2 + 2 =" -> "4" check does not hold for tiny
# instruct-tuned models: SmolLM2-135M-Instruct explicitly refuses arithmetic
# ("I don't have the capability to perform arithmetic operations") even at
# --max-tokens 30 -- a genuine model-capability limitation, not an engine
# bug (confirmed deterministic, coherent refusal text, not garbage). Use a
# factual-completion prompt this model handles reliably instead.
out=$(run_once "auto" "4" "The opposite of hot is" "")
check_contains "factual completion" "$out" "cold"

echo "--- JSON mode: syntactic validity ---"
out=$(run_once "auto" "4" "Return a JSON object with a single key 'ok' set to true." "--json")
if command -v python3 >/dev/null 2>&1; then
    if echo "$out" | python3 -c "import sys, json; json.loads(sys.stdin.read())" 2>/dev/null; then
        echo "  [PASS] json_mode output parses as valid JSON: $out"
        n_pass=$((n_pass + 1))
    else
        echo "  [FAIL] json_mode output does NOT parse as JSON: $out"
        n_fail=$((n_fail + 1))
    fi
else
    echo "  [SKIP] python3 not found, cannot validate JSON syntax"
fi

echo ""
echo "=== $n_pass passed, $n_fail failed ==="
[ "$n_fail" -eq 0 ]

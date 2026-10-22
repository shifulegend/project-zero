#!/usr/bin/env bash
# e2e_mamba_reference.sh — Phase 22-SSM: the engine's Mamba forward pass must
# agree with tools/mamba_reference.py (independent numpy implementation) on a
# real GGUF: identical greedy token for the first steps and top-1 logits
# within a small tolerance (engine classifier is BF16, reference is F32).
#
# Usage: tests/e2e_mamba_reference.sh <mamba.gguf>   (needs python3 + numpy)
set -u
MODEL="${1:?usage: e2e_mamba_reference.sh <mamba.gguf>}"
OUT=$(./pz --model "$MODEL" --prompt "The capital of France is" \
      --max-tokens 3 --temperature 0 --threads 1 --verbose 2>&1)
TOKS=$(echo "$OUT" | grep -m1 "n_prompt=" | sed 's/.*tokens: //' | tr ' ' ',')
REF=$(python3 tools/mamba_reference.py "$MODEL" "$TOKS" 2)
python3 - "$OUT" "$REF" <<'PY'
import re, sys
eng = re.findall(r"step=\d+ sampled=(\d+)\s+top5:\[(\d+)\(([-\d.]+)\)", sys.argv[1])
ref = re.findall(r"gen\d+ top5: (\d+)\(([-\d.]+)\)", sys.argv[2])
fails = 0
for i, ((s, et, ev), (rt, rv)) in enumerate(zip(eng, ref)):
    ok = et == rt and abs(float(ev) - float(rv)) < 0.5
    print(f"  [{'PASS' if ok else 'FAIL'}] step {i}: engine {et}({ev}) reference {rt}({rv})")
    fails += not ok
if len(ref) < 2 or len(eng) < 2:
    print("  [FAIL] missing output"); fails += 1
sys.exit(1 if fails else 0)
PY

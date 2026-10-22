#!/usr/bin/env python3
"""
run_perf_runs.py — Run one /usr/bin/time -v monitored run at the best thread
                   count for each suite.  Outputs to sweep_2026-06-21/<suite>/perf_run.txt

Usage:
    python3 tools/run_perf_runs.py
"""

import datetime
import os
import re
import subprocess
import sys

PROJECT       = "/home/ubuntu/project-zero"
PZ            = f"{PROJECT}/pz"
PZ_BITNET     = f"{PROJECT}/models/bitnet-b1.58-2B-4T.bin"
PZ_BITNET_TOK = f"{PROJECT}/models/bitnet-b1.58-2B-4T_tokenizer_proper.bin"
PZ_SMOL       = f"{PROJECT}/models/SmolLM2-135M-Instruct-f16.gguf"
PZ_SMOL_TOK   = f"{PROJECT}/models/smollm2-135m-tokenizer.bin"
MSFT_DIR      = "/home/ubuntu/bitnet-cpp"
MSFT_GGUF     = f"{MSFT_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA         = "/home/ubuntu/llama.cpp/build/bin/llama-cli"
SWEEP         = f"{PROJECT}/benchmark_results/sweep_2026-06-21"

PROMPT = (
    "Explain what machine learning is and how neural networks work. "
    "Discuss supervised learning, unsupervised learning, and reinforcement learning "
    "with real-world examples of each."
)
MAX_TOKENS = 500

def read_best_thread(suite_name):
    """Parse summary.txt to find the thread count with best gen tok/s."""
    summary_path = f"{SWEEP}/{suite_name}/summary.txt"
    if not os.path.exists(summary_path):
        print(f"  WARNING: no summary for {suite_name}")
        return None
    best_t, best_val = None, -1
    with open(summary_path) as f:
        for line in f:
            m = re.match(r'^(\d+)\s+([\d.]+)', line)
            if m:
                t, val = int(m.group(1)), float(m.group(2))
                if val > best_val:
                    best_val, best_t = val, t
    print(f"  {suite_name}: best thread = t={best_t}  ({best_val:.2f} tok/s)")
    return best_t

def build_cmd(suite_name, t):
    if suite_name == "bitnet_pz_bf16":
        return ([PZ, "--model", PZ_BITNET, "--tokenizer", PZ_BITNET_TOK,
                 "--prompt", PROMPT, "--threads", str(t),
                 "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
                 "--classifier", "bf16"], PROJECT)
    elif suite_name == "bitnet_pz_int4":
        return ([PZ, "--model", PZ_BITNET, "--tokenizer", PZ_BITNET_TOK,
                 "--prompt", PROMPT, "--threads", str(t),
                 "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
                 "--classifier", "int4"], PROJECT)
    elif suite_name == "bitnet_msft":
        return (["python3", "run_inference.py", "-m", MSFT_GGUF,
                 "-p", PROMPT, "-n", str(MAX_TOKENS), "-t", str(t)], MSFT_DIR)
    elif suite_name == "smollm2_pz":
        return ([PZ, "--model", PZ_SMOL, "--tokenizer", PZ_SMOL_TOK,
                 "--prompt", PROMPT, "--threads", str(t),
                 "--max-tokens", str(MAX_TOKENS), "--temperature", "0"], PROJECT)
    elif suite_name == "smollm2_llama":
        return ([LLAMA, "--model", PZ_SMOL, "--prompt", PROMPT,
                 "--threads", str(t), "--n-predict", str(MAX_TOKENS),
                 "--temp", "0", "--single-turn", "--simple-io"], PROJECT)
    return None, None

SUITES = ["bitnet_pz_bf16", "bitnet_pz_int4", "bitnet_msft", "smollm2_pz", "smollm2_llama"]

if __name__ == "__main__":
    os.chdir(PROJECT)
    for suite_name in SUITES:
        t = read_best_thread(suite_name)
        if t is None:
            continue
        cmd, cwd = build_cmd(suite_name, t)
        if cmd is None:
            continue

        # wrap with /usr/bin/time -v
        time_cmd = ["/usr/bin/time", "-v"] + cmd
        print(f"\n[{datetime.datetime.now()}]  PERF RUN: {suite_name}  t={t}")
        print(f"  CMD: {' '.join(time_cmd)}")
        sys.stdout.flush()

        try:
            res = subprocess.run(time_cmd, capture_output=True, text=True,
                                 timeout=600, cwd=cwd)
            out = res.stdout + res.stderr
            header = (
                f"=== PERF RUN: {suite_name}  t={t} ===\n"
                f"DATE: {datetime.datetime.now()}\n"
                f"CMD: {' '.join(time_cmd)}\n"
                f"{'='*60}\n"
            )
            out_path = f"{SWEEP}/{suite_name}/perf_run.txt"
            with open(out_path, "w") as f:
                f.write(header + out)
            print(f"  -> saved {out_path}")

            # print key perf metrics
            for kw in ["Maximum resident", "Elapsed (wall clock)", "Percent of CPU"]:
                for line in out.splitlines():
                    if kw in line:
                        print(f"  {line.strip()}")
        except Exception as e:
            print(f"  ERROR: {e}")

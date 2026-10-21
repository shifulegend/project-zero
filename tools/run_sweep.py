#!/usr/bin/env python3
"""
run_sweep.py — Full benchmark sweep: PZ BF16 + INT4, MSFT bitnet.cpp, llama.cpp
               t=1..8, best-of-3, 500 tokens, sequential runs only.

Usage:
    python3 tools/run_sweep.py --suite bitnet_pz_bf16
    python3 tools/run_sweep.py --suite bitnet_pz_int4
    python3 tools/run_sweep.py --suite bitnet_msft
    python3 tools/run_sweep.py --suite smollm2_pz
    python3 tools/run_sweep.py --suite smollm2_llama
    python3 tools/run_sweep.py --suite all
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
import time

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT       = "/home/ubuntu/project-zero"
PZ            = f"{PROJECT}/pz"
PZ_BITNET     = f"{PROJECT}/models/bitnet-b1.58-2B-4T.bin"
PZ_BITNET_TOK = f"{PROJECT}/models/bitnet-b1.58-2B-4T_tokenizer_proper.bin"
PZ_SMOL       = f"{PROJECT}/models/SmolLM2-135M-Instruct-f16.gguf"
PZ_SMOL_TOK   = f"{PROJECT}/models/smollm2-135m-tokenizer.bin"
MSFT_DIR      = "/home/ubuntu/bitnet-cpp"
MSFT_GGUF     = f"{MSFT_DIR}/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"
LLAMA         = "/home/ubuntu/llama.cpp/build/bin/llama-cli"
OUT_DIR       = f"{PROJECT}/benchmark_results/sweep_2026-06-21"

# ── benchmark config ──────────────────────────────────────────────────────────
THREADS    = list(range(1, 9))
N_RUNS     = 3
MAX_TOKENS = 500
COOLDOWN_S = 10

PROMPT = (
    "Explain what machine learning is and how neural networks work. "
    "Discuss supervised learning, unsupervised learning, and reinforcement learning "
    "with real-world examples of each."
)

# ── command builders ──────────────────────────────────────────────────────────
def cmd_pz_bitnet(t, classifier="bf16"):
    return [PZ, "--model", PZ_BITNET, "--tokenizer", PZ_BITNET_TOK,
            "--prompt", PROMPT, "--threads", str(t),
            "--max-tokens", str(MAX_TOKENS), "--temperature", "0",
            "--classifier", classifier]

def cmd_pz_smol(t):
    return [PZ, "--model", PZ_SMOL, "--tokenizer", PZ_SMOL_TOK,
            "--prompt", PROMPT, "--threads", str(t),
            "--max-tokens", str(MAX_TOKENS), "--temperature", "0"]

def cmd_msft(t):
    return ["python3", "run_inference.py", "-m", MSFT_GGUF,
            "-p", PROMPT, "-n", str(MAX_TOKENS), "-t", str(t)]

def cmd_llama_smol(t):
    return [LLAMA, "--model", PZ_SMOL, "--prompt", PROMPT,
            "--threads", str(t), "--n-predict", str(MAX_TOKENS),
            "--temp", "0", "--single-turn", "--simple-io"]

# ── parsers ───────────────────────────────────────────────────────────────────
def parse_pz(text):
    m = re.search(r'\[gen\]\s+([\d.]+)\s+tok/s', text)
    return (float(m.group(1)), None) if m else (None, None)

def parse_msft(text):
    gen   = re.search(r'eval time\s*=.*?\(\s*([\d.]+)\s+tokens per second\)', text)
    prmpt = re.search(r'prompt eval time\s*=.*?\(\s*([\d.]+)\s+tokens per second\)', text)
    return (float(gen.group(1))   if gen   else None,
            float(prmpt.group(1)) if prmpt else None)

def parse_llama(text):
    m = re.search(r'Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s', text)
    return (float(m.group(2)), float(m.group(1))) if m else (None, None)

# ── run one benchmark ─────────────────────────────────────────────────────────
def run_one(label, cmd, t, r, cwd=None, parser=parse_pz):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}]  {label}  t={t}  run {r}/{N_RUNS}")
    print(f"  CMD: {' '.join(cmd)}")
    sys.stdout.flush()
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=cwd)
        elapsed = time.time() - t0
        out = res.stdout + res.stderr
        gen, prompt = parser(out)
        print(f"  -> gen={gen} tok/s  prompt={prompt} tok/s  ({elapsed:.1f}s)")
        return out, gen, prompt
    except subprocess.TimeoutExpired:
        print("  ERROR: timeout")
        return "", None, None
    except Exception as e:
        print(f"  ERROR: {e}")
        return "", None, None

# ── suite runner ──────────────────────────────────────────────────────────────
def run_suite(suite_name, cmd_fn, parser, t_list=THREADS, cwd=None):
    suite_dir = f"{OUT_DIR}/{suite_name}"
    os.makedirs(suite_dir, exist_ok=True)
    summary = {}

    for t in t_list:
        t_dir = f"{suite_dir}/t{t}"
        os.makedirs(t_dir, exist_ok=True)
        runs_gen, runs_prompt = [], []

        for r in range(1, N_RUNS + 1):
            cmd = cmd_fn(t)
            out_text, gen, prompt = run_one(suite_name, cmd, t, r, cwd=cwd, parser=parser)

            header = (
                f"=== {suite_name.upper()}  t={t}  run={r}/{N_RUNS} ===\n"
                f"DATE: {datetime.datetime.now()}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"PROMPT: {PROMPT}\n"
                f"{'='*60}\n"
            )
            with open(f"{t_dir}/run{r}.txt", "w") as f:
                f.write(header + out_text)

            if gen    is not None: runs_gen.append(gen)
            if prompt is not None: runs_prompt.append(prompt)

            if r < N_RUNS:
                print(f"  cooldown {COOLDOWN_S}s ...")
                time.sleep(COOLDOWN_S)

        best_gen    = max(runs_gen)    if runs_gen    else None
        best_prompt = max(runs_prompt) if runs_prompt else None
        summary[t]  = {"best_gen": best_gen, "best_prompt": best_prompt,
                        "runs_gen": runs_gen, "runs_prompt": runs_prompt}

        print(f"\n  *** t={t}  BEST gen={best_gen} tok/s  prompt={best_prompt} tok/s")
        print(f"      all runs gen: {runs_gen}")

        if t < t_list[-1]:
            print(f"  cooldown {COOLDOWN_S}s before next thread count ...")
            time.sleep(COOLDOWN_S)

    # write summary
    with open(f"{suite_dir}/summary.txt", "w") as f:
        f.write(f"Suite: {suite_name}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Prompt: {PROMPT}\n")
        f.write(f"Max tokens: {MAX_TOKENS}  |  Runs per thread: {N_RUNS}\n\n")
        f.write(f"{'t':<4}  {'best_gen (tok/s)':<20}  {'best_prompt (tok/s)':<22}  all_gen_runs\n")
        f.write("-" * 80 + "\n")
        for t, d in summary.items():
            bg = f"{d['best_gen']:.2f}"    if d['best_gen']    else "N/A"
            bp = f"{d['best_prompt']:.2f}" if d['best_prompt'] else "N/A"
            f.write(f"{t:<4}  {bg:<20}  {bp:<22}  {d['runs_gen']}\n")
    print(f"  Summary -> {suite_dir}/summary.txt")
    return summary

# ── suites ────────────────────────────────────────────────────────────────────
SUITES = {
    "bitnet_pz_bf16": (lambda t: cmd_pz_bitnet(t, "bf16"), parse_pz,    None),
    "bitnet_pz_int4": (lambda t: cmd_pz_bitnet(t, "int4"), parse_pz,    None),
    "bitnet_msft":    (cmd_msft,                            parse_msft,  MSFT_DIR),
    "smollm2_pz":     (cmd_pz_smol,                         parse_pz,    None),
    "smollm2_llama":  (cmd_llama_smol,                      parse_llama, None),
}

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite",   default="all",
                    help="Suite or 'all'. Options: " + ", ".join(SUITES))
    ap.add_argument("--threads", default=None,
                    help="Comma-separated, e.g. '1,2,4,8' (default: 1-8)")
    args = ap.parse_args()

    t_list = THREADS
    if args.threads:
        t_list = [int(x) for x in args.threads.split(",")]

    os.chdir(PROJECT)

    suites_to_run = list(SUITES) if args.suite == "all" else [args.suite]
    all_results = {}

    for idx, sname in enumerate(suites_to_run):
        if sname not in SUITES:
            print(f"Unknown suite: {sname}. Options: {list(SUITES)}")
            sys.exit(1)
        cmd_fn, parse_fn, cwd = SUITES[sname]
        print(f"\n{'='*70}")
        print(f"  SUITE {idx+1}/{len(suites_to_run)}: {sname}  "
              f"t={t_list}  best-of-{N_RUNS}  {MAX_TOKENS} tok")
        print(f"{'='*70}")
        all_results[sname] = run_suite(sname, cmd_fn, parse_fn, t_list, cwd)

        if idx < len(suites_to_run) - 1:
            print(f"\n  *** SUITE DONE — 30s cooldown before next suite ***")
            time.sleep(30)

    print(f"\n{'='*70}")
    print("  COMBINED RESULTS")
    print(f"{'='*70}")
    for sname, results in all_results.items():
        if not results:
            continue
        peak_t   = max(results, key=lambda t: results[t]["best_gen"] or 0)
        peak_gen = results[peak_t]["best_gen"]
        peak_pmt = results[peak_t]["best_prompt"]
        bp = f"{peak_pmt:.2f}" if peak_pmt else "N/A"
        print(f"  {sname:<22}  peak gen={peak_gen:.2f} tok/s @ t={peak_t}  "
              f"prompt={bp} tok/s")

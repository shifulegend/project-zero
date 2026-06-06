#!/usr/bin/env python3
"""
bench_phase34.py — Phase 34.1 Performance Benchmark
Measures T=1..8 engine tok/s with full psutil system monitoring.

Usage:
    python tools/bench_phase34.py                          # clean run
    python tools/bench_phase34.py --bg-load               # with synthetic background load
    python tools/bench_phase34.py --threads 1 2 3 4       # specific thread counts only
    python tools/bench_phase34.py --out-csv bench_out.csv # save CSV

Requirements: psutil (pip install psutil)
"""

import argparse
import csv
import math
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time

try:
    import psutil
except ImportError:
    print("ERROR: pip install psutil")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE      = os.path.join(PROJECT_ROOT, "adaptive_ai_engine")
MODEL       = os.path.join(PROJECT_ROOT, "models", "bitnet-b1.58-2B-4T.bin")
TOKENIZER   = os.path.join(PROJECT_ROOT, "models", "bitnet-b1.58-2B-4T_tokenizer_proper.bin")
PROMPT      = "Explain the difference between supervised and unsupervised learning in machine learning."
MAX_TOKENS  = 40
COOLDOWN_S  = 15   # seconds between runs

# ── psutil snapshot ──────────────────────────────────────────────────────────
def system_snapshot():
    vm     = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1, percpu=True)
    load   = os.getloadavg()
    procs  = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            c = p.info['cpu_percent']
            if c and c > 1.0:
                procs.append((p.info['name'], c))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: -x[1])
    return {
        'ram_used_gb'  : vm.used  / 1e9,
        'ram_avail_gb' : vm.available / 1e9,
        'ram_total_gb' : vm.total / 1e9,
        'cpu_per_core' : cpu_pct,
        'cpu_mean'     : sum(cpu_pct) / len(cpu_pct),
        'load_1m'      : load[0],
        'load_5m'      : load[1],
        'top_procs'    : procs[:6],
    }


def monitor_loop(stop_evt, samples, interval=0.5):
    """Background thread: collect psutil snapshots every `interval` seconds."""
    while not stop_evt.is_set():
        samples.append(system_snapshot())
        time.sleep(interval)


# ── Background CPU load ───────────────────────────────────────────────────────
def _cpu_burner():
    """Endless tight loop consuming one logical CPU."""
    while True:
        _ = math.sqrt(12345678.9)

def start_bg_load(n_workers=2):
    procs = []
    for _ in range(n_workers):
        p = multiprocessing.Process(target=_cpu_burner, daemon=True)
        p.start()
        procs.append(p)
    return procs

def stop_bg_load(procs):
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=3)


# ── Parse tok/s from engine output ───────────────────────────────────────────
_TOKS_RE = re.compile(r'(\d+\.?\d*)\s*tok/s', re.IGNORECASE)

def extract_toks(output: str) -> float:
    hits = _TOKS_RE.findall(output)
    if hits:
        # Take the last number (generation rate, not prefill)
        return float(hits[-1])
    # Fallback: look for "N tokens in M.Ms" pattern
    m = re.search(r'(\d+)\s+tokens?\s+in\s+([\d.]+)\s*s', output, re.IGNORECASE)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return 0.0


# ── Single benchmark run ─────────────────────────────────────────────────────
def run_once(n_threads: int, extra_flags=None) -> dict:
    cmd = [
        ENGINE,
        "--model",     MODEL,
        "--tokenizer", TOKENIZER,
        "--prompt",    PROMPT,
        "--max-tokens", str(MAX_TOKENS),
        "--threads",   str(n_threads),
    ]
    if extra_flags:
        cmd += extra_flags

    # Pre-run snapshot
    snap_pre = system_snapshot()

    # Start monitor thread
    stop_evt = threading.Event()
    samples  = []
    mon = threading.Thread(target=monitor_loop, args=(stop_evt, samples, 0.25), daemon=True)
    mon.start()

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        stop_evt.set()
        return {"error": "timeout", "n_threads": n_threads}
    elapsed = time.perf_counter() - t0

    stop_evt.set()
    mon.join(timeout=2)

    # Post-run snapshot
    snap_post = system_snapshot()

    stdout = proc.stdout + proc.stderr
    toks   = extract_toks(stdout)

    # Aggregate monitor samples
    if samples:
        mean_cpu   = sum(s['cpu_mean']     for s in samples) / len(samples)
        mean_ram   = sum(s['ram_used_gb']  for s in samples) / len(samples)
        peak_ram   = max(s['ram_used_gb']  for s in samples)
        avail_ram  = min(s['ram_avail_gb'] for s in samples)
    else:
        mean_cpu = snap_pre['cpu_mean']
        mean_ram = snap_pre['ram_used_gb']
        peak_ram = snap_pre['ram_used_gb']
        avail_ram = snap_pre['ram_avail_gb']

    return {
        "n_threads"   : n_threads,
        "toks_s"      : round(toks, 2),
        "elapsed_s"   : round(elapsed, 2),
        "cpu_mean_pct": round(mean_cpu, 1),
        "ram_used_gb" : round(mean_ram, 2),
        "ram_peak_gb" : round(peak_ram, 2),
        "ram_avail_gb": round(avail_ram, 2),
        "load_1m"     : round(snap_pre['load_1m'], 2),
        "top_procs"   : snap_pre['top_procs'],
        "stdout"      : stdout,
        "returncode"  : proc.returncode,
    }


# ── Suite runner ─────────────────────────────────────────────────────────────
def run_suite(thread_counts, label, extra_flags=None):
    print(f"\n{'='*60}")
    print(f"Suite: {label}")
    print(f"{'='*60}")

    # Pre-suite system state
    snap = system_snapshot()
    print(f"  RAM: {snap['ram_used_gb']:.2f} GB used / {snap['ram_total_gb']:.2f} GB total "
          f"({snap['ram_avail_gb']:.2f} GB avail)")
    print(f"  CPU avg: {snap['cpu_mean']:.1f}%  load: {snap['load_1m']:.2f}")
    top = ", ".join(f"{n}={c:.0f}%" for n, c in snap['top_procs'][:4])
    print(f"  Top procs: {top}")

    results = []
    for t in thread_counts:
        print(f"\n  T={t} ... ", end="", flush=True)
        r = run_once(t, extra_flags)
        if "error" in r:
            print(f"TIMEOUT")
        else:
            print(f"{r['toks_s']:.2f} tok/s  |  CPU {r['cpu_mean_pct']:.1f}%  "
                  f"RAM {r['ram_used_gb']:.2f}GB used  load {r['load_1m']:.2f}")
        results.append(r)
        if t != thread_counts[-1]:
            print(f"    [cooldown {COOLDOWN_S}s]", flush=True)
            time.sleep(COOLDOWN_S)

    return results


# ── Table printer ─────────────────────────────────────────────────────────────
def print_table(results, label):
    print(f"\n### {label}\n")
    print(f"| T | tok/s | CPU% | RAM used | RAM avail | load 1m | elapsed |")
    print(f"|---|-------|------|----------|-----------|---------|---------|")
    for r in results:
        if "error" in r:
            print(f"| {r['n_threads']} | TIMEOUT | — | — | — | — | — |")
        else:
            print(f"| {r['n_threads']} | {r['toks_s']:.2f} | {r['cpu_mean_pct']:.1f} | "
                  f"{r['ram_used_gb']:.2f} GB | {r['ram_avail_gb']:.2f} GB | "
                  f"{r['load_1m']:.2f} | {r['elapsed_s']:.1f}s |")


# ── Top-process snapshot for report ──────────────────────────────────────────
def print_top_procs(results):
    if not results:
        return
    # Use the first sample's top_procs as representative
    for r in results:
        if "top_procs" in r and r["top_procs"]:
            print("\n**Background processes during run:**")
            print("| Process | CPU% |")
            print("|---------|------|")
            for name, cpu in r["top_procs"]:
                print(f"| {name} | {cpu:.1f} |")
            break


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(all_results, path):
    fields = ["suite", "n_threads", "toks_s", "cpu_mean_pct",
              "ram_used_gb", "ram_peak_gb", "ram_avail_gb", "load_1m", "elapsed_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for suite_name, results in all_results:
            for r in results:
                row = dict(r)
                row["suite"] = suite_name
                w.writerow(row)
    print(f"\nCSV saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    global COOLDOWN_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+",
                    default=list(range(1, 9)),
                    help="Thread counts to test (default: 1-8)")
    ap.add_argument("--bg-load", action="store_true",
                    help="Add synthetic background CPU load (2 workers)")
    ap.add_argument("--bg-workers", type=int, default=2,
                    help="Number of background CPU burner workers (default: 2)")
    ap.add_argument("--out-csv", default=None, help="Save results to CSV file")
    ap.add_argument("--simd", default=None,
                    help="SIMD override (avx2, avx512f, vnni, scalar)")
    ap.add_argument("--classifier", default=None,
                    help="Classifier override (bf16, int8, int4, auto-fast)")
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_S,
                    help=f"Cooldown seconds between runs (default: {COOLDOWN_S})")
    args = ap.parse_args()

    COOLDOWN_S = args.cooldown

    if not os.path.exists(ENGINE):
        print(f"ERROR: engine not found at {ENGINE}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"ERROR: model not found at {MODEL}")
        sys.exit(1)

    extra_flags = []
    if args.simd:
        extra_flags += ["--simd", args.simd]
    if args.classifier:
        extra_flags += ["--classifier", args.classifier]

    print("Phase 34.1 — Engine Benchmark with psutil Monitoring")
    print("=====================================================")
    print(f"Engine:     {ENGINE}")
    print(f"Model:      {MODEL}")
    print(f"Prompt:     {PROMPT[:60]}...")
    print(f"Max tokens: {MAX_TOKENS}")
    print(f"Threads:    {args.threads}")
    print(f"SIMD:       {args.simd or 'auto'}")
    print(f"Classifier: {args.classifier or 'auto (bf16)'}")
    print(f"BG load:    {'yes (' + str(args.bg_workers) + ' workers)' if args.bg_load else 'no'}")

    all_results = []
    bg_procs = []

    # ── Suite 1: Clean run ────────────────────────────────────────────────────
    r_clean = run_suite(args.threads, "Clean (no background load)", extra_flags)
    all_results.append(("clean", r_clean))
    print_table(r_clean, "Clean Run — tok/s vs Thread Count")
    print_top_procs(r_clean)

    # ── Suite 2: With background CPU load ─────────────────────────────────────
    if args.bg_load:
        print(f"\n[Starting {args.bg_workers} background CPU burner(s)...]")
        bg_procs = start_bg_load(args.bg_workers)
        time.sleep(2)  # let them stabilize

        r_load = run_suite(args.threads, f"With {args.bg_workers} background CPU burner(s)", extra_flags)
        all_results.append((f"bg_{args.bg_workers}workers", r_load))
        print_table(r_load, f"Background Load ({args.bg_workers} workers) — tok/s vs Thread Count")
        print_top_procs(r_load)

        stop_bg_load(bg_procs)
        print("[Background load stopped]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for suite_name, results in all_results:
        valid = [r for r in results if "error" not in r and r["toks_s"] > 0]
        if valid:
            best = max(valid, key=lambda r: r["toks_s"])
            print(f"  {suite_name:30s}  peak {best['toks_s']:.2f} tok/s @ T={best['n_threads']}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.out_csv:
        save_csv(all_results, args.out_csv)

    return all_results


if __name__ == "__main__":
    main()

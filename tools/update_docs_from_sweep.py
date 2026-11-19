#!/usr/bin/env python3
"""
update_docs_from_sweep.py — Read sweep_2026-06-21 summaries and update
README.md + PERFORMANCE_CEILING_REPORT.md with fresh results.

Run AFTER run_sweep.py and run_perf_runs.py have completed.
"""

import os
import re

PROJECT = os.environ.get("PZ_PROJECT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SWEEP   = f"{PROJECT}/benchmark_results/sweep_2026-06-21"

# ── load summaries ────────────────────────────────────────────────────────────
def load_summary(suite_name):
    """Returns dict: t -> {best_gen, best_prompt}"""
    path = f"{SWEEP}/{suite_name}/summary.txt"
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return {}
    results = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^(\d+)\s+([\d.N/A]+)\s+([\d.N/A]+)', line)
            if m:
                t    = int(m.group(1))
                gen  = float(m.group(2)) if m.group(2) != "N/A" else None
                pmt  = float(m.group(3)) if m.group(3) != "N/A" else None
                results[t] = {"best_gen": gen, "best_prompt": pmt}
    return results

def load_perf(suite_name):
    """Returns dict with perf metrics: wall_time, max_rss, cpu_pct."""
    path = f"{SWEEP}/{suite_name}/perf_run.txt"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        text = f.read()
    result = {}
    m = re.search(r'Elapsed \(wall clock\) time.*?:\s*(\S+)', text)
    if m: result["wall_time"] = m.group(1)
    m = re.search(r'Maximum resident set size.*?:\s*(\d+)', text)
    if m: result["max_rss_kb"] = int(m.group(1))
    m = re.search(r'Percent of CPU this job got:\s*(\S+)', text)
    if m: result["cpu_pct"] = m.group(1)
    return result

# ── table builders ────────────────────────────────────────────────────────────
def make_bitnet_table(pz_bf16, pz_int4, msft):
    """BitNet 3-engine comparison table."""
    rows = ["| Threads | PZ BF16 (tok/s) | PZ INT4 (tok/s) | MSFT bitnet.cpp | BF16 Gain | INT4 Gain |",
            "|---|---|---|---|---|---|"]
    for t in range(1, 9):
        b16   = pz_bf16.get(t, {}).get("best_gen")
        i4    = pz_int4.get(t, {}).get("best_gen")
        ms    = msft.get(t, {}).get("best_gen")
        b16s  = f"{b16:.2f}" if b16 else "N/A"
        i4s   = f"**{i4:.2f}**" if i4 else "N/A"
        mss   = f"{ms:.2f}"   if ms  else "N/A"
        b16g  = f"+{(b16/ms-1)*100:.0f}%" if b16 and ms else "N/A"
        i4g   = f"+{(i4/ms-1)*100:.0f}%" if i4  and ms else "N/A"
        rows.append(f"| {t} | {b16s} | {i4s} | {mss} | {b16g} | {i4g} |")
    return "\n".join(rows)

def make_smollm2_table(pz, llama):
    """SmolLM2 PZ vs llama.cpp table."""
    rows = ["| Threads | Project Zero BF16 (tok/s) | llama.cpp (tok/s) | PZ Gain |",
            "|---|---|---|---|"]
    for t in range(1, 9):
        pzv  = pz.get(t, {}).get("best_gen")
        lcv  = llama.get(t, {}).get("best_gen")
        pzs  = f"{pzv:.2f}" if pzv else "N/A"
        lcs  = f"{lcv:.2f}" if lcv else "N/A"
        if pzv and lcv:
            gain = (pzv/lcv - 1)*100
            gs   = f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"
        else:
            gs = "N/A"
        rows.append(f"| {t} | {pzs} | {lcs} | {gs} |")
    return "\n".join(rows)

def make_prompt_speed_table(pz_bf16, pz_int4, msft, pz_smol, llama):
    """Prompt tok/s table for engines that report it."""
    rows = ["| Engine | Model | t=best | Prompt tok/s | Gen tok/s |",
            "|---|---|---|---|---|"]

    def best_row(data, label, model):
        if not data:
            return None
        t = max(data, key=lambda x: data[x].get("best_gen") or 0)
        gen  = data[t].get("best_gen")
        pmt  = data[t].get("best_prompt")
        gs = f"{gen:.2f}" if gen else "N/A"
        ps = f"{pmt:.2f}" if pmt else "N/A (not reported)"
        return f"| {label} | {model} | {t} | {ps} | {gs} |"

    for row in [
        best_row(pz_bf16, "PZ BF16",          "BitNet b1.58-2B-4T"),
        best_row(pz_int4, "PZ INT4",           "BitNet b1.58-2B-4T"),
        best_row(msft,    "MSFT bitnet.cpp",   "BitNet b1.58-2B-4T"),
        best_row(pz_smol, "PZ BF16",           "SmolLM2-135M F16"),
        best_row(llama,   "llama.cpp",         "SmolLM2-135M F16"),
    ]:
        if row:
            rows.append(row)
    return "\n".join(rows)

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading sweep summaries ...")
    pz_bf16  = load_summary("bitnet_pz_bf16")
    pz_int4  = load_summary("bitnet_pz_int4")
    msft     = load_summary("bitnet_msft")
    pz_smol  = load_summary("smollm2_pz")
    llama    = load_summary("smollm2_llama")

    perf_pz_bf16 = load_perf("bitnet_pz_bf16")
    perf_pz_int4 = load_perf("bitnet_pz_int4")
    perf_msft    = load_perf("bitnet_msft")
    perf_smol    = load_perf("smollm2_pz")
    perf_llama   = load_perf("smollm2_llama")

    def peak(data):
        if not data: return (None, None)
        t = max(data, key=lambda x: data[x].get("best_gen") or 0)
        return t, data[t].get("best_gen")

    pk_bf16_t,  pk_bf16  = peak(pz_bf16)
    pk_int4_t,  pk_int4  = peak(pz_int4)
    pk_msft_t,  pk_msft  = peak(msft)
    pk_smol_t,  pk_smol  = peak(pz_smol)
    pk_llama_t, pk_llama = peak(llama)

    print("\n=== PEAKS ===")
    print(f"  PZ BF16 (BitNet):   {pk_bf16:.2f} tok/s @ t={pk_bf16_t}")
    print(f"  PZ INT4 (BitNet):   {pk_int4:.2f} tok/s @ t={pk_int4_t}")
    print(f"  MSFT bitnet.cpp:    {pk_msft:.2f} tok/s @ t={pk_msft_t}")
    print(f"  PZ BF16 (SmolLM2):  {pk_smol:.2f} tok/s @ t={pk_smol_t}")
    print(f"  llama.cpp (SmolLM2):{pk_llama:.2f} tok/s @ t={pk_llama_t}")

    if pk_msft and pk_int4:
        print(f"\n  INT4 vs MSFT speedup: {pk_int4/pk_msft:.2f}x")
    if pk_msft and pk_bf16:
        print(f"  BF16 vs MSFT speedup: {pk_bf16/pk_msft:.2f}x")

    print("\n=== BITNET TABLE ===")
    print(make_bitnet_table(pz_bf16, pz_int4, msft))

    print("\n=== SMOLLM2 TABLE ===")
    print(make_smollm2_table(pz_smol, llama))

    print("\n=== PROMPT SPEED TABLE ===")
    print(make_prompt_speed_table(pz_bf16, pz_int4, msft, pz_smol, llama))

    print("\n=== PERF STATS ===")
    for name, perf in [
        ("PZ BF16 BitNet",      perf_pz_bf16),
        ("PZ INT4 BitNet",      perf_pz_int4),
        ("MSFT bitnet.cpp",     perf_msft),
        ("PZ BF16 SmolLM2",     perf_smol),
        ("llama.cpp SmolLM2",   perf_llama),
    ]:
        if perf:
            rss_mb = perf.get("max_rss_kb", 0) / 1024
            print(f"  {name:<22}: wall={perf.get('wall_time','?')}  "
                  f"rss={rss_mb:.0f}MB  cpu={perf.get('cpu_pct','?')}")

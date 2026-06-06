#!/usr/bin/env python3
"""
compare_tensors.py — Step-by-step tensor comparison between llama.cpp and our engine.

Usage:
    python3 compare_tensors.py llama_tensors.csv ours_tensors.csv [--layer N] [--step NAME]

Each CSV has columns:
    layer, step, n_elem, v0..v7, mean, absmax

The script matches rows by (layer, step) and reports:
  - Whether values match within tolerance
  - The first divergence point
  - A side-by-side diff for any mismatch
"""

import sys
import csv
import argparse
import math

# Tolerance for float comparison (relative error threshold)
REL_TOL  = 1e-3   # 0.1% relative error
ABS_TOL  = 1e-5   # absolute tolerance for near-zero values

STEP_ORDER = [
    "attn_norm", "kv_cmpr", "k_pe", "q", "q_pe", "kv",
    "Qcur", "Kcur", "Vcur_cont",
    "ffn_inp", "ffn_norm",
    "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_out",
    "ffn_shexp", "ffn_out", "l_out",
    "result_norm", "result_output",
]

ANSI_RED    = "\033[91m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN   = "\033[96m"
ANSI_RESET  = "\033[0m"


def load_csv(path):
    """Load a tensor dump CSV into a dict keyed by (layer, step)."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer = int(row["layer"])
            step  = row["step"].strip()
            n     = int(row["n_elem"])
            vals  = [float(row[f"v{i}"]) for i in range(8)]
            mean  = float(row["mean"])
            abmx  = float(row["absmax"])
            key = (layer, step)
            # Keep first occurrence (first token position)
            if key not in rows:
                rows[key] = {"n": n, "vals": vals, "mean": mean, "absmax": abmx}
    return rows


def rel_err(a, b):
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom


def check_match(a, b, label):
    """Returns (matched, details_str)."""
    errs = []
    for i in range(8):
        re = rel_err(a["vals"][i], b["vals"][i])
        if re > REL_TOL and abs(a["vals"][i] - b["vals"][i]) > ABS_TOL:
            errs.append((i, a["vals"][i], b["vals"][i], re))

    mean_re  = rel_err(a["mean"],   b["mean"])
    abmx_re  = rel_err(a["absmax"], b["absmax"])

    matched = (len(errs) == 0 and mean_re < REL_TOL and abmx_re < REL_TOL)

    lines = []
    if not matched:
        lines.append(f"  {ANSI_CYAN}n_elem:{ANSI_RESET} llama={a['n']}  ours={b['n']}")
        lines.append(f"  {'idx':>4}  {'llama':>14}  {'ours':>14}  {'rel_err':>10}")
        for i, lv, ov, re in errs[:5]:
            flag = f"{ANSI_RED}✗{ANSI_RESET}"
            lines.append(f"  {i:>4}  {lv:>14.7g}  {ov:>14.7g}  {re:>10.4%} {flag}")
        lines.append(f"  {'mean':>4}  {a['mean']:>14.7g}  {b['mean']:>14.7g}  {mean_re:>10.4%}")
        lines.append(f"  {'absmax':>4}  {a['absmax']:>14.7g}  {b['absmax']:>14.7g}  {abmx_re:>10.4%}")
    return matched, "\n".join(lines)


def compare(llama_csv, ours_csv, filter_layer=None, filter_step=None):
    print(f"\n{ANSI_CYAN}Loading llama.cpp dump:{ANSI_RESET} {llama_csv}")
    llama = load_csv(llama_csv)
    print(f"{ANSI_CYAN}Loading our engine dump:{ANSI_RESET} {ours_csv}")
    ours  = load_csv(ours_csv)

    print(f"\n{ANSI_CYAN}llama.cpp checkpoints: {len(llama)}{ANSI_RESET}")
    print(f"{ANSI_CYAN}Our engine checkpoints: {len(ours)}{ANSI_RESET}\n")

    # All keys that appear in either dump
    all_keys = sorted(set(llama.keys()) | set(ours.keys()),
                      key=lambda k: (k[0], STEP_ORDER.index(k[1]) if k[1] in STEP_ORDER else 99))

    first_divergence = None
    n_match = 0
    n_mismatch = 0
    n_missing = 0

    print(f"{'LAYER':>6}  {'STEP':<22}  {'STATUS':>8}  DETAILS")
    print("─" * 72)

    for (layer, step) in all_keys:
        if filter_layer is not None and layer != filter_layer:
            continue
        if filter_step is not None and step != filter_step:
            continue

        in_llama = (layer, step) in llama
        in_ours  = (layer, step) in ours

        if in_llama and not in_ours:
            n_missing += 1
            status = f"{ANSI_YELLOW}MISSING_OURS{ANSI_RESET}"
            print(f"{layer:>6}  {step:<22}  {status}")
            continue

        if in_ours and not in_llama:
            n_missing += 1
            status = f"{ANSI_YELLOW}MISSING_LLAMA{ANSI_RESET}"
            print(f"{layer:>6}  {step:<22}  {status}")
            continue

        matched, details = check_match(llama[(layer, step)], ours[(layer, step)], step)

        if matched:
            n_match += 1
            status = f"{ANSI_GREEN}  OK{ANSI_RESET}"
            print(f"{layer:>6}  {step:<22}  {status}")
        else:
            n_mismatch += 1
            if first_divergence is None:
                first_divergence = (layer, step)
            status = f"{ANSI_RED}MISMATCH{ANSI_RESET}"
            print(f"{layer:>6}  {step:<22}  {status}")
            print(details)

    print("\n" + "═" * 72)
    print(f"  Matched:    {ANSI_GREEN}{n_match}{ANSI_RESET}")
    print(f"  Mismatched: {ANSI_RED}{n_mismatch}{ANSI_RESET}")
    print(f"  Missing:    {ANSI_YELLOW}{n_missing}{ANSI_RESET}")
    if first_divergence:
        l, s = first_divergence
        print(f"\n  {ANSI_RED}⚡ First divergence: layer={l}, step={s}{ANSI_RESET}")
    else:
        print(f"\n  {ANSI_GREEN}✓ All present checkpoints match!{ANSI_RESET}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Compare tensor dumps from llama.cpp and our engine")
    ap.add_argument("llama_csv", help="CSV dump from llama_dump (llama.cpp)")
    ap.add_argument("ours_csv",  help="CSV dump from our engine (--dump-tensors)")
    ap.add_argument("--layer", type=int, default=None, help="Filter to specific layer")
    ap.add_argument("--step",  type=str, default=None, help="Filter to specific step name")
    args = ap.parse_args()

    compare(args.llama_csv, args.ours_csv,
            filter_layer=args.layer, filter_step=args.step)


if __name__ == "__main__":
    main()

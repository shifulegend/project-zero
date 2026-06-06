#!/usr/bin/env python3
"""
parse_llama_log.py — Parses llama-eval-callback stderr output into a CSV
that matches the format written by our engine's --dump-tensors.

The eval-callback output looks like:
    common_debug_cb_eval:          attn_norm = (f32) RMS_NORM(...) = {2048, 1, 1, 1}
        [
            [
                [      1.234,    -0.567, ..., ],
            ],
        ]
        sum = 42.0

Usage:
    python3 parse_llama_log.py llama_raw.txt llama_tensors.csv
"""

import sys
import re
import csv
import math

# Tensor names we care about (must match CAPTURE_PATTERNS in compare_tensors.py)
KEEP = {
    "attn_norm", "ffn_norm", "kv_cmpr", "kv", "q", "q_pe", "k_pe",
    "Qcur", "Kcur", "Vcur_cont", "Vcur",
    "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_out",
    "ffn_shexp", "ffn_out", "l_out",
    "result_norm", "result_output",
}

# Matches the header line for each tensor
RE_HEADER = re.compile(
    r"common_debug_cb_eval:\s+(\S+)\s+=\s+\((\w+)\).*?=\s+\{([^}]+)\}"
)
# Matches float values in content lines
RE_FLOAT  = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?")
# Matches the sum line
RE_SUM    = re.compile(r"\bsum\s*=\s*([-+]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def extract_layer(name):
    """Try to strip a trailing _<N> layer index from the tensor name."""
    m = re.search(r"_(\d+)$", name)
    if m:
        return name[: m.start()], int(m.group(1))
    return name, -1


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} llama_raw.txt llama_tensors.csv")
        sys.exit(1)

    raw_path = sys.argv[1]
    csv_path = sys.argv[2]

    rows = []          # (layer, step, vals_list)
    cur_name  = None
    cur_layer = -1
    cur_vals  = []
    collecting = False

    with open(raw_path, errors="replace") as f:
        for line in f:
            m = RE_HEADER.search(line)
            if m:
                # Flush any pending tensor
                if cur_name and cur_vals:
                    rows.append((cur_layer, cur_name, cur_vals[:]))

                raw_name = m.group(1)
                base, layer = extract_layer(raw_name)
                cur_name  = base
                cur_layer = layer
                cur_vals  = []
                collecting = (base in KEEP)
                continue

            if not collecting:
                continue

            # Accumulate float values
            floats = RE_FLOAT.findall(line)
            cur_vals.extend(float(v) for v in floats)

            # Sum line marks end of this tensor's value block
            sm = RE_SUM.search(line)
            if sm and cur_vals:
                rows.append((cur_layer, cur_name, cur_vals[:]))
                cur_name  = None
                cur_vals  = []
                collecting = False

    # Flush last tensor if file ended without a sum line
    if cur_name and cur_vals:
        rows.append((cur_layer, cur_name, cur_vals[:]))

    # Write CSV
    written = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer","step","n_elem",
                    "v0","v1","v2","v3","v4","v5","v6","v7",
                    "mean","absmax"])
        for (layer, name, vals) in rows:
            if not vals:
                continue
            n    = len(vals)
            v8   = (vals + [0.0] * 8)[:8]
            mean = sum(vals) / n
            abmx = max(abs(v) for v in vals)
            w.writerow([layer, name, n] + v8 + [mean, abmx])
            written += 1

    print(f"[parse_llama_log] {written} tensor rows → {csv_path}")


if __name__ == "__main__":
    main()

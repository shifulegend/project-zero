#!/usr/bin/env python3
"""
make_sweep_screenshots.py — Generate PNG screenshots for all sweep results.

Reads summary.txt + run{1..3}.txt files from each suite, renders the best run
as a dark-theme terminal screenshot and saves to benchmark_results/sweep_2026-06-21/screenshots/.

Usage:
    python3 tools/make_sweep_screenshots.py                      # 2026-06-21 sweep (defaults)
    python3 tools/make_sweep_screenshots.py --sweep DIR --hardware "..." \
        [--suite NAME=Label ...]   # any sweep from tools/bench_sweep_seq.py (reads DIR/meta.json)
"""

import argparse
import json
import os
import re
import sys
from PIL import Image, ImageDraw, ImageFont

# ── colour palette ────────────────────────────────────────────────────────────
BG       = (15,  15,  20)
FG       = (204, 204, 204)
GREEN    = (78,  200, 78)
CYAN     = (86,  218, 220)
YELLOW   = (255, 215, 0)
WHITE    = (255, 255, 255)
DIM      = (100, 100, 110)
ORANGE   = (230, 140, 30)
TITLE_BG = (25,  25,  35)
BLUE     = (100, 149, 237)

PADDING = 28
LINE_H  = 21
WIDTH   = 1000

PROJECT = "/home/ubuntu/project-zero"
SWEEP   = f"{PROJECT}/benchmark_results/sweep_2026-06-21"
SS_DIR  = f"{SWEEP}/screenshots"

# ── font ──────────────────────────────────────────────────────────────────────
def load_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

FONT       = load_font(14)
FONT_BOLD  = load_font(15)
FONT_SMALL = load_font(12)

def text_w(text, font=FONT):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]

# ── draw helpers ──────────────────────────────────────────────────────────────
def draw_line(draw, y, text, colour=FG, font=FONT, x=PADDING):
    draw.text((x, y), text, fill=colour, font=font)
    return y + LINE_H

def draw_separator(draw, y, colour=DIM):
    draw.line([(PADDING, y + LINE_H//2), (WIDTH - PADDING, y + LINE_H//2)],
              fill=colour, width=1)
    return y + LINE_H

# ── content extractors ────────────────────────────────────────────────────────
def extract_gen_toks(text):
    m = re.search(r'\[gen\]\s+([\d.]+)\s+tok/s', text)
    return m.group(1) if m else "N/A"

def extract_prompt_toks_msft(text):
    # Order: sampling → prompt eval → eval; second-to-last is prompt eval
    vals = re.findall(r'([\d.]+) tokens per second', text)
    return vals[-2] if len(vals) >= 2 else "N/A"

def extract_eval_toks_msft(text):
    # Last entry is generation eval speed
    vals = re.findall(r'([\d.]+) tokens per second', text)
    return vals[-1] if vals else "N/A"

def extract_llama_speeds(text):
    m = re.search(r'Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s', text)
    if m:
        return m.group(1), m.group(2)
    return "N/A", "N/A"

def extract_date(text):
    m = re.search(r'DATE:\s*(.+)', text)
    return m.group(1).strip() if m else "2026-06-21"

def extract_cmd(text):
    m = re.search(r'CMD:\s*(.+)', text)
    return m.group(1).strip() if m else ""

def shorten_cmd(cmd, max_len=90):
    if len(cmd) <= max_len:
        return cmd
    # shorten model paths
    cmd = re.sub(r'/home/ubuntu/project-zero/', '', cmd)
    cmd = re.sub(r'/home/ubuntu/bitnet-cpp/', '[bitnet-cpp]/', cmd)
    cmd = re.sub(r'/home/ubuntu/llama\.cpp/build/bin/', '[llama.cpp]/', cmd)
    cmd = re.sub(r'models/', 'm/', cmd)
    if len(cmd) > max_len:
        cmd = cmd[:max_len-3] + "..."
    return cmd

# ── main screenshot renderer ──────────────────────────────────────────────────
def make_screenshot(suite_name, t, run_text, out_path, run_num=None):
    """Render one benchmark run as a PNG screenshot."""
    is_msft  = "msft"  in suite_name
    is_llama = "llama" in suite_name
    is_pz    = not is_msft and not is_llama

    date_str = extract_date(run_text)
    cmd_str  = shorten_cmd(extract_cmd(run_text))

    if is_pz:
        gen_speed = extract_gen_toks(run_text)
        pmt_speed = "N/A (not reported by PZ)"
    elif is_msft:
        pmt_speed = extract_prompt_toks_msft(run_text)
        gen_speed = extract_eval_toks_msft(run_text)
    else:  # llama
        pmt_speed, gen_speed = extract_llama_speeds(run_text)

    # extract model output lines (everything between header and perf stats)
    output_lines = []
    in_output = False
    for line in run_text.splitlines():
        if line.startswith("===") or line.startswith("DATE:") or \
           line.startswith("CMD:") or line.startswith("PROMPT:") or \
           line.startswith("="):
            if "===" in line and in_output:
                break
            continue
        if "[gen]" in line or "llama_perf" in line or "[ Prompt:" in line or line.startswith("PERF:"):
            break
        # engine start-up chatter (profile box, loader steps) is not model output
        if is_pz and (line[:1] in "│┌├└" or line.startswith(("[", "Active:", "KV cache", "SIMD", "Classifier",
                                                                 "Attention:", "Hardware", "Calibration"))):
            continue
        if line.strip():
            in_output = True
        if in_output:
            output_lines.append(line)

    # trim output to fit (max 18 lines)
    if len(output_lines) > 18:
        output_lines = output_lines[:8] + ["    [... output truncated ...]"] + output_lines[-4:]

    # ── build rows ────────────────────────────────────────────────────────────
    engine_label = {
        "bitnet_pz_bf16": "Project Zero  BF16  (ternary BitNet b1.58-2B-4T)",
        "bitnet_pz_int4": "Project Zero  INT4  (ternary BitNet b1.58-2B-4T)",
        "bitnet_msft":    "Microsoft bitnet.cpp  i2_s  (BitNet b1.58-2B-4T)",
        "smollm2_pz":     "Project Zero  BF16  (SmolLM2-135M-Instruct F16)",
        "smollm2_llama":  "llama.cpp  (SmolLM2-135M-Instruct F16)",
    }.get(suite_name, suite_name)

    engine_label = LABELS.get(suite_name, engine_label)
    perf = re.search(r'PERF:\s*(.+)', run_text)
    header_rows = [
        ("date",    "Date/Time",        date_str,               CYAN),
        ("hw",      "Hardware",         HARDWARE,               FG),
        ("engine",  "Engine",           engine_label,           GREEN),
        ("threads", "Threads",          str(t),                 YELLOW),
        ("tokens",  "Max tokens",       str(MAX_TOKENS),        FG),
        ("prompt",  "Prompt tok/s",     pmt_speed,              BLUE if pmt_speed != "N/A (not reported by PZ)" else DIM),
        ("gen",     "Generation tok/s", gen_speed + " tok/s",   YELLOW),
    ]
    if perf:
        header_rows.append(("perf", "Wall / RSS / CPU", perf.group(1), FG))
    if run_num:
        header_rows.insert(0, ("run", "Run", run_num, DIM))

    # estimate height
    n_rows = (
        1 +                          # title bar
        2 +                          # separator + cmd
        len(header_rows) + 2 +       # header box
        1 +                          # output header
        len(output_lines) + 2 +      # output
        3                            # bottom margin
    )
    height = n_rows * LINE_H + PADDING * 2

    img  = Image.new("RGB", (WIDTH, height), BG)
    draw = ImageDraw.Draw(img)

    # title bar
    draw.rectangle([(0, 0), (WIDTH, LINE_H + 12)], fill=TITLE_BG)
    title = f"  {engine_label}  ·  t={t}  ·  {date_str}"
    draw.text((PADDING, 6), title, fill=CYAN, font=FONT_BOLD)

    y = LINE_H + 16

    # command
    draw.text((PADDING, y), "$ " + cmd_str, fill=ORANGE, font=FONT_SMALL)
    y += LINE_H + 4

    # separator
    draw.line([(PADDING, y), (WIDTH-PADDING, y)], fill=DIM, width=1)
    y += 8

    # header box
    label_w = 22 * 8  # ~22 chars
    for _, label, value, colour in header_rows:
        draw.text((PADDING, y),              f"{label:<22}", fill=DIM,    font=FONT)
        draw.text((PADDING + label_w, y),    value,          fill=colour, font=FONT_BOLD if label in ("Generation tok/s", "Engine") else FONT)
        y += LINE_H

    # separator
    y += 4
    draw.line([(PADDING, y), (WIDTH-PADDING, y)], fill=DIM, width=1)
    y += 10

    # output section
    draw.text((PADDING, y), "OUTPUT:", fill=DIM, font=FONT_SMALL)
    y += LINE_H
    for line in output_lines:
        clipped = line[:110]
        draw.text((PADDING + 8, y), clipped, fill=WHITE, font=FONT)
        y += LINE_H

    # bottom tok/s highlight
    y += 6
    draw.line([(PADDING, y), (WIDTH-PADDING, y)], fill=DIM, width=1)
    y += 8
    speed_text = f"Generation: {gen_speed} tok/s"
    if pmt_speed not in ("N/A", "N/A (not reported by PZ)"):
        speed_text += f"   |   Prompt: {pmt_speed} tok/s"
    draw.text((PADDING, y), speed_text, fill=YELLOW, font=FONT_BOLD)

    img.save(out_path)
    print(f"  Saved: {out_path}")

# ── process all suites ────────────────────────────────────────────────────────
def find_best_run(suite_name, t):
    """Return (run_num, text) of the best run (highest gen tok/s)."""
    t_dir = f"{SWEEP}/{suite_name}/t{t}"
    if not os.path.isdir(t_dir):
        return None, None

    is_msft  = "msft"  in suite_name
    is_llama = "llama" in suite_name

    best_val, best_r, best_text = -1, None, None
    for r in range(1, RUNS + 1):
        fpath = f"{t_dir}/run{r}.txt"
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            text = f.read()
        if is_msft:
            # last entry in 'X tokens per second' sequence is gen speed
            vals = re.findall(r'([\d.]+) tokens per second', text)
            val = float(vals[-1]) if vals else None
        elif is_llama:
            m = re.search(r'Generation:\s*([\d.]+)\s*t/s', text)
            val = float(m.group(1)) if m else None
        else:
            m = re.search(r'\[gen\]\s+([\d.]+)\s+tok/s', text)
            val = float(m.group(1)) if m else None
        if val is not None and val > best_val:
            best_val  = val
            best_r    = r
            best_text = text
    return best_r, best_text

SUITES = ["bitnet_pz_bf16", "bitnet_pz_int4", "bitnet_msft", "smollm2_pz", "smollm2_llama"]
HARDWARE = "i5-11300H @ 3.10GHz · 4c/8t · 16GB DDR4 · AVX-512 VNNI"
MAX_TOKENS = 500
RUNS = 3
THREADS = list(range(1, 9))
LABELS = {}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", help="sweep directory (default: the 2026-06-21 sweep)")
    ap.add_argument("--hardware", help="hardware line shown on every screenshot")
    ap.add_argument("--suite", action="append", default=[], help="NAME=Label (repeatable)")
    args = ap.parse_args()
    if args.sweep:
        SWEEP = args.sweep
        SS_DIR = os.path.join(SWEEP, "screenshots")
        meta_path = os.path.join(SWEEP, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            MAX_TOKENS = meta.get("max_tokens", MAX_TOKENS)
            RUNS = meta.get("runs", RUNS)
            THREADS = meta.get("threads", THREADS)
            SUITES = list(meta.get("suites", {}).keys())
    if args.hardware:
        HARDWARE = args.hardware
    for item in args.suite:
        k, _, v = item.partition("=")
        LABELS[k] = v
        if k not in SUITES:
            SUITES.append(k)
    os.makedirs(SS_DIR, exist_ok=True)
    total = 0
    for suite_name in SUITES:
        suite_dir = f"{SWEEP}/{suite_name}"
        if not os.path.isdir(suite_dir):
            print(f"Skipping {suite_name} — no data")
            continue
        print(f"\n=== {suite_name} ===")
        for t in THREADS:
            best_r, text = find_best_run(suite_name, t)
            if text is None:
                print(f"  t={t}: no data")
                continue
            out_path = f"{SS_DIR}/{suite_name}_t{t}.png"
            make_screenshot(suite_name, t, text, out_path, run_num=f"best of {RUNS} (run {best_r})")
            total += 1

    print(f"\nDone — {total} screenshots saved to {SS_DIR}/")

#!/usr/bin/env python3
"""
bench_sweep_seq.py — strictly sequential thread sweep (Addendum AJ methodology).

One process at a time, best of N runs per thread count, fixed cooldowns, same
prompt for every run. Each run is saved verbatim with DATE/CMD/PROMPT headers
(the format tools/make_sweep_screenshots.py renders), plus wall time, max RSS
and CPU% measured with os.wait4 (no /usr/bin/time needed).

Usage:
  python3 tools/bench_sweep_seq.py --out benchmark_results/sweep_YYYY-MM-DD \\
      --suite smollm2_pz_bf16:models/SmolLM2-135M-Instruct-f16.gguf:--classifier,bf16 \\
      [--suite ...] [--threads 1-4] [--runs 3] [--tokens 500] [--cooldown 10]

A suite is NAME:MODEL[:extra,args,comma,separated].
"""
import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
import time

PROMPT = ("Explain what machine learning is and how neural networks work. Discuss supervised "
          "learning, unsupervised learning, and reinforcement learning with real-world examples of each.")
ENGINE = "./adaptive_ai_engine"


def parse_threads(spec):
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def run_once(cmd, log_path):
    """Run cmd, capture stdout+stderr, return (text, wall_s, maxrss_mb, cpu_pct)."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.read()
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t0
    cpu = (ru.ru_utime + ru.ru_stime) / wall * 100.0 if wall > 0 else 0.0
    return out.decode("utf-8", "replace"), wall, ru.ru_maxrss / 1024.0, cpu, proc.returncode


def gen_toks(text):
    m = re.search(r"\[gen\]\s+([\d.]+)\s+tok/s\s+\((\d+) tokens\)", text)
    return (float(m.group(1)), int(m.group(2))) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", action="append", required=True)
    ap.add_argument("--threads", default="1-4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=500)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    threads = parse_threads(a.threads)
    summary_path = os.path.join(a.out, "summary.csv")
    new = not os.path.exists(summary_path)
    sf = open(summary_path, "a", newline="")
    sw = csv.writer(sf)
    if new:
        sw.writerow(["suite", "threads", "run", "gen_tok_s", "tokens", "wall_s", "max_rss_mb", "cpu_pct", "exit"])

    meta = {"prompt": a.prompt, "max_tokens": a.tokens, "runs": a.runs, "cooldown_s": a.cooldown,
            "threads": threads, "suites": {}, "started": datetime.datetime.now().astimezone().isoformat()}

    for spec in a.suite:
        parts = spec.split(":", 2)
        name, model = parts[0], parts[1]
        extra = parts[2].split(",") if len(parts) > 2 and parts[2] else []
        meta["suites"][name] = {"model": model, "extra": extra}
        print(f"\n=== {name}  ({model} {' '.join(extra)}) ===", flush=True)
        for t in threads:
            tdir = os.path.join(a.out, name, f"t{t}")
            os.makedirs(tdir, exist_ok=True)
            best = None
            for r in range(1, a.runs + 1):
                cmd = [ENGINE, "--model", model, "--prompt", a.prompt, "--max-tokens", str(a.tokens),
                       "--temperature", "0", "--threads", str(t), "--no-update-check"] + extra
                text, wall, rss, cpu, rc = run_once(cmd, None)
                tps, ntok = gen_toks(text)
                stamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                with open(os.path.join(tdir, f"run{r}.txt"), "w") as f:
                    f.write(f"=== {name} t={t} run {r} ===\nDATE: {stamp}\nCMD: {' '.join(cmd)}\n"
                            f"PROMPT: {a.prompt}\n{'=' * 60}\n{text}\n"
                            f"PERF: wall={wall:.2f}s max_rss={rss:.0f}MB cpu={cpu:.0f}% exit={rc}\n")
                sw.writerow([name, t, r, tps if tps is not None else "", ntok or "", f"{wall:.2f}",
                             f"{rss:.0f}", f"{cpu:.0f}", rc])
                sf.flush()
                print(f"  t={t} run {r}: {tps} tok/s ({ntok} tok)  wall {wall:.1f}s  rss {rss:.0f}MB  "
                      f"cpu {cpu:.0f}%", flush=True)
                if tps is not None and (best is None or tps > best):
                    best = tps
                time.sleep(a.cooldown)
            print(f"  t={t} BEST {best} tok/s", flush=True)
            time.sleep(a.cooldown)

    meta["finished"] = datetime.datetime.now().astimezone().isoformat()
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    sf.close()


if __name__ == "__main__":
    sys.exit(main())

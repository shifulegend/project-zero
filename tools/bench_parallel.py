#!/usr/bin/env python3
"""
bench_parallel.py — Phase 23 stage 2b: aggregate throughput of the API server
with --parallel N versus one-at-a-time serving, for C concurrent clients.

For each (parallel, clients) pair it starts `pz --server`, sends one warm-up
request, then has C client threads each send one chat request at the same
moment (temperature 0, ignore_eos, fixed max_tokens, distinct prompts), and
reports:
    aggregate tok/s  = all completion tokens / wall time of the whole burst
    per-request tok/s = each request's completion tokens / its own latency
    mean / max latency
With --parallel 1 the server needs --queue-size >= C so requests wait in
order instead of getting 429. Runs are strictly sequential (one server at a
time, nothing else on the CPU). Writes one JSON per config and a summary
(over every config JSON in --out) to --out.

--show-server sends the server's own output (startup banner, hardware
profile, per-request [gen] lines) to this terminal instead of a log file —
used when each run is captured with tools/screenshots/cli/capture.mjs, so the
screenshot is the real session. --summary-only just rebuilds the summary.

Usage:
    python3 tools/bench_parallel.py --model models/smollm2-q8_0.gguf \
        --parallel 1 4 --clients 1 2 4 --max-tokens 128 --threads 4 \
        --out benchmark_results/parallel_2026-10-22/smollm2_135m
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

PROMPTS = [
    "Write a short story about a lighthouse keeper.",
    "Explain how a refrigerator works.",
    "Describe a walk through a forest in autumn.",
    "List some tips for learning a new language and explain each.",
    "Tell me about the history of the bicycle.",
    "Write a letter to a friend about a recent trip.",
    "Explain why the sky is blue.",
    "Describe how bread is made, step by step.",
]


def post(port, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - t0


def wait_ready(port, proc, seconds=300):
    for _ in range(seconds * 4):
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return True
        except Exception:
            time.sleep(0.25)
    return False


def run_config(a, parallel, clients, port):
    cmd = [a.bin, "--model", a.model, "--server", "--port", str(port), "--web-ui", "off",
           "--threads", str(a.threads), "--parallel", str(parallel),
           "--queue-size", str(max(clients, 1) * 2), "--prompt-cache", "off"]
    log = None if a.show_server else open(os.path.join(a.out, f"server_p{parallel}_c{clients}.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT if log else None)
    try:
        if not wait_ready(port, proc):
            raise RuntimeError(f"server did not start: {' '.join(cmd)}")
        cfg = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/config").read())
        slots = cfg["features"]["parallel"]["slots"]
        base = {"max_tokens": a.max_tokens, "temperature": 0, "ignore_eos": True}
        post(port, dict(base, messages=[{"role": "user", "content": "Hello"}], max_tokens=8))
        results = [None] * clients
        start = threading.Barrier(clients + 1)

        def client(i):
            body = dict(base, messages=[{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}])
            start.wait()
            data, dt = post(port, body)
            results[i] = (data["usage"]["completion_tokens"], data["usage"]["prompt_tokens"], dt)

        th = [threading.Thread(target=client, args=(i,)) for i in range(clients)]
        for t in th:
            t.start()
        start.wait()
        t0 = time.perf_counter()
        for t in th:
            t.join()
        wall = time.perf_counter() - t0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log:
            log.close()
    toks = sum(r[0] for r in results)
    lat = [r[2] for r in results]
    per_req = [r[0] / r[2] for r in results]
    return {
        "parallel": parallel, "slots": slots, "clients": clients, "completion_tokens": toks,
        "prompt_tokens": sum(r[1] for r in results), "wall_s": wall,
        "aggregate_tok_s": toks / wall, "per_request_tok_s": sum(per_req) / len(per_req),
        "mean_latency_s": sum(lat) / len(lat), "max_latency_s": max(lat),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--bin", default="./pz")
    ap.add_argument("--parallel", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=18600)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show-server", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if not a.summary_only:
        for p in a.parallel:
            for c in a.clients:
                r = run_config(a, p, c, a.port)
                r.update(model=a.model, threads=a.threads, max_tokens=a.max_tokens,
                         date=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
                with open(os.path.join(a.out, f"p{p}_c{c}.json"), "w") as f:
                    json.dump(r, f, indent=2)
                print(f"\n>>> parallel={p} slots={r['slots']} clients={c}: "
                      f"{r['aggregate_tok_s']:.1f} tok/s aggregate, "
                      f"{r['per_request_tok_s']:.1f} tok/s per request, "
                      f"latency mean {r['mean_latency_s']:.1f}s max {r['max_latency_s']:.1f}s "
                      f"({r['completion_tokens']} tokens in {r['wall_s']:.1f}s)", flush=True)
        if a.show_server:
            return

    rows = []
    for name in sorted(os.listdir(a.out)):
        if name.startswith("p") and name.endswith(".json"):
            with open(os.path.join(a.out, name)) as f:
                rows.append(json.load(f))
    rows.sort(key=lambda r: (r["parallel"], r["clients"]))
    base = {r["clients"]: r for r in rows if r["parallel"] == 1}
    lines = [
        f"$ python3 tools/bench_parallel.py --model {a.model} --parallel {' '.join(map(str, a.parallel))} "
        f"--clients {' '.join(map(str, a.clients))} --max-tokens {a.max_tokens} --threads {a.threads}",
        "",
        f"model: {os.path.basename(a.model)}   threads: {a.threads}   max_tokens: {a.max_tokens} (ignore_eos)",
        "",
        f"{'parallel':>8} {'clients':>7} {'aggregate tok/s':>16} {'per-req tok/s':>14} "
        f"{'mean lat s':>10} {'max lat s':>9} {'vs --parallel 1':>16}",
        "-" * 86,
    ]
    for r in rows:
        ref = base.get(r["clients"])
        gain = f"{r['aggregate_tok_s'] / ref['aggregate_tok_s']:.2f}x" if ref and r["parallel"] != 1 else "-"
        lines.append(f"{r['parallel']:>8} {r['clients']:>7} {r['aggregate_tok_s']:>16.1f} "
                     f"{r['per_request_tok_s']:>14.1f} {r['mean_latency_s']:>10.1f} "
                     f"{r['max_latency_s']:>9.1f} {gain:>16}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(text)
    sys.stdout.write("\n" + text)


if __name__ == "__main__":
    main()

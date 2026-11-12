"""Answer equivalence: 2 prompts x stream on/off, greedy (temperature 0, seed 1),
against a BASE and a FIX build. Prints text + usage per case and compares.
Usage: python3 equivalence.py <base-dir> <fix-dir>"""
import http.client, json, subprocess, sys, time, os
PROMPTS = ["The capital of France is", "Write one sentence about the sea."]

def serve(d, port):
    srv = subprocess.Popen([f"{d}/pz", "--model", os.path.abspath(f"{d}/../models/smollm2-q8_0.gguf"),
                            "--server", "--port", str(port), "--threads", "1"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=d)
    for _ in range(120):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=3); c.request("GET", "/health")
            r = c.getresponse(); r.read(); c.close()
            if r.status == 200: break
        except Exception: time.sleep(0.5)
    return srv

def ask(port, prompt, stream):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": 32,
                       "temperature": 0, "seed": 1, "stream": stream,
                       "stream_options": {"include_usage": True}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); raw = r.read().decode(); c.close()
    if not stream:
        d = json.loads(raw)
        return r.status, d["choices"][0]["message"]["content"], d.get("usage")
    text, usage = "", None
    for line in raw.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]": continue
        d = json.loads(line[6:])
        for ch in d.get("choices", []):
            text += (ch.get("delta") or {}).get("content") or ""
        if d.get("usage"): usage = d["usage"]
    return r.status, text, usage

res = {}
for tag, d, port in (("BASE", sys.argv[1], 8701), ("FIX", sys.argv[2], 8702)):
    srv = serve(d, port)
    try:
        for p in PROMPTS:
            for s in (False, True):
                res[(tag, p, s)] = ask(port, p, s)
    finally:
        srv.terminate(); srv.wait()
same = 0
for p in PROMPTS:
    for s in (False, True):
        b, f = res[("BASE", p, s)], res[("FIX", p, s)]
        ok = b == f
        same += ok
        print(f"[{'SAME' if ok else 'DIFF'}] stream={s!s:5} prompt={p!r}")
        print(f"   BASE {b[0]} {b[1]!r} usage={json.dumps(b[2], sort_keys=True)}")
        print(f"   FIX  {f[0]} {f[1]!r} usage={json.dumps(f[2], sort_keys=True)}")
print(f"=== {same}/4 identical (text + usage + status) ===")
sys.exit(0 if same == 4 else 1)

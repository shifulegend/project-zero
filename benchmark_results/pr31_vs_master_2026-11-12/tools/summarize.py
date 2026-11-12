#!/usr/bin/env python3
"""Parses the raw per-run logs (transcripts/N*_r*_C*.err, excluding warm-ups) and prints best-of-5 and median tok/s per config and N."""
import re, glob, os, statistics, hashlib, sys
EV = os.environ.get("EV", os.path.expanduser("~/evidence")); T = f"{EV}/transcripts"
NAMES = {"C1":"master fefa37e","C2":"pr31 default (rowsplit-fused) f26f36e","C3":"pr31 --moe-threading legacy","C4":"pr31base bf66526","C5":"pr31 --moe-threading rowsplit (extra)"}
data = {}
for f in sorted(glob.glob(f"{T}/N*_r*_C*.err")):
    m = re.search(r"N(\d+)_r(\d+)_(C\d)\.err$", f)
    n, r, c = int(m[1]), int(m[2]), m[3]
    g = re.findall(rb"\[gen\] ([\d.]+) tok/s", open(f, "rb").read())
    data.setdefault(n, {}).setdefault(c, {})[r] = float(g[-1]) if g else None
print(os.environ.get("HDR", "tok/s, decode, DeepSeek-V2-Lite-Chat Q4_K_S, 64 tokens, greedy; rounds interleaved C1..C5"))
for n in sorted(data):
    rs = sorted({r for c in data[n] for r in data[n][c]})
    print(f"\nN={n} threads  (best-of-{len(rs)})")
    print(f"{'config':40s} " + " ".join(f"{'r'+str(r):>6s}" for r in rs) + f" {'best':>7s} {'median':>7s}")
    for c in sorted(data[n]):
        v = [data[n][c].get(r) for r in rs]
        ok = [x for x in v if x is not None]
        cells = " ".join(f"{x:6.2f}" if x is not None else "  FAIL" for x in v)
        print(f"{NAMES[c]:40s} {cells} {max(ok):7.2f} {statistics.median(ok):7.2f}")
def gen_text(path):
    t = open(path, "rb").read().replace(b"\r", b"").decode("utf-8", "replace")
    import re as _re; m = _re.search(r"^KV (cache|Strategy):.*$", t, _re.M); a = m.end() + 1
    b = t.find("--- Expert Utilisation Summary", a)
    return t[a:b].strip()
print("\nsha256 (first 12 hex) of the generated text only (between the KV line and expert summary), N=4, per round:")
ref = {}
for c in ["C1","C2","C3","C4","C5"]:
    hs = []
    for r in range(1, 6):
        p = f"{T}/N4_r{r}_{c}.out"
        if os.path.exists(p):
            g = gen_text(p); hs.append(hashlib.sha256(g.encode()).hexdigest()[:12]); ref.setdefault(c, g)
    print(f"{c} {NAMES[c]:40s} {' '.join(hs)}  distinct={len(set(hs))}")
print("\nfirst differing line vs C1 (round 1 text):")
for c in ["C2","C3","C4","C5"]:
    a, b = ref["C1"].splitlines(), ref[c].splitlines()
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else ""; y = b[i] if i < len(b) else ""
        if x != y:
            j = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
            print(f"{c}: text line {i+1}, first differing char {j+1}"); break
    else: print(f"{c}: identical")

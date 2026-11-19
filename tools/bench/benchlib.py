"""
benchlib.py -- shared helpers for the tools/bench scripts.

Every script in this directory is a standalone command (hardware_profile.py, calibration_test.py,
h2h_bench.py, build_compare.py, setup_engines.py). They share only this module:

  * UTC time stamps in every folder and file name, and again inside each screenshot caption;
  * one run folder per invocation: benchmark_results/<kind>_<YYYYMMDD-HHMMSSZ>[_<label>]/
    with command.txt, system_info.txt, meta.json, raw/ (verbatim output of every run) and
    screenshots/ (live-terminal captures, PNG);
  * live-terminal screenshots through tools/screenshots/cli/capture.mjs (a real pty rendered with
    xterm.js -- never an image drawn from a log), with the raw terminal bytes kept next to them;
  * the engine and model registries (engines.json, models.json);
  * an isolated pz environment (own HOME / cache dir) so calibration never touches, or depends on,
    the user's ~/.project-zero.

Nothing here commits, pushes or edits tracked files: results are plain files in the working tree.
"""
import csv
import datetime
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_sweep_seq import PROMPT, gen_toks, prompt_toks, run_once  # noqa: E402,F401  (reused, not copied)

CAPTURE = ROOT / "tools" / "screenshots" / "cli" / "capture.mjs"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r")


# ── time ────────────────────────────────────────────────────────────────────────────────────
def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def utc_stamp(t=None):
    """20261118-043015Z -- sortable, filesystem-safe, used in folder and file names."""
    return (t or utc_now()).strftime("%Y%m%d-%H%M%SZ")


def iso_utc(t=None):
    return (t or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── argument helpers ────────────────────────────────────────────────────────────────────────
def parse_list(spec, allowed, what):
    """'a,b' | 'all' | 'none' -> list, validated against `allowed` (order of `allowed` kept for 'all')."""
    if spec in ("all", "*"):
        return list(allowed)
    if spec in ("none", ""):
        return []
    items = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [i for i in items if i not in allowed]
    if bad:
        raise SystemExit(f"unknown {what}: {', '.join(bad)} (choose from: {', '.join(allowed)}, all, none)")
    return items


def parse_threads(spec):
    """'4' | '1-4' | '1,2,4' -> [ints]."""
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def add_common_args(ap, kind):
    g = ap.add_argument_group("run folder and output")
    g.add_argument("--out", help=f"output folder (default: benchmark_results/{kind}_<UTC stamp>[_<label>])")
    g.add_argument("--label", default="", help="short name appended to the folder, e.g. 'native-vs-portable'")
    g.add_argument("--dry-run", action="store_true", help="print the plan (commands, counts, estimate) and exit")
    g.add_argument("--screenshots", choices=["none", "summary", "all"], default="summary",
                   help="live-terminal screenshots: none, one per engine/model ('summary'), or one per run ('all')")
    g.add_argument("--shot-rows", type=int, default=0, help="terminal rows for screenshots (0 = tool default 70)")
    g = ap.add_argument_group("Project Zero binary")
    g.add_argument("--pz", default="./pz", help="pz binary to test (default ./pz)")
    g.add_argument("--calibration", choices=["isolated", "off", "existing"], default="isolated",
                   help="isolated: own empty HOME/cache, calibrates once like a first-time user (default); "
                        "off: --no-calibrate; existing: use the real ~/.project-zero")
    import gate          # local import: gate.py imports this module
    gate.add_args(ap)
    return ap


# ── run folder ──────────────────────────────────────────────────────────────────────────────
class RunDir:
    def __init__(self, kind, out=None, label="", argv=None):
        self.kind = kind
        self.t0 = utc_now()
        self.stamp = utc_stamp(self.t0)
        name = f"{kind}_{self.stamp}" + (f"_{label}" if label else "")
        self.path = pathlib.Path(out) if out else ROOT / "benchmark_results" / name
        self.raw = self.path / "raw"
        self.shots = self.path / "screenshots"
        self.argv = argv if argv is not None else sys.argv
        self._log = None

    def create(self):
        for d in (self.path, self.raw, self.shots):
            d.mkdir(parents=True, exist_ok=True)
        cmdfile = self.path / "command.txt"          # a resumed run appends its command; the original stays
        with open(cmdfile, "a") as f:
            f.write(shlex.join(["python3"] + [str(a) for a in self.argv]) + "\n")
        partial = self.path / "PARTIAL.md"
        if not (self.path / "meta.json").exists():      # meta.json is written only by finish()
            partial.write_text("# INCOMPLETE RUN\n\nThis folder is a checkpoint of a run that has not finished (or was interrupted). "
                               "Its numbers are partial; do not quote them.\n\nResume with the same command plus "
                               f"`--out {self.rel()} --resume`:\n\n    {shlex.join(['python3'] + [str(a) for a in self.argv])}\n\n"
                               "`progress.json` says how far it got. This file is deleted when the run finishes.\n")
        self._log = open(self.path / "run.log", "a")
        return self

    def close(self):
        if self._log:
            self._log.close()
            self._log = None

    def log(self, msg=""):
        line = f"[{iso_utc()}] {msg}" if msg else ""
        print(msg, flush=True)
        if self._log:
            self._log.write(line + "\n")
            self._log.flush()

    def write(self, rel, text):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def rel(self, p=None):
        try:
            return str((p or self.path).resolve().relative_to(ROOT))
        except ValueError:
            return str(p or self.path)


# ── host / build identity ───────────────────────────────────────────────────────────────────
def _sh(cmd):
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model():
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def host_info():
    flags = ""
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            flags = line.split(":", 1)[1]
            break
    # (/proc/cpuinfo spells VBMI "avx512vbmi" but VNNI/BF16 "avx512_vnni"/"avx512_bf16": match both)
    want = ("avx2", "fma", "avx512f", "avx512bw", "avx512vl", "avx512_?vnni", "avx_vnni", "avx512_?vbmi",
            "avx512_?bf16", "amx_tile", "f16c")
    have = [w.replace("_?", "_") for w in want if re.search(rf"\b{w}\b", flags)]
    mem = ""
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            mem = f"{int(line.split()[1]) / 1048576:.1f} GiB"
            break
    return {"cpu": cpu_model(), "vcpu": os.cpu_count(), "isa": have, "mem_total": mem,
            "kernel": _sh("uname -sr"), "lscpu": _sh("lscpu"), "loadavg": os.getloadavg()}


def git_info():
    dirty = bool(_sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    return {"commit": _sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
            "branch": _sh(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"]), "dirty": dirty}


def bin_info(path):
    p = pathlib.Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    ver = _sh([str(p), "--version"]).splitlines()
    return {"path": str(path), "exists": True, "sha256_16": sha, "size": p.stat().st_size,
            "version": ver[0] if ver else ""}


def system_info_text(run, extra=None):
    h, g = host_info(), git_info()
    lines = [f"DATE: {iso_utc(run.t0)}", f"HOST_CPU: {h['cpu']} ({h['vcpu']} vCPU)", f"ISA: {' '.join(h['isa'])}",
             f"MEM: {h['mem_total']}", f"KERNEL: {h['kernel']}", f"LOADAVG_AT_START: {h['loadavg']}",
             f"PROJECT_ZERO_COMMIT: {g['commit']}{' (dirty)' if g['dirty'] else ''}  branch {g['branch']}"]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    lines += ["--- lscpu", h["lscpu"]]
    return "\n".join(lines) + "\n"


# ── pz environment ──────────────────────────────────────────────────────────────────────────
def pz_env(run, calibration, tag=""):
    """Environment for pz runs. 'isolated' gives pz its own empty HOME and cache directory (inside the run
    folder, one per `tag`), so the first run calibrates exactly like a first-time user and nothing leaks in
    from ~/.project-zero. `run` is a RunDir or a path."""
    env = dict(os.environ)
    env["PZ_NO_CONFIG"] = "1"
    env["PZ_NO_UPDATE_CHECK"] = "1"
    if calibration == "isolated":
        home = pathlib.Path(getattr(run, "path", run)) / ("pz_home" + (f"_{tag}" if tag else ""))
        (home / "cache").mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["PZ_CACHE_DIR"] = str(home / "cache")
        env["XDG_CACHE_HOME"] = str(home / "cache")
        env["XDG_CONFIG_HOME"] = str(home / "config")
    return env


def pz_flags(calibration):
    return ["--no-calibrate"] if calibration == "off" else []


# ── screenshots ─────────────────────────────────────────────────────────────────────────────
def _ensure_capture_deps():
    mod = ROOT / "tools" / "screenshots" / "cli" / "node_modules" / "@xterm"
    if not mod.exists():
        subprocess.run(["npm", "--prefix", str(CAPTURE.parent), "install", "--no-audit", "--no-fund"],
                       check=True, capture_output=True)


def strip_ansi(text):
    return ANSI.sub("", text)


def screenshot(run, name, cmd, env, rows=0, timeout_s=600, note=""):
    """Run `cmd` in a real pty, render the exact bytes with xterm.js, save PNG + raw bytes.
    Returns (png_path or None, plain-text of the terminal session). The caption carries the UTC
    date/time, host CPU, commit and command, and the file name carries the UTC stamp."""
    if not shutil.which("node") or not shutil.which("script"):
        run.log(f"  (screenshot skipped for {name}: needs node and script(1))")
        return None, ""
    try:
        _ensure_capture_deps()
    except (OSError, subprocess.SubprocessError) as e:
        run.log(f"  (screenshot skipped for {name}: cannot install capture dependencies: {e})")
        return None, ""
    t = utc_now()
    base = f"{utc_stamp(t)}_{name}"
    png, raw = run.shots / f"{base}.png", run.raw / f"{base}.pty"
    g = git_info()
    caption = f"{iso_utc(t)} UTC | {cpu_model()} | pz {g['commit']} | {shlex.join(cmd)[:200]}" + (f" | {note}" if note else "")
    e = dict(env)
    e["PZ_CAPTURE_RAW_OUT"] = str(raw)
    e["PZ_CAPTURE_TIMEOUT_MS"] = str(int(timeout_s * 1000))
    if rows:
        e["PZ_CAPTURE_ROWS"] = str(rows)
    r = subprocess.run(["node", str(CAPTURE), str(png), caption, "--"] + cmd, env=e, capture_output=True, text=True)
    if r.returncode != 0 or not png.exists():
        run.log(f"  (screenshot failed for {name}: {r.stderr.strip()[:200]})")
        return None, ""
    text = strip_ansi(raw.read_text(errors="replace")) if raw.exists() else ""
    return png, text


# ── registries ──────────────────────────────────────────────────────────────────────────────
def load_engines():
    d = json.loads((HERE / "engines.json").read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_models():
    d = json.loads((HERE / "models.json").read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def default_engine_dir():
    return pathlib.Path(os.environ.get("PZ_BENCH_ENGINE_DIR") or pathlib.Path.home() / "ext")


def engine_binary(key, engine_dir):
    e = load_engines()[key]
    p = pathlib.Path(engine_dir) / e["dir"] / e["binary"]
    return p if p.exists() else None


def model_path(key, models_dir="models"):
    p = pathlib.Path(models_dir) / load_models()[key]["file"]
    return p if p.exists() else None


# ── statistics / tables ─────────────────────────────────────────────────────────────────────
def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0, "best": None, "mean": None, "stdev": None}
    return {"n": len(vals), "best": max(vals), "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def fnum(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def save_run_log(run, relname, header_lines, text, tail_lines=()):
    """Verbatim output of one run with the DATE/HOST_CPU/CMD header used by the published sweeps."""
    body = "\n".join(header_lines) + "\n" + "=" * 60 + "\n" + text + "\n" + "\n".join(tail_lines) + "\n"
    p = run.raw / relname
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def finish(run, meta):
    meta = dict(meta)
    meta["finished"] = iso_utc()
    meta["started"] = iso_utc(run.t0)
    (run.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (run.path / "PARTIAL.md").unlink(missing_ok=True)
    (run.path / "progress.json").unlink(missing_ok=True)
    run.log(f"\nResults: {run.rel()}")
    run.close()


# ── running and measuring ───────────────────────────────────────────────────────────────────
def run_cmd(cmd, env=None, timeout=None):
    """Run cmd (stdout+stderr merged); returns dict(text, wall, rss_mb, cpu_pct, rc). One process at a time."""
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    try:
        out = proc.stdout.read()
        _, status, ru = os.wait4(proc.pid, 0)
    except KeyboardInterrupt:
        proc.kill()
        raise
    rc = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t0
    cpu = (ru.ru_utime + ru.ru_stime) / wall * 100.0 if wall > 0 else 0.0
    return {"text": out.decode("utf-8", "replace"), "wall": wall, "rss_mb": ru.ru_maxrss / 1024.0, "cpu_pct": cpu, "rc": rc}


def pz_cmd(pz, model, tokens, threads, extra=(), fixed_length=True, prompt=PROMPT, flags=()):
    cmd = [str(pz), "--model", str(model), "--prompt", prompt, "--max-tokens", str(tokens), "--temperature", "0",
           "--threads", str(threads), "--no-update-check"] + list(flags) + list(extra)
    if fixed_length:
        cmd += ["--raw-prompt", "--ignore-eos"]
    return cmd


def ref_cmd(binary, runner, model, tokens, threads, extra=(), fixed_length=True, prompt=PROMPT):
    """llama.cpp family. runner 'ref' = llama-completion (-no-cnv), 'cli' = llama-cli (ik_llama.cpp)."""
    cmd = [str(binary), "-m", str(model), "-p", prompt, "-n", str(tokens), "--temp", "0", "-t", str(threads)]
    if runner == "ref":
        cmd.append("-no-cnv")
    cmd += list(extra)
    if fixed_length:
        cmd.append("--ignore-eos")
    return cmd


def ppl_cmd_pz(pz, model, text, ctx, chunks, threads, extra=()):
    """pz perplexity run (llama-perplexity's method, built into pz)."""
    return [str(pz), "--model", str(model), "--perplexity", str(text), "--ppl-ctx", str(ctx), "--ppl-chunks", str(chunks),
            "--threads", str(threads), "--no-update-check"] + list(extra)


def ppl_cmd_ref(binary, model, text, ctx, chunks, threads, extra=()):
    """llama-perplexity of the llama.cpp family."""
    return [str(binary), "-m", str(model), "-f", str(text), "-c", str(ctx), "--chunks", str(chunks), "-t", str(threads)] + list(extra)


def engine_ppl_binary(key, engine_dir):
    spec = load_engines()[key].get("perplexity")
    p = pathlib.Path(engine_dir) / load_engines()[key]["dir"] / spec["binary"] if spec else None
    return p if p and p.exists() else None


# A variant is one way of running a model: {label, engine, cmd, ppl, env[, modified]}.
#   cmd(key, info, path, tokens, threads, fixed_length, prompt=PROMPT) -> argv | None   (generation)
#   ppl(key, info, path, text, ctx, chunks, threads, extra=())        -> argv | None   (gate perplexity)
# None = the variant cannot run that model. Every script builds its variants with these two factories,
# so command construction exists once.
def pz_variant(label, pz, flags=(), env=None):
    def cmd(key, info, path, tokens, threads, fixed, prompt=PROMPT):
        return pz_cmd(pz, path, tokens, threads, flags, fixed, prompt) if "pz" in info["engines"] else None

    def ppl(key, info, path, text, ctx, chunks, threads, extra=()):
        return ppl_cmd_pz(pz, path, text, ctx, chunks, threads, list(flags) + list(extra)) if "pz" in info["engines"] else None
    return {"label": label, "engine": "pz", "cmd": cmd, "ppl": ppl, "env": env}


def ref_variant(label, engine, binary, ppl_binary=None, env=None):
    e = load_engines()[engine]

    def cmd(key, info, path, tokens, threads, fixed, prompt=PROMPT):
        if engine not in info["engines"]:
            return None
        return ref_cmd(binary, e["runner"], path, tokens, threads, info.get("extra", {}).get(engine, []), fixed, prompt)

    def ppl(key, info, path, text, ctx, chunks, threads, extra=()):
        if engine not in info["engines"] or not ppl_binary:
            return None
        return ppl_cmd_ref(ppl_binary, path, text, ctx, chunks, threads, extra)
    return {"label": label, "engine": engine, "cmd": cmd, "ppl": ppl, "env": env, "modified": bool(e.get("modified"))}


def measure(run, suite, threads, rnd, cmd, env=None, prompt=PROMPT):
    """One measured run: verbatim log under raw/<suite>/t<threads>/run<rnd>.txt, parsed numbers returned."""
    r = run_cmd(cmd, env)
    tps, ntok = gen_toks(r["text"])
    header = [f"=== {suite} t={threads} run {rnd} ===", f"DATE: {iso_utc()}", f"HOST_CPU: {cpu_model()} ({os.cpu_count()} vCPU)",
              f"CMD: {shlex.join(cmd)}", f"PROMPT: {prompt}"]
    tail = [f"PERF: wall={r['wall']:.2f}s max_rss={r['rss_mb']:.0f}MB cpu={r['cpu_pct']:.0f}% exit={r['rc']}"]
    pt = prompt_toks(r["text"])
    if pt:
        tail.append(f"PROMPT_TPS: {pt}")
    p = save_run_log(run, f"{suite}/t{threads}/run{rnd}.txt", header, r["text"], tail)
    return {"tps": tps, "tokens": ntok, "wall": r["wall"], "rss_mb": r["rss_mb"], "cpu_pct": r["cpu_pct"], "rc": r["rc"],
            "log": run.rel(p)}


def parse_step_timing(text):
    """TN_STEP_TIMING=1 output -> {step name: total_ms} for steps that ran."""
    d = {}
    for m in re.finditer(r'step=\d+ name="([^"]*)" total_ms=([\d.]+) count=(\d+)', text):
        if int(m.group(3)) > 0:
            d[m.group(1)] = float(m.group(2))
    return d


# ── the measurement matrix (shared by h2h_bench, build_compare, calibration_test) ──────────
SUMMARY_HEADER = ["model", "variant", "threads", "round", "gen_tok_s", "tokens", "wall_s", "max_rss_mb", "cpu_pct", "exit", "log",
                  "gate"]


def load_rows(path):
    rows = []
    if pathlib.Path(path).exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                r["threads"], r["round"] = int(r["threads"]), int(r["round"])
                r["gen_tok_s"] = float(r["gen_tok_s"]) if r["gen_tok_s"] else None
                rows.append(r)
    return rows


def write_progress(run, phase, done, total, last=""):
    """Checkpoint marker: how far the run got (readable by people and by the next session)."""
    (run.path / "progress.json").write_text(json.dumps(
        {"phase": phase, "done": done, "total": total, "last": last, "updated": iso_utc()}, indent=2) + "\n")


def ensure_gate_column(path):
    """A summary.csv written before the gate existed gets an empty `gate` column (so appended rows line up)."""
    rows = read_csv_dicts(path)
    if rows and "gate" not in rows[0]:
        write_csv(path, SUMMARY_HEADER, [[r.get(h, "") for h in SUMMARY_HEADER] for r in rows])


def run_matrix(run, variants, models, threads, rounds, tokens=0, cooldown=5.0, order="interleaved",
               fixed_length=True, resume=False, gate_of=None, skip_of=None):
    """Measure every variant on every model.

    variants: [{'label', 'cmd': f(model_key, model_info, path, tokens, threads, fixed_length) -> argv or None,
                'env': dict or None}]  (cmd returns None when the variant cannot run that model)
    models:   [(key, info, path)]
    order:    'interleaved' (default) -- inside each round every variant runs once, the starting
              variant rotating, so thermal drift and background load hit all variants alike;
              'blocks' -- all rounds of one variant, then the next (the published sweeps' layout).
    gate_of:  optional f(model_key, variant_label) -> verdict string stored in the `gate` column.
    skip_of:  optional f(model_key, variant_label) -> True when that cell must not be timed (it failed the gate).
    Rows are appended to summary.csv as they complete, so an interrupted run can be resumed."""
    path = run.path / "summary.csv"
    if resume and path.exists():
        ensure_gate_column(path)
    done = {(r["model"], r["variant"], r["threads"], r["round"]) for r in load_rows(path)} if resume else set()
    new = not path.exists() or not resume
    f = open(path, "a" if resume else "w", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(SUMMARY_HEADER)
    rows = []
    total = sum(sum(1 for v in variants if v["cmd"](k, i, p, tokens or i.get("tokens", 128), threads[0], fixed_length) is not None
                    and not (skip_of and skip_of(k, v["label"])))
                for k, i, p in models) * len(threads) * rounds
    counter = [len(done)]

    def one(key, info, mpath, v, t, rnd, ntok):
        if (key, v["label"], t, rnd) in done:
            run.log(f"  {key} {v['label']} t={t} run {rnd}: recorded (resume)")
            return
        cmd = v["cmd"](key, info, mpath, ntok, t, fixed_length)
        if cmd is None:
            return
        if skip_of and skip_of(key, v["label"]):
            return                                  # reported once per cell below, not once per run
        res = measure(run, f"{key}__{v['label']}", t, rnd, cmd, v.get("env"))
        w.writerow([key, v["label"], t, rnd, res["tps"] if res["tps"] is not None else "", res["tokens"] or "",
                    f"{res['wall']:.2f}", f"{res['rss_mb']:.0f}", f"{res['cpu_pct']:.0f}", res["rc"], res["log"],
                    gate_of(key, v["label"]) if gate_of else ""])
        f.flush()
        rows.append({"model": key, "variant": v["label"], "threads": t, "round": rnd, "gen_tok_s": res["tps"]})
        flag = "" if res["rc"] == 0 else f"  FAILED (exit {res['rc']})"
        run.log(f"  {key:<20} {v['label']:<26} t={t} run {rnd}: {fnum(res['tps'], 2):>8} tok/s  wall {res['wall']:.1f}s{flag}")
        counter[0] += 1
        write_progress(run, "measuring", counter[0], total, f"{key} {v['label']} t={t} run {rnd}")
        time.sleep(cooldown)

    for key, info, mpath in models:
        ntok = tokens or info.get("tokens", 128)
        usable = [v for v in variants if v["cmd"](key, info, mpath, ntok, threads[0], fixed_length) is not None]
        for v in usable:
            if skip_of and skip_of(key, v["label"]):
                run.log(f"  {key} {v['label']}: not timed (fails the correctness gate)")
        usable = [v for v in usable if not (skip_of and skip_of(key, v["label"]))]
        run.log(f"\n=== {key}: {len(usable)} variants, threads {threads}, {rounds} rounds, {ntok} tokens ===")
        if order == "blocks":
            for v in usable:
                for t in threads:
                    for rnd in range(1, rounds + 1):
                        one(key, info, mpath, v, t, rnd, ntok)
        else:
            for rnd in range(1, rounds + 1):
                for ti, t in enumerate(threads):
                    for k in range(len(usable)):
                        one(key, info, mpath, usable[(k + rnd + ti) % len(usable)], t, rnd, ntok)
    f.close()
    return load_rows(path)


def summarize(rows, reference=None):
    """-> list of dict per (model, variant, threads): n, best, mean, stdev, ratio_to_reference (mean/mean)."""
    groups, verdicts = {}, {}
    for r in rows:
        groups.setdefault((r["model"], r["variant"], r["threads"]), []).append(r["gen_tok_s"])
        verdicts.setdefault((r["model"], r["variant"], r["threads"]), r.get("gate", ""))
    out = []
    for (m, v, t), vals in groups.items():
        out.append({"model": m, "variant": v, "threads": t, "gate": verdicts[(m, v, t)], **stats(vals)})
    if reference:
        ref = {(o["model"], o["threads"]): o["mean"] for o in out if o["variant"] == reference}
        for o in out:
            base = ref.get((o["model"], o["threads"]))
            o["ratio"] = (o["mean"] / base) if (base and o["mean"] is not None) else None
    return sorted(out, key=lambda o: (o["model"], o["threads"], o["variant"]))


def summary_md(summary, with_ratio=False, hide_invalid=False):
    """hide_invalid (gate strict): INVALID cells show no throughput, only the verdict."""
    show_gate = any(o.get("gate") for o in summary)
    header = (["model", "variant", "threads", "runs", "best tok/s", "mean tok/s", "stdev"] + (["mean / reference"] if with_ratio else [])
              + (["gate"] if show_gate else []))
    rows = []
    for o in summary:
        hide = hide_invalid and o.get("gate") == "INVALID"
        na = "n/a"
        r = [o["model"], o["variant"], o["threads"], o["n"], na if hide else fnum(o["best"], 2),
             na if hide else fnum(o["mean"], 2), na if hide else fnum(o["stdev"], 2)]
        if with_ratio:
            r.append(na if hide or o.get("ratio") is None else f"{o['ratio'] * 100:.0f}%")
        if show_gate:
            r.append(o.get("gate") or "-")
        rows.append(r)
    return md_table(header, rows)


def take_screenshots(run, variants, models, shot_threads, tokens, mode, rows_override=0, note_of=None, skip_of=None):
    """One live-terminal screenshot per (model, variant) ('summary') after the measured runs. The run is
    a separate, shorter one (tokens capped) because a pty run prints the live tok/s display; its own
    tok/s is parsed from the captured bytes and returned for the record, not mixed into the statistics."""
    if mode == "none":
        return []
    out = []
    for key, info, mpath in models:
        for v in variants:
            cmd = v["cmd"](key, info, mpath, tokens or info.get("tokens", 128), shot_threads, True)
            if cmd is None or (skip_of and skip_of(key, v["label"])):
                continue
            name = f"{key}__{v['label']}_t{shot_threads}"
            have = sorted(run.shots.glob(f"*_{name}.png"))           # a screenshot already taken (resumed run)
            if have:
                png = have[-1]
                pty = run.raw / (png.stem + ".pty")
                text = strip_ansi(pty.read_text(errors="replace")) if pty.exists() else ""
                run.log(f"  screenshot {name}: exists, kept (resume)")
            else:
                png, text = screenshot(run, name, cmd, v.get("env") or dict(os.environ),
                                       rows=rows_override or 130, timeout_s=1800,
                                       note=note_of(key, v["label"]) if note_of else "")
            tps, ntok = gen_toks(text)
            out.append({"model": key, "variant": v["label"], "threads": shot_threads, "tok_s_in_screenshot_run": tps,
                        "screenshot": run.rel(png) if png else None})
            run.log(f"  screenshot {key} {v['label']}: {run.rel(png) if png else 'FAILED'}  ({fnum(tps, 2)} tok/s in that run)")
    return out

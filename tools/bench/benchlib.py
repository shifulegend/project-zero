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

import argparse
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
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TextIO

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_sweep_seq import PROMPT, gen_toks, prompt_toks, run_once  # noqa: F401  (reused, not copied)
from benchlib_variants import (  # re-exported for the bench scripts
    GenCmd as GenCmd,
    PplCmd as PplCmd,
    Variant as Variant,
    default_engine_dir as default_engine_dir,
    engine_binary as engine_binary,
    engine_ppl_binary as engine_ppl_binary,
    load_engines as load_engines,
    load_models as load_models,
    model_path as model_path,
    ppl_cmd_pz as ppl_cmd_pz,
    ppl_cmd_ref as ppl_cmd_ref,
    pz_cmd as pz_cmd,
    pz_variant as pz_variant,
    ref_cmd as ref_cmd,
    ref_variant as ref_variant,
)

CAPTURE = ROOT / "tools" / "screenshots" / "cli" / "capture.mjs"
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\r")


# ── time ────────────────────────────────────────────────────────────────────────────────────
def utc_now() -> datetime.datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.datetime.now(datetime.timezone.utc)


def utc_stamp(t: datetime.datetime | None = None) -> str:
    """20261118-043015Z -- sortable, filesystem-safe, used in folder and file names."""
    return (t or utc_now()).strftime("%Y%m%d-%H%M%SZ")


def iso_utc(t: datetime.datetime | None = None) -> str:
    """Time (default: now) formatted as ISO 8601 UTC, e.g. 2026-11-19T20:00:00Z."""
    return (t or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── argument helpers ────────────────────────────────────────────────────────────────────────
def parse_list(spec: str, allowed: Sequence[str], what: str) -> list[str]:
    """'a,b' | 'all' | 'none' -> list, validated against `allowed` (order of `allowed` kept for 'all')."""
    if spec in ("all", "*"):
        return list(allowed)
    if spec in ("none", ""):
        return []
    items = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [i for i in items if i not in allowed]
    if bad:
        raise SystemExit(
            f"unknown {what}: {', '.join(bad)} (choose from: {', '.join(allowed)}, all, none)"
        )
    return items


def parse_threads(spec: str | int) -> list[int]:
    """'4' | '1-4' | '1,2,4' -> [ints]."""
    out: list[int] = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def add_common_args(ap: argparse.ArgumentParser, kind: str) -> argparse.ArgumentParser:
    """Adds the flags every benchmark script shares (output folder, label, calibration, ...) to the parser `ap`; `kind` names the run folder."""
    g = ap.add_argument_group("run folder and output")
    g.add_argument(
        "--out", help=f"output folder (default: benchmark_results/{kind}_<UTC stamp>[_<label>])"
    )
    g.add_argument(
        "--label", default="", help="short name appended to the folder, e.g. 'native-vs-portable'"
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (commands, counts, estimate) and exit",
    )
    g.add_argument(
        "--screenshots",
        choices=["none", "summary", "all"],
        default="summary",
        help="live-terminal screenshots: none, one per engine/model ('summary'), or one per run ('all')",
    )
    g.add_argument(
        "--shot-rows",
        type=int,
        default=0,
        help="terminal rows for screenshots (0 = tool default 70)",
    )
    g = ap.add_argument_group("Project Zero binary")
    g.add_argument("--pz", default="./pz", help="pz binary to test (default ./pz)")
    g.add_argument(
        "--calibration",
        choices=["isolated", "off", "existing"],
        default="isolated",
        help="isolated: own empty HOME/cache, calibrates once like a first-time user (default); "
        "off: --no-calibrate; existing: use the real ~/.project-zero",
    )
    import gate  # local import: gate.py imports this module

    gate.add_args(ap)
    return ap


# ── run folder ──────────────────────────────────────────────────────────────────────────────
class RunDir:
    """One benchmark run folder with its raw output, screenshots and run log."""

    def __init__(
        self,
        kind: str,
        out: str | pathlib.Path | None = None,
        label: str = "",
        argv: Sequence[str] | None = None,
    ) -> None:
        self.kind = kind
        self.t0 = utc_now()
        self.stamp = utc_stamp(self.t0)
        name = f"{kind}_{self.stamp}" + (f"_{label}" if label else "")
        self.path = pathlib.Path(out) if out else ROOT / "benchmark_results" / name
        self.raw = self.path / "raw"
        self.shots = self.path / "screenshots"
        self.argv = argv if argv is not None else sys.argv
        self._log: TextIO | None = None

    def create(self) -> "RunDir":
        """Creates the run folder with its raw/ and screenshot folders and records the command line."""
        for d in (self.path, self.raw, self.shots):
            d.mkdir(parents=True, exist_ok=True)
        cmdfile = self.path / "command.txt"  # a resumed run appends its command; the original stays
        with open(cmdfile, "a", encoding="utf-8") as f:
            f.write(shlex.join(["python3"] + [str(a) for a in self.argv]) + "\n")
        partial = self.path / "PARTIAL.md"
        if not (self.path / "meta.json").exists():  # meta.json is written only by finish()
            partial.write_text(
                "# INCOMPLETE RUN\n\nThis folder is a checkpoint of a run that has not finished (or was interrupted). "
                "Its numbers are partial; do not quote them.\n\nResume with the same command plus "
                f"`--out {self.rel()} --resume`:\n\n    {shlex.join(['python3'] + [str(a) for a in self.argv])}\n\n"
                "`progress.json` says how far it got. This file is deleted when the run finishes.\n"
            )
        self._log = open(self.path / "run.log", "a", encoding="utf-8")
        return self

    def close(self) -> None:
        """Closes the run log."""
        if self._log:
            self._log.close()
            self._log = None

    def log(self, msg: str = "") -> None:
        """Prints a message and appends it to the run log with a timestamp."""
        line = f"[{iso_utc()}] {msg}" if msg else ""
        print(msg, flush=True)
        if self._log:
            self._log.write(line + "\n")
            self._log.flush()

    def write(self, rel: str, text: str) -> pathlib.Path:
        """Writes text to a file under the run folder (creating directories) and returns its path."""
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def rel(self, p: pathlib.Path | None = None) -> str:
        """Path relative to the repository root, or as given when it lies outside it."""
        try:
            return str((p or self.path).resolve().relative_to(ROOT))
        except ValueError:
            return str(p or self.path)


# ── host / build identity ───────────────────────────────────────────────────────────────────
def _sh(cmd: str | list[str]) -> str:
    """Output of a short command (empty when it cannot run)."""
    try:
        return subprocess.run(
            cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model() -> str:
    """CPU model name from /proc/cpuinfo, or the processor name / "unknown" when it is not listed."""
    for line in pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def host_info() -> dict[str, Any]:
    """Host description for the run folder: CPU model, vCPU count, memory and the relevant CPU flags."""
    flags = ""
    for line in pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("flags"):
            flags = line.split(":", 1)[1]
            break
    # (/proc/cpuinfo spells VBMI "avx512vbmi" but VNNI/BF16 "avx512_vnni"/"avx512_bf16": match both)
    want = (
        "avx2",
        "fma",
        "avx512f",
        "avx512bw",
        "avx512vl",
        "avx512_?vnni",
        "avx_vnni",
        "avx512_?vbmi",
        "avx512_?bf16",
        "amx_tile",
        "f16c",
    )
    have = [w.replace("_?", "_") for w in want if re.search(rf"\b{w}\b", flags)]
    mem = ""
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal"):
            mem = f"{int(line.split()[1]) / 1048576:.1f} GiB"
            break
    return {
        "cpu": cpu_model(),
        "vcpu": os.cpu_count(),
        "isa": have,
        "mem_total": mem,
        "kernel": _sh("uname -sr"),
        "lscpu": _sh("lscpu"),
        "loadavg": os.getloadavg(),
    }


def git_info() -> dict[str, Any]:
    """Git state of the checkout: short commit, branch and whether tracked files are modified."""
    dirty = bool(_sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    return {
        "commit": _sh(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]),
        "branch": _sh(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": dirty,
    }


def bin_info(path: str | pathlib.Path) -> dict[str, Any]:
    """Facts about a binary under test: path, whether it exists, size and a short SHA-256."""
    p = pathlib.Path(path)
    if not p.exists():
        return {"path": str(path), "exists": False}
    sha = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    ver = _sh([str(p), "--version"]).splitlines()
    return {
        "path": str(path),
        "exists": True,
        "sha256_16": sha,
        "size": p.stat().st_size,
        "version": ver[0] if ver else "",
    }


def system_info_text(run: RunDir, extra: dict[str, Any] | None = None) -> str:
    """Text of the run's system-info file: date, host CPU, git state and any `extra` lines."""
    h, g = host_info(), git_info()
    lines = [
        f"DATE: {iso_utc(run.t0)}",
        f"HOST_CPU: {h['cpu']} ({h['vcpu']} vCPU)",
        f"ISA: {' '.join(h['isa'])}",
        f"MEM: {h['mem_total']}",
        f"KERNEL: {h['kernel']}",
        f"LOADAVG_AT_START: {h['loadavg']}",
        f"PROJECT_ZERO_COMMIT: {g['commit']}{' (dirty)' if g['dirty'] else ''}  branch {g['branch']}",
    ]
    for k, v in (extra or {}).items():
        lines.append(f"{k}: {v}")
    lines += ["--- lscpu", h["lscpu"]]
    return "\n".join(lines) + "\n"


# ── pz environment ──────────────────────────────────────────────────────────────────────────
def pz_env(run: "RunDir | str | pathlib.Path", calibration: str, tag: str = "") -> dict[str, str]:
    """Environment for pz runs. 'isolated' gives pz its own empty HOME and cache directory (inside the run
    folder, one per `tag`), so the first run calibrates exactly like a first-time user and nothing leaks in
    from ~/.project-zero. `run` is a RunDir or a path."""
    env = dict(os.environ)
    env["PZ_NO_CONFIG"] = "1"
    env["PZ_NO_UPDATE_CHECK"] = "1"
    if calibration == "isolated":
        home = (run.path if isinstance(run, RunDir) else pathlib.Path(run)) / (
            "pz_home" + (f"_{tag}" if tag else "")
        )
        (home / "cache").mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["PZ_CACHE_DIR"] = str(home / "cache")
        env["XDG_CACHE_HOME"] = str(home / "cache")
        env["XDG_CONFIG_HOME"] = str(home / "config")
    return env


def pz_flags(calibration: str) -> list[str]:
    """Flags to pass to pz for a calibration mode ("off" adds --no-calibrate)."""
    return ["--no-calibrate"] if calibration == "off" else []


# ── screenshots ─────────────────────────────────────────────────────────────────────────────
def _ensure_capture_deps() -> None:
    """Installs the screenshot tool's npm packages when they are missing."""
    mod = ROOT / "tools" / "screenshots" / "cli" / "node_modules" / "@xterm"
    if not mod.exists():
        subprocess.run(
            ["npm", "--prefix", str(CAPTURE.parent), "install", "--no-audit", "--no-fund"],
            check=True,
            capture_output=True,
        )


def strip_ansi(text: str) -> str:
    """Removes ANSI escape sequences from terminal output."""
    return ANSI.sub("", text)


def screenshot(
    run: RunDir,
    name: str,
    cmd: list[str],
    env: dict[str, str],
    rows: int = 0,
    timeout_s: float = 600,
    note: str = "",
) -> tuple[pathlib.Path | None, str]:
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
    caption = f"{iso_utc(t)} UTC | {cpu_model()} | pz {g['commit']} | {shlex.join(cmd)[:200]}" + (
        f" | {note}" if note else ""
    )
    cap_env = dict(env)
    cap_env["PZ_CAPTURE_RAW_OUT"] = str(raw)
    cap_env["PZ_CAPTURE_TIMEOUT_MS"] = str(int(timeout_s * 1000))
    if rows:
        cap_env["PZ_CAPTURE_ROWS"] = str(rows)
    r = subprocess.run(
        ["node", str(CAPTURE), str(png), caption, "--"] + cmd,
        env=cap_env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0 or not png.exists():
        run.log(f"  (screenshot failed for {name}: {r.stderr.strip()[:200]})")
        return None, ""
    text = strip_ansi(raw.read_text(errors="replace")) if raw.exists() else ""
    return png, text


# ── statistics / tables ─────────────────────────────────────────────────────────────────────
def stats(vals: Iterable[float | None]) -> dict[str, Any]:
    """Summary of the non-None samples: count, best, mean and standard deviation (None values when empty)."""
    nums = [v for v in vals if v is not None]
    if not nums:
        return {"n": 0, "best": None, "mean": None, "stdev": None}
    return {
        "n": len(nums),
        "best": max(nums),
        "mean": statistics.mean(nums),
        "stdev": statistics.stdev(nums) if len(nums) > 1 else 0.0,
    }


def write_csv(
    path: str | pathlib.Path, header: Sequence[str], rows: Iterable[Sequence[Any]]
) -> None:
    """Writes `rows` to a CSV file with `header` as the first line (UTF-8)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_csv_dicts(path: str | pathlib.Path) -> list[dict[str, str]]:
    """Reads a CSV file into a list of dicts keyed by the header."""
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Renders a Markdown table from a header and rows."""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def fnum(v: float | None, nd: int = 1) -> str:
    """Formats a number to `nd` decimals, or "-" for None."""
    return "-" if v is None else f"{v:.{nd}f}"


def save_run_log(
    run: RunDir,
    relname: str,
    header_lines: Sequence[str],
    text: str,
    tail_lines: Sequence[str] = (),
) -> pathlib.Path:
    """Verbatim output of one run with the DATE/HOST_CPU/CMD header used by the published sweeps."""
    body = (
        "\n".join(header_lines)
        + "\n"
        + "=" * 60
        + "\n"
        + text
        + "\n"
        + "\n".join(tail_lines)
        + "\n"
    )
    p = run.raw / relname
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def finish(run: RunDir, meta: dict[str, Any]) -> None:
    """Writes meta.json with the start and finish times, removes the partial-run markers, logs the results folder and closes the run."""
    meta = dict(meta)
    meta["finished"] = iso_utc()
    meta["started"] = iso_utc(run.t0)
    (run.path / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (run.path / "PARTIAL.md").unlink(missing_ok=True)
    (run.path / "progress.json").unlink(missing_ok=True)
    run.log(f"\nResults: {run.rel()}")
    run.close()


# ── running and measuring ───────────────────────────────────────────────────────────────────
def run_cmd(
    cmd: list[str], env: dict[str, str] | None = None, timeout: float | None = None
) -> dict[str, Any]:
    """Run cmd (stdout+stderr merged); returns dict(text, wall, rss_mb, cpu_pct, rc). One process at a time."""
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env
    )
    assert proc.stdout is not None  # stdout=PIPE above
    try:
        out = proc.stdout.read()
        _, status, ru = os.wait4(proc.pid, 0)
    except KeyboardInterrupt:
        proc.kill()
        raise
    rc = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t0
    cpu = (ru.ru_utime + ru.ru_stime) / wall * 100.0 if wall > 0 else 0.0
    return {
        "text": out.decode("utf-8", "replace"),
        "wall": wall,
        "rss_mb": ru.ru_maxrss / 1024.0,
        "cpu_pct": cpu,
        "rc": rc,
    }


def measure(
    run: RunDir,
    suite: str,
    threads: int,
    rnd: int,
    cmd: list[str],
    env: dict[str, str] | None = None,
    prompt: str = PROMPT,
) -> dict[str, Any]:
    """One measured run: verbatim log under raw/<suite>/t<threads>/run<rnd>.txt, parsed numbers returned."""
    r = run_cmd(cmd, env)
    tps, ntok = gen_toks(r["text"])
    header = [
        f"=== {suite} t={threads} run {rnd} ===",
        f"DATE: {iso_utc()}",
        f"HOST_CPU: {cpu_model()} ({os.cpu_count()} vCPU)",
        f"CMD: {shlex.join(cmd)}",
        f"PROMPT: {prompt}",
    ]
    tail = [
        f"PERF: wall={r['wall']:.2f}s max_rss={r['rss_mb']:.0f}MB cpu={r['cpu_pct']:.0f}% exit={r['rc']}"
    ]
    pt = prompt_toks(r["text"])
    if pt:
        tail.append(f"PROMPT_TPS: {pt}")
    p = save_run_log(run, f"{suite}/t{threads}/run{rnd}.txt", header, r["text"], tail)
    return {
        "tps": tps,
        "tokens": ntok,
        "wall": r["wall"],
        "rss_mb": r["rss_mb"],
        "cpu_pct": r["cpu_pct"],
        "rc": r["rc"],
        "log": run.rel(p),
    }


def parse_step_timing(text: str) -> dict[str, float]:
    """TN_STEP_TIMING=1 output -> {step name: total_ms} for steps that ran."""
    d: dict[str, float] = {}
    for m in re.finditer(r'step=\d+ name="([^"]*)" total_ms=([\d.]+) count=(\d+)', text):
        if int(m.group(3)) > 0:
            d[m.group(1)] = float(m.group(2))
    return d


# ── the measurement matrix (shared by h2h_bench, build_compare, calibration_test) ──────────
SUMMARY_HEADER = [
    "model",
    "variant",
    "threads",
    "round",
    "gen_tok_s",
    "tokens",
    "wall_s",
    "max_rss_mb",
    "cpu_pct",
    "exit",
    "log",
    "gate",
]


def load_rows(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Rows of an earlier results CSV (an empty list when the file does not exist), used to resume a run."""
    rows: list[dict[str, Any]] = []
    if pathlib.Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row: dict[str, Any] = dict(r)
                row["threads"], row["round"] = int(r["threads"]), int(r["round"])
                row["gen_tok_s"] = float(r["gen_tok_s"]) if r["gen_tok_s"] else None
                rows.append(row)
    return rows


def write_progress(run: RunDir, phase: str, done: int, total: int, last: str = "") -> None:
    """Checkpoint marker: how far the run got (readable by people and by the next session)."""
    (run.path / "progress.json").write_text(
        json.dumps(
            {"phase": phase, "done": done, "total": total, "last": last, "updated": iso_utc()},
            indent=2,
        )
        + "\n"
    )


def ensure_gate_column(path: str | pathlib.Path) -> None:
    """A summary.csv written before the gate existed gets an empty `gate` column (so appended rows line up)."""
    rows = read_csv_dicts(path)
    if rows and "gate" not in rows[0]:
        write_csv(path, SUMMARY_HEADER, [[r.get(h, "") for h in SUMMARY_HEADER] for r in rows])


def run_matrix(
    run: RunDir,
    variants: Sequence[Variant],
    models: Sequence[tuple[str, dict[str, Any], pathlib.Path]],
    threads: Sequence[int],
    rounds: int,
    tokens: int = 0,
    cooldown: float = 5.0,
    order: str = "interleaved",
    fixed_length: bool = True,
    resume: bool = False,
    gate_of: Callable[[str, str], str] | None = None,
    skip_of: Callable[[str, str], bool] | None = None,
) -> list[dict[str, Any]]:
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
    done = (
        {(r["model"], r["variant"], r["threads"], r["round"]) for r in load_rows(path)}
        if resume
        else set()
    )
    new = not path.exists() or not resume
    f = open(path, "a" if resume else "w", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(SUMMARY_HEADER)
    rows: list[dict[str, Any]] = []
    total = (
        sum(
            sum(
                1
                for v in variants
                if v["cmd"](k, i, p, tokens or i.get("tokens", 128), threads[0], fixed_length)
                is not None
                and not (skip_of and skip_of(k, v["label"]))
            )
            for k, i, p in models
        )
        * len(threads)
        * rounds
    )
    counter = [len(done)]

    def one(
        key: str,
        info: dict[str, Any],
        mpath: pathlib.Path,
        v: Variant,
        t: int,
        rnd: int,
        ntok: int,
    ) -> None:
        """Measures one (model, variant, threads, round) cell and appends its row, unless it is already recorded, has no command or is skipped."""
        if (key, v["label"], t, rnd) in done:
            run.log(f"  {key} {v['label']} t={t} run {rnd}: recorded (resume)")
            return
        cmd = v["cmd"](key, info, mpath, ntok, t, fixed_length)
        if cmd is None:
            return
        if skip_of and skip_of(key, v["label"]):
            return  # reported once per cell below, not once per run
        res = measure(run, f"{key}__{v['label']}", t, rnd, cmd, v.get("env"))
        w.writerow(
            [
                key,
                v["label"],
                t,
                rnd,
                res["tps"] if res["tps"] is not None else "",
                res["tokens"] or "",
                f"{res['wall']:.2f}",
                f"{res['rss_mb']:.0f}",
                f"{res['cpu_pct']:.0f}",
                res["rc"],
                res["log"],
                gate_of(key, v["label"]) if gate_of else "",
            ]
        )
        f.flush()
        rows.append(
            {
                "model": key,
                "variant": v["label"],
                "threads": t,
                "round": rnd,
                "gen_tok_s": res["tps"],
            }
        )
        flag = "" if res["rc"] == 0 else f"  FAILED (exit {res['rc']})"
        run.log(
            f"  {key:<20} {v['label']:<26} t={t} run {rnd}: {fnum(res['tps'], 2):>8} tok/s  wall {res['wall']:.1f}s{flag}"
        )
        counter[0] += 1
        write_progress(run, "measuring", counter[0], total, f"{key} {v['label']} t={t} run {rnd}")
        time.sleep(cooldown)

    for key, info, mpath in models:
        ntok = tokens or info.get("tokens", 128)
        usable = [
            v
            for v in variants
            if v["cmd"](key, info, mpath, ntok, threads[0], fixed_length) is not None
        ]
        for v in usable:
            if skip_of and skip_of(key, v["label"]):
                run.log(f"  {key} {v['label']}: not timed (fails the correctness gate)")
        usable = [v for v in usable if not (skip_of and skip_of(key, v["label"]))]
        run.log(
            f"\n=== {key}: {len(usable)} variants, threads {threads}, {rounds} rounds, {ntok} tokens ==="
        )
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


def summarize(rows: Iterable[dict[str, Any]], reference: str | None = None) -> list[dict[str, Any]]:
    """-> list of dict per (model, variant, threads): n, best, mean, stdev, ratio_to_reference (mean/mean)."""
    groups: dict[tuple[str, str, int], list[float | None]] = {}
    verdicts: dict[tuple[str, str, int], str] = {}
    for r in rows:
        groups.setdefault((r["model"], r["variant"], r["threads"]), []).append(r["gen_tok_s"])
        verdicts.setdefault((r["model"], r["variant"], r["threads"]), r.get("gate", ""))
    out: list[dict[str, Any]] = []
    for (m, v, t), vals in groups.items():
        out.append(
            {"model": m, "variant": v, "threads": t, "gate": verdicts[(m, v, t)], **stats(vals)}
        )
    if reference:
        ref = {(o["model"], o["threads"]): o["mean"] for o in out if o["variant"] == reference}
        for o in out:
            base = ref.get((o["model"], o["threads"]))
            o["ratio"] = (o["mean"] / base) if (base and o["mean"] is not None) else None
    return sorted(out, key=lambda o: (o["model"], o["threads"], o["variant"]))


def summary_md(
    summary: Sequence[dict[str, Any]], with_ratio: bool = False, hide_invalid: bool = False
) -> str:
    """hide_invalid (gate strict): INVALID cells show no throughput, only the verdict."""
    show_gate = any(o.get("gate") for o in summary)
    header = (
        ["model", "variant", "threads", "runs", "best tok/s", "mean tok/s", "stdev"]
        + (["mean / reference"] if with_ratio else [])
        + (["gate"] if show_gate else [])
    )
    rows: list[list[Any]] = []
    for o in summary:
        hide = hide_invalid and o.get("gate") == "INVALID"
        na = "n/a"
        r = [
            o["model"],
            o["variant"],
            o["threads"],
            o["n"],
            na if hide else fnum(o["best"], 2),
            na if hide else fnum(o["mean"], 2),
            na if hide else fnum(o["stdev"], 2),
        ]
        if with_ratio:
            r.append(na if hide or o.get("ratio") is None else f"{o['ratio'] * 100:.0f}%")
        if show_gate:
            r.append(o.get("gate") or "-")
        rows.append(r)
    return md_table(header, rows)


def take_screenshots(
    run: RunDir,
    variants: Sequence[Variant],
    models: Sequence[tuple[str, dict[str, Any], pathlib.Path]],
    shot_threads: int,
    tokens: int,
    mode: str,
    rows_override: int = 0,
    note_of: Callable[[str, str], str] | None = None,
    skip_of: Callable[[str, str], bool] | None = None,
) -> list[dict[str, Any]]:
    """One live-terminal screenshot per (model, variant) ('summary') after the measured runs. The run is
    a separate, shorter one (tokens capped) because a pty run prints the live tok/s display; its own
    tok/s is parsed from the captured bytes and returned for the record, not mixed into the statistics."""
    if mode == "none":
        return []
    out: list[dict[str, Any]] = []
    for key, info, mpath in models:
        for v in variants:
            cmd = v["cmd"](key, info, mpath, tokens or info.get("tokens", 128), shot_threads, True)
            if cmd is None or (skip_of and skip_of(key, v["label"])):
                continue
            name = f"{key}__{v['label']}_t{shot_threads}"
            png: pathlib.Path | None
            have = sorted(
                run.shots.glob(f"*_{name}.png")
            )  # a screenshot already taken (resumed run)
            if have:
                png = have[-1]
                pty = run.raw / (png.stem + ".pty")
                text = strip_ansi(pty.read_text(errors="replace")) if pty.exists() else ""
                run.log(f"  screenshot {name}: exists, kept (resume)")
            else:
                png, text = screenshot(
                    run,
                    name,
                    cmd,
                    v.get("env") or dict(os.environ),
                    rows=rows_override or 130,
                    timeout_s=1800,
                    note=note_of(key, v["label"]) if note_of else "",
                )
            tps, ntok = gen_toks(text)
            out.append(
                {
                    "model": key,
                    "variant": v["label"],
                    "threads": shot_threads,
                    "tok_s_in_screenshot_run": tps,
                    "screenshot": run.rel(png) if png else None,
                }
            )
            run.log(
                f"  screenshot {key} {v['label']}: {run.rel(png) if png else 'FAILED'}  ({fnum(tps, 2)} tok/s in that run)"
            )
    return out

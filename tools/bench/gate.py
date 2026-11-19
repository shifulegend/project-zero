#!/usr/bin/env python3
"""
gate.py -- the correctness gate. One module, used by every benchmark script.

A speed number means something only if the same binary, with the same flags, also computes the model
correctly (MLPerf Inference rule). The gate therefore runs before timing and attaches a verdict to every
(model, variant) result:

  PASS        perplexity within `tolerance_pct` of the reference engine on a fixed text
  WARN        within `warn_factor` x tolerance
  INVALID     outside that, the answer to the smoke prompt is wrong, or the engine failed
  UNVERIFIED  no reference result to compare with
  REF         the reference itself (it still has to pass the smoke check)
  APPROX      a variant that is lossy by design (gate.json `approximate`): the gap is reported, not judged

Checks per (model, variant), both from gate.json (nothing hard-coded here):
  1. smoke:  a short greedy factual prompt must contain the expected answer;
  2. quality: perplexity of the engine on gate_text.txt (the llama-perplexity method every engine exposes,
     including pz --perplexity) against the reference engine named in models.json.

Modes (--gate): off | warn (verdicts shown, nothing excluded) | strict (default: INVALID results are
excluded from charts and summaries, and the script exits non-zero).

Used from a script:
    cfg  = gate.config_from_args(a)
    g    = gate.Gate(cfg, run)
    res  = g.check(variants, models, threads)     # writes gate.json, returns {model/variant: result}
    gate.apply_to_summary(run.path / "summary.csv", res)
Standalone use on an existing run folder: every script has `--gate-only` (with `--out <folder> --resume`).
"""

import hashlib
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402

CONFIG = bl.HERE / "gate.json"
GOOD = ("PASS", "WARN", "REF", "APPROX")  # verdicts whose numbers may be published
FLAG_TO_KEY = {
    "gate": "mode",
    "gate_tolerance": "tolerance_pct",
    "gate_text": "text",
    "gate_chunks": "ppl_chunks",
    "gate_ctx": "ppl_ctx",
}


# ── configuration ───────────────────────────────────────────────────────────────────────────
def load_config(overrides=None):
    """gate.json plus overrides (None values are ignored)."""
    cfg = json.loads(CONFIG.read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    text = pathlib.Path(cfg["text"])
    text = text if text.is_absolute() else bl.ROOT / text
    cfg["text_path"] = str(text)
    cfg["text_sha256"] = hashlib.sha256(text.read_bytes()).hexdigest() if text.exists() else ""
    return cfg


def add_args(ap):
    """The gate flags, shared by every script (called once from benchlib.add_common_args)."""
    g = ap.add_argument_group("correctness gate (defaults in tools/bench/gate.json)")
    g.add_argument(
        "--gate",
        choices=["off", "warn", "strict"],
        help="off | warn | strict (default from gate.json)",
    )
    g.add_argument(
        "--gate-tolerance",
        type=float,
        metavar="PCT",
        help="allowed perplexity gap to the reference, percent",
    )
    g.add_argument("--gate-text", metavar="FILE", help="text for the perplexity check")
    g.add_argument("--gate-chunks", type=int, help="perplexity chunks")
    g.add_argument("--gate-ctx", type=int, help="perplexity context length")
    g.add_argument(
        "--gate-reference",
        metavar="VARIANT",
        help="variant to use as the reference for every model (default: models.json)",
    )
    g.add_argument(
        "--allow-dirty-binary",
        action="store_true",
        help="benchmark a pz binary built from uncommitted state (its version ends in -dirty); the run is marked",
    )
    g.add_argument(
        "--gate-only",
        action="store_true",
        help="run only the gate (and update summary.csv/summary.md of --out FOLDER if they exist), then exit",
    )
    return ap


def is_dirty(version):
    """True for a version string of a build of uncommitted state (git describe --dirty)."""
    return bool(re.search(r"-dirty\s*$", version or ""))


def require_clean(binaries, cfg, allow=False, log=print):
    """Refuse (SystemExit) to time binaries built from uncommitted state, which no one can reproduce.
    binaries: {label: path}. With allow=True it only warns. Returns the labels that were dirty."""
    dirty = [l for l, p in binaries.items() if is_dirty(bl.bin_info(p).get("version", ""))]
    if dirty and cfg.get("require_clean_binary", True):
        msg = (
            f"not reproducible: {', '.join(dirty)} {'is' if len(dirty) == 1 else 'are'} a build of uncommitted state "
            "(version ends in -dirty). Commit or stash, rebuild (make clean && make release), or pass --allow-dirty-binary."
        )
        if not allow:
            raise SystemExit(msg)
        log("WARNING: " + msg + "  (allowed; recorded in meta.json)")
    return dirty


def config_from_args(a):
    cfg = load_config({FLAG_TO_KEY[f]: getattr(a, f, None) for f in FLAG_TO_KEY})
    cfg["reference_override"] = getattr(a, "gate_reference", None)
    return cfg


# ── pure decision logic (unit-tested) ───────────────────────────────────────────────────────
def parse_ppl(text, regex):
    """-> (ppl, err) from the last match of the regex; err parsed from '+/- X' when present."""
    last = None
    for m in re.finditer(regex, text):
        last = m
    if not last:
        return None, None
    err = re.search(r"\+/-\s*([0-9.]+)", text[last.start() : last.start() + 80])
    return float(last.group(1)), (float(err.group(1)) if err else None)


LOG_LINE = re.compile(
    r"^\d+\.\d+\.\d+\.\d+ [IWED] .*$|^(llama|ggml|load|system_info|sampler|generate|main)[_:].*$",
    re.M,
)


def answer_text(text):
    """The model's output without llama.cpp-style log lines and colour codes (what a person would read)."""
    return re.sub(r"\n{3,}", "\n\n", LOG_LINE.sub("", bl.strip_ansi(text))).strip()


def smoke_ok(text, expect, prompt=""):
    """True when the output contains one of the expected strings (expected strings must not be in the prompt)."""
    if any(e in prompt for e in expect):
        raise ValueError("a smoke expectation appears in the prompt itself")
    return any(e in text for e in expect)


def verdict(ppl, ref_ppl, tolerance_pct, warn_factor):
    """-> (verdict, delta_pct). Two-sided: an engine far *better* than the reference is as suspect as one worse."""
    if ppl is None:
        return "INVALID", None
    if ref_ppl is None:
        return "UNVERIFIED", None
    delta = (ppl - ref_ppl) / ref_ppl * 100.0
    if abs(delta) <= tolerance_pct:
        return "PASS", delta
    if abs(delta) <= tolerance_pct * warn_factor:
        return "WARN", delta
    return "INVALID", delta


def judge(results, cfg, reference_of):
    """Fill result['verdict'], ['delta_pct'], ['ref'] in place. reference_of(model, variant) -> 'model/variant' or None."""
    for key, r in results.items():
        ref_key = reference_of(r["model"], r["variant"])
        r["ref"] = ref_key
        if r.get("smoke_ok") is False:
            r["verdict"], r["delta_pct"] = "INVALID", None
        elif ref_key == key:
            r["verdict"], r["delta_pct"] = ("REF" if r.get("ppl") is not None else "INVALID"), 0.0
        else:
            ref = results.get(ref_key) if ref_key else None
            ref_bad = ref is None or ref.get("smoke_ok") is False or ref.get("ppl") is None
            r["ref_ppl"] = None if ref_bad else ref["ppl"]
            r["verdict"], r["delta_pct"] = verdict(
                r.get("ppl"), r["ref_ppl"], cfg["tolerance_pct"], cfg["warn_factor"]
            )
            approx = cfg.get("approximate", {}).get(r["variant"])
            if (
                approx is not None
                and r["verdict"] in ("WARN", "INVALID")
                and r.get("ppl") is not None
            ):
                bound = approx.get("tolerance_pct")  # optional hard bound for a lossy variant
                if bound is None or abs(r["delta_pct"]) <= bound:
                    r["verdict"], r["approx_note"] = "APPROX", approx.get("why", "")
    return results


def should_time(verdict, cfg):
    """False for a cell that fails the gate in strict mode (unless gate.json time_invalid): nothing is timed or
    screenshotted for a wrong computation."""
    return not (
        cfg["mode"] == "strict" and verdict == "INVALID" and not cfg.get("time_invalid", False)
    )


def blocked(res, cfg):
    """True when strict mode must keep this result out of charts and summaries."""
    return cfg["mode"] == "strict" and res and res.get("verdict") == "INVALID"


def exit_status(results, cfg):
    return (
        1
        if cfg["mode"] == "strict" and any(r.get("verdict") == "INVALID" for r in results.values())
        else 0
    )


# ── running the checks ──────────────────────────────────────────────────────────────────────
def run_smoke(v, key, info, mpath, cfg, threads):
    """One smoke run of a variant on a model -> {'smoke_ok', 'smoke_tail'} or None when it cannot run that model."""
    sm = cfg["smoke"]
    cmd = v["cmd"](key, info, mpath, sm["tokens"], threads, True, sm["prompt"])
    if cmd is None:
        return None
    r = bl.run_cmd(cmd, v.get("env"))
    answer = answer_text(r["text"])
    return {
        "smoke_ok": smoke_ok(answer, sm["expect"], sm["prompt"]) and r["rc"] == 0,
        "smoke_tail": answer[-240:],
    }


class Gate:
    def __init__(self, cfg, run, models_dir="models", threads=0, resume=False):
        self.cfg, self.run, self.models_dir, self.threads = cfg, run, models_dir, threads
        self.path = run.path / "gate.json"
        self.results = {}
        if resume and self.path.exists():
            self.results = json.loads(self.path.read_text()).get("results", {})

    @property
    def enabled(self):
        return self.cfg["mode"] != "off"

    def reference_of(self, models):
        by_key = {k: info for k, info, _ in models}
        override = self.cfg.get("reference_override")

        def ref(model, variant):
            if override:
                return f"{model}/{override}"
            spec = (
                by_key.get(model, bl.load_models().get(model, {}))
                .get("gate", {})
                .get("reference", {})
            )
            return f"{spec.get('model', model)}/{spec['variant']}" if spec.get("variant") else None

        return ref

    def _one(self, v, key, info, mpath, threads):
        """Smoke + perplexity for one (model, variant); skips what the variant cannot run."""
        cfg, rkey = self.cfg, f"{key}/{v['label']}"
        if rkey in self.results and self.results[rkey].get("ppl") is not None:
            self.run.log(f"  gate {rkey}: recorded (resume)")
            return
        smoke = run_smoke(v, key, info, mpath, cfg, threads)
        ppl_cmd = v["ppl"](
            key,
            info,
            mpath,
            cfg["text_path"],
            cfg["ppl_ctx"],
            cfg["ppl_chunks"],
            threads,
            info.get("gate", {}).get("extra", {}).get(v["engine"], []),
        )
        if smoke is None or ppl_cmd is None:
            return
        out = {
            "model": key,
            "variant": v["label"],
            "engine": v["engine"],
            "modified": bool(v.get("modified")),
            **smoke,
        }
        p = bl.run_cmd(ppl_cmd, v.get("env"))
        out["ppl"], out["ppl_err"] = parse_ppl(bl.strip_ansi(p["text"]), cfg["ppl_regex"])
        out["ppl_wall_s"] = round(p["wall"], 1)
        out["ppl_rc"] = p["rc"]
        out["log"] = self.run.rel(
            bl.save_run_log(
                self.run,
                f"gate/{key}__{v['label']}.txt",
                [f"=== gate {rkey} ===", f"DATE: {bl.iso_utc()}", "CMD: " + bl.shlex.join(ppl_cmd)],
                p["text"],
            )
        )
        self.results[rkey] = out
        self.run.log(
            f"  gate {rkey:<36} smoke {'ok' if out['smoke_ok'] else 'WRONG'}  ppl {bl.fnum(out['ppl'], 4)}"
        )

    def check(self, variants, models, threads):
        """Run every check, judge, persist gate.json. Returns {'model/variant': result}."""
        if not self.enabled:
            return {}
        t = self.threads or max(threads)
        self.run.log(
            f"\n== correctness gate ({self.cfg['mode']}, tolerance {self.cfg['tolerance_pct']} %, "
            f"text {pathlib.Path(self.cfg['text_path']).name}) =="
        )
        order = sorted(
            variants, key=lambda v: not self._is_reference(v, models)
        )  # references first
        for key, info, mpath in models:
            for v in order:
                self._one(v, key, info, mpath, t)
                self._save(models)
        self._save(models)
        return self.results

    def _is_reference(self, v, models):
        ref = self.reference_of(models)
        return any(ref(k, v["label"]) == f"{k}/{v['label']}" for k, _, _ in models)

    def _save(self, models):
        judge(self.results, self.cfg, self.reference_of(models))
        self.path.write_text(
            json.dumps({"config": self.cfg, "results": self.results}, indent=2) + "\n"
        )

    def summary_for_meta(self):
        return {
            "mode": self.cfg["mode"],
            "tolerance_pct": self.cfg["tolerance_pct"],
            "text_sha256": self.cfg["text_sha256"],
            "verdicts": {k: r["verdict"] for k, r in sorted(self.results.items())},
        }

    def timed(self, model, variant):
        """Should this (model, variant) be timed and screenshotted? Used as the `skip_of` of run_matrix and screenshots."""
        return should_time(self.verdict_of(model, variant), self.cfg)

    def verdict_of(self, model, variant):
        return self.results.get(f"{model}/{variant}", {}).get(
            "verdict", "OFF" if not self.enabled else "UNVERIFIED"
        )


# ── presenting the verdicts ─────────────────────────────────────────────────────────────────
def caption_note(res):
    """One short string for a screenshot caption."""
    if not res:
        return ""
    d = "" if res.get("delta_pct") in (None, 0.0) else f" ({res['delta_pct']:+.2f} % vs reference)"
    return f"gate {res['verdict']}: ppl {bl.fnum(res.get('ppl'), 3)}{d}"


def render_md(results, cfg):
    """Markdown section for summary.md."""
    rows = []
    for key, r in sorted(results.items()):
        rows.append(
            [
                r["model"],
                r["variant"] + (" (modified)" if r.get("modified") else ""),
                bl.fnum(r.get("ppl"), 4),
                bl.fnum(r.get("ref_ppl"), 4) if r.get("ref") != key else "-",
                "-" if r.get("delta_pct") is None else f"{r['delta_pct']:+.2f} %",
                "ok" if r.get("smoke_ok") else "WRONG",
                r["verdict"],
                "yes" if should_time(r["verdict"], cfg) else "no (fails the gate)",
            ]
        )
    return (
        "## Correctness gate\n\n"
        f"Mode {cfg['mode']}; tolerance {cfg['tolerance_pct']} % (WARN up to {cfg['warn_factor']}x); "
        f"text `{pathlib.Path(cfg['text_path']).name}` (sha256 {cfg['text_sha256'][:12]}), "
        f"{cfg['ppl_chunks']} chunks of {cfg['ppl_ctx']}.\n\n"
        + bl.md_table(
            ["model", "variant", "ppl", "reference ppl", "delta", "smoke", "verdict", "timed"], rows
        )
        + "\n"
    )


def finish_standalone(run, g):
    """--gate-only for scripts without their own report: verdicts into summary.csv, plus gate_summary.md."""
    apply_to_summary(run.path / "summary.csv", g.results)
    run.write("gate_summary.md", render_md(g.results, g.cfg))
    run.close()
    return exit_status(g.results, g.cfg)


def apply_to_summary(csv_path, results):
    """Add/refresh the `gate` column of an existing summary.csv."""
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        return
    rows = bl.read_csv_dicts(csv_path)
    header = list(rows[0].keys()) if rows else bl.SUMMARY_HEADER
    if "gate" not in header:
        header.append("gate")
    for r in rows:
        r["gate"] = results.get(f"{r['model']}/{r['variant']}", {}).get("verdict", "")
    bl.write_csv(csv_path, header, [[r.get(h, "") for h in header] for r in rows])

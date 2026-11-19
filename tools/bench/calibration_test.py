#!/usr/bin/env python3
"""
calibration_test.py -- the calibration and SIMD-tier tests, runnable with one command.

Each test is selectable (--tests), every parameter is a flag, and pz always runs in its own isolated
HOME/cache (--calibration isolated, the default), so the user's ~/.project-zero is never read or written.

Tests
  profile      the branded pz hardware profile + first-run calibration, live-terminal screenshot
  sweeps       N forced re-calibrations (--calibrations N, `pz --calibrate`), each captured in a real
               terminal; the sweep table is parsed into calibration_sweeps.csv and the picks counted
               (is "auto" kept every time? how often does a tier replace it, and which?)
  tiers        decode throughput per SIMD tier, interleaved (--tiers default,vnni,avx2,avx512f,scalar,vnni256)
  classifiers  decode throughput per LM-head format (--classifiers default,bf16,int8,int4,auto-fast)
  unit         the C unit/characterisation tests (make test) -- slow, rebuilds, opt-in
Default: profile,sweeps,tiers.

Flags
  --tests LIST|all   --calibrations N (default 3)   --tiers LIST   --classifiers LIST
  --models LIST|all (default smollm2-135m,bitnet-2b where present)   --threads 4   --rounds 3   --tokens 128
  --cooldown S   --order interleaved|blocks   --cc gcc|clang (for 'unit')   --models-dir DIR
  --calibration isolated|off|existing   --screenshots none|summary|all   --dry-run   --label   --out   --pz

Examples
  python3 tools/bench/calibration_test.py --tests sweeps --calibrations 5 --label stability
  python3 tools/bench/calibration_test.py --tests tiers --tiers default,vnni,avx2,scalar --models bitnet-2b --rounds 5
  python3 tools/bench/calibration_test.py --tests all --cc clang
"""

import argparse
import collections
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402
import hardware_profile as hp  # noqa: E402
import gate  # noqa: E402

TESTS = ["profile", "sweeps", "tiers", "classifiers", "unit"]
TIERS = ["default", "vnni", "avx2", "avx512f", "scalar", "vnni256"]
CLASSIFIERS = ["default", "auto", "bf16", "int8", "int4", "auto-fast"]
CAND = re.compile(
    r"^\s*(auto|scalar|avx2|avx512f|vnni|vnni256)\s+([\d.]+)\s+\[\s*([\d.]+)\s*[–-]\s*([\d.]+)\s*\]\s*tok/s"
    r"(?:\s+beat auto in\s+(\d+)/(\d+) rounds)?",
    re.M,
)
THREAD = re.compile(
    r"^\s*T=(\d+)\s+bg:\s*(\d+|n/a)%?.*?([\d.]+)\s+\[\s*([\d.]+)\s*[–-]\s*([\d.]+)\s*\]\s*tok/s",
    re.M,
)
PICK_SIMD = re.compile(r"SIMD\s*:\s*(\w+)\s+\(\s*([\d.]+) tok/s")
PICK_THREADS = re.compile(r"Threads\s*:\s*T=(\d+)")


def parse_sweep(text):
    return {
        "candidates": [m.groups() for m in CAND.finditer(text)],
        "threads": [m.groups() for m in THREAD.finditer(text)],
        "pick_simd": (PICK_SIMD.search(text) or [None, None])[1],
        "pick_threads": (PICK_THREADS.search(text) or [None, None])[1],
    }


def test_sweeps(run, a, env, small):
    rows, picks, tpicks = [], collections.Counter(), collections.Counter()
    for i in range(1, a.calibrations + 1):
        run.log(f"\n-- forced calibration {i}/{a.calibrations}")
        cmd = [
            a.pz,
            "--model",
            str(small),
            "--calibrate",
            "--prompt",
            "Hello",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
            "--threads",
            str(bl.os.cpu_count() or 1),
            "--no-update-check",
        ]
        png, text = bl.screenshot(
            run,
            f"calibration_{i}",
            cmd,
            env,
            rows=a.shot_rows or 100,
            timeout_s=900,
            note=f"calibration {i}/{a.calibrations}",
        )
        if not text:
            r = bl.run_cmd(cmd, env)
            text = bl.strip_ansi(r["text"])
        run.write(f"raw/calibration_{i}.txt", text)
        s = parse_sweep(text)
        for c in s["candidates"]:
            rows.append([i, "simd", c[0], c[1], c[2], c[3], c[4] or "", c[5] or ""])
        for t in s["threads"]:
            rows.append([i, "threads", f"T={t[0]}", t[2], t[3], t[4], "", ""])
        picks[s["pick_simd"] or "?"] += 1
        tpicks[s["pick_threads"] or "?"] += 1
        run.log(f"   picked SIMD {s['pick_simd']}, threads {s['pick_threads']}")
    bl.write_csv(
        run.path / "calibration_sweeps.csv",
        ["calibration", "kind", "candidate", "median_tok_s", "min", "max", "wins", "rounds"],
        rows,
    )
    md = [
        "## Calibration sweeps",
        "",
        f"{a.calibrations} forced calibrations (`pz --calibrate`, isolated cache).",
        "",
        bl.md_table(["SIMD pick", "count"], sorted(picks.items())),
        bl.md_table(["thread pick", "count"], sorted(tpicks.items())),
    ]
    return md, {"simd_picks": dict(picks), "thread_picks": dict(tpicks)}


def variant_set(a, env, kind, labels):
    flag = {"tier": "--simd", "cls": "--classifier"}[kind]
    return [
        bl.pz_variant(
            f"{kind}-{l}",
            a.pz,
            bl.pz_flags(a.calibration) + ([] if l == "default" else [flag, l]),
            env,
        )
        for l in labels
    ]


def test_unit(run, a):
    run.log(
        f"\n-- unit tests: make test CC={a.cc} (rebuilds in the debug variant; ./pz is replaced, run `make release` after)"
    )
    r = subprocess.run(
        ["make", "test", f"CC={a.cc}", f"-j{bl.os.cpu_count() or 2}"],
        cwd=bl.ROOT,
        capture_output=True,
        text=True,
    )
    run.write("raw/unit_tests.txt", r.stdout + r.stderr)
    passed = re.findall(r"=== (\d+)/(\d+) tests passed", r.stdout)
    return [
        "## Unit tests (make test)",
        "",
        f"exit code {r.returncode}; {len(passed)} test programs reported results (full output in raw/unit_tests.txt)",
    ], {"returncode": r.returncode, "programs_reporting": len(passed)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bl.add_common_args(ap, "calibration")
    ap.add_argument("--tests", default="profile,sweeps,tiers")
    ap.add_argument("--calibrations", type=int, default=3)
    ap.add_argument("--tiers", default="default,vnni,avx2,scalar")
    ap.add_argument("--classifiers", default="default,bf16,int8,int4")
    ap.add_argument("--models", default="auto")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--order", choices=["interleaved", "blocks"], default="interleaved")
    ap.add_argument("--cc", default="gcc")
    a = ap.parse_args()

    tests = bl.parse_list(a.tests, TESTS, "test")
    tiers, classifiers = (
        bl.parse_list(a.tiers, TIERS, "tier"),
        bl.parse_list(a.classifiers, CLASSIFIERS, "classifier"),
    )
    threads = bl.parse_threads(a.threads)
    allm = bl.load_models()
    wanted = (
        [k for k in ("smollm2-135m", "bitnet-2b") if bl.model_path(k, a.models_dir)]
        if a.models == "auto"
        else bl.parse_list(a.models, list(allm), "model")
    )
    models = [
        (k, allm[k], bl.model_path(k, a.models_dir))
        for k in wanted
        if bl.model_path(k, a.models_dir) and "pz" in allm[k]["engines"]
    ]
    run = bl.RunDir("calibration", a.out, a.label)
    n_runs = (
        (
            (len(tiers) if "tiers" in tests else 0)
            + (len(classifiers) if "classifiers" in tests else 0)
        )
        * len(models)
        * len(threads)
        * a.rounds
    )
    print(
        f"PLAN calibration_test -> {run.rel()}\n  tests: {', '.join(tests)}; calibration {a.calibration}; screenshots {a.screenshots}"
    )
    print(
        f"  sweeps: {a.calibrations if 'sweeps' in tests else 0} forced calibrations (about 1 min each); "
        f"tiers {tiers if 'tiers' in tests else '-'}; classifiers {classifiers if 'classifiers' in tests else '-'}"
    )
    print(
        f"  models {', '.join(k for k, _, _ in models) or '(none)'}; threads {threads}; {a.rounds} rounds x {a.tokens} tokens -> {n_runs} measured runs"
    )
    if a.dry_run:
        return 0
    small = hp.find_small_model(a.models_dir)
    if small is None:
        raise SystemExit("no model on disk: tools/bench/setup_engines.py --models smollm2-135m")
    gate.require_clean({"pz": a.pz}, gate.config_from_args(a), a.allow_dirty_binary)
    run.create()
    env = bl.pz_env(run, a.calibration)
    run.write(
        "system_info.txt",
        bl.system_info_text(run, {"PZ_BINARY": bl.bin_info(a.pz), "CALIBRATION": a.calibration}),
    )
    md = [
        f"# Calibration tests ({', '.join(tests)})",
        "",
        f"UTC start: {bl.iso_utc(run.t0)}  |  host: {bl.cpu_model()}  |  pz commit: {bl.git_info()['commit']}",
        "",
    ]
    meta = {"kind": "calibration", "tests": tests, "calibration": a.calibration}

    exit_code = 0
    if (
        "profile" in tests
        or ("tiers" in tests or "classifiers" in tests)
        and a.calibration == "isolated"
    ):
        run.log("\n== hardware profile + first-run calibration ==")
        _, _, box = hp.capture_profile(run, a.pz, small, env, a.calibration, rows=a.shot_rows)
        md += ["## Hardware profile", "", "```", box, "```", ""]
    if "sweeps" in tests:
        m, mm = test_sweeps(run, a, env, small)
        md += m
        meta["sweeps"] = mm
    for kind, name, labels in (("tier", "tiers", tiers), ("cls", "classifiers", classifiers)):
        if name in tests and models:
            run.log(f"\n== {name} ==")
            variants = variant_set(a, env, kind, labels)
            sub = bl.RunDir(name, run.path / name)
            sub.create()
            sub.t0 = run.t0
            gcfg = gate.config_from_args(a)
            gcfg["reference_override"] = (
                a.gate_reference or variants[0]["label"]
            )  # every tier/classifier vs the default one
            g = gate.Gate(gcfg, sub, a.models_dir, resume=a.gate_only)
            g.check(variants, models, threads)
            if a.gate_only:
                exit_code |= gate.finish_standalone(sub, g)
                continue
            rows = bl.run_matrix(
                sub,
                variants,
                models,
                threads,
                a.rounds,
                a.tokens,
                a.cooldown,
                a.order,
                True,
                gate_of=g.verdict_of,
                skip_of=lambda k, v: not g.timed(k, v),
            )
            summary = bl.summarize(rows, variants[0]["label"])
            hide = g.cfg["mode"] == "strict"
            md += [
                f"## {name} (reference {variants[0]['label']})",
                "",
                bl.summary_md(summary, with_ratio=True, hide_invalid=hide),
            ]
            if g.results:
                md += ["", gate.render_md(g.results, g.cfg)]
            meta[name] = summary
            meta[f"{name}_gate"] = g.summary_for_meta()
            exit_code |= gate.exit_status(g.results, g.cfg)
            if a.screenshots != "none":
                shots = bl.take_screenshots(
                    sub,
                    variants,
                    models[:1],
                    max(threads),
                    32,
                    a.screenshots,
                    a.shot_rows,
                    note_of=lambda k, v: gate.caption_note(g.results.get(f"{k}/{v}")),
                    skip_of=lambda k, v: not g.timed(k, v),
                )
                meta[f"{name}_screenshots"] = shots
    if "unit" in tests:
        m, mm = test_unit(run, a)
        md += m
        meta["unit"] = mm
    run.write("summary.md", "\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    bl.finish(run, meta)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
h2h_bench.py -- head-to-head decode benchmark: pz, llama.cpp, bitnet.cpp, ik_llama.cpp.

Everything is selectable. Nothing is assumed installed: missing engines and models are reported with
the exact command that fetches them (or fetched for you with --engine-fetch / --model-fetch).

What a run does
  1. prints the plan (what will run, how many runs, rough time); --dry-run stops here;
  2. captures the host and the branded pz hardware profile (live terminal screenshot);
  3. with --calibration isolated (default) lets pz calibrate once in its own empty HOME, like a
     first-time user, and saves that calibration output with the run;
  4. runs the correctness gate before timing (gate.py: smoke prompt + perplexity against the reference engine;
     INVALID results are marked, excluded from charts and summaries in strict mode, and the exit status is non-zero);
  5. measures every selected engine x model x thread count x round, one process at a time,
     interleaved so drift and background load hit all engines alike (--order blocks = one engine at a time);
  6. writes summary.csv / summary.md and raw/<model>__<engine>/t<N>/run<R>.txt (verbatim, with
     DATE/HOST_CPU/CMD/PROMPT headers);
  7. captures live-terminal screenshots (pz banner and hardware profile visible) stamped with UTC date/time.

Selecting what runs
  --engines pz,llamacpp,bitnetcpp,ik | all          (default: pz + every reference engine that is present)
  --models  smollm2-135m,bitnet-2b | all            (default: every non-heavy model that is present)
  --threads 4 | 1-4 | 1,2,4      --rounds 3     --tokens N (default: per model, 500; DeepSeek 200)
  --pz-classifiers default,bf16,int8,int4,auto-fast   (one pz variant per classifier; 'default' = no flag)
  --pz-extra "--kv-type i8 --simd vnni"                (extra pz flags for every pz variant)
  --fixed-length / --chat     fixed-length (default): raw prompt, EOS ignored, exactly --tokens generated
  --order interleaved|blocks  --cooldown S (default 5)   --resume (continue an interrupted run folder)
  --screenshots none|summary|all   --shot-threads N   --shot-tokens N (default 0 = the model's full length)
  --preset published   everything the published sweeps contained: all engines, all non-heavy models, threads 1-4,
                       500 tokens, 3 rounds, 10 s cooldown, pz default + int4, a screenshot per run cell, chart
Getting things (default never: the plan says what is missing)
  --engine-fetch never|missing|always   --model-fetch never|missing|always   --ref pinned|latest|<sha>
  --engine-dir DIR (default $PZ_BENCH_ENGINE_DIR or ~/ext)   --models-dir DIR (default models)

Examples
  python3 tools/bench/h2h_bench.py --dry-run
  python3 tools/bench/h2h_bench.py --preset published --engine-fetch missing --model-fetch missing
  python3 tools/bench/h2h_bench.py --engines pz --models smollm2-135m --threads 4 --rounds 3
  python3 tools/bench/h2h_bench.py --engines all --models all --threads 1-4 --engine-fetch missing --model-fetch missing
  python3 tools/bench/h2h_bench.py --engines pz --models bitnet-2b --pz-classifiers default,int4 --label classifiers
"""

import argparse
import json
import pathlib
import os
import shlex
import sys
from collections.abc import Sequence
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl
import hardware_profile as hp
import setup_engines as se
import chart
import gate

# The bare minimum of a published sweep (2026-10-19 / 10-29): every engine on every model, 1..4 threads,
# 500 tokens, 3 rounds with a 10 s cooldown, pz default plus INT4, a live-terminal screenshot of every run
# cell at the full length, and the throughput-vs-threads chart. Heavy models (DeepSeek) stay opt-in.
PRESETS: dict[str, dict[str, Any]] = {
    "published": {
        "engines": "all",
        "models": "all-light",
        "threads": "1-4",
        "rounds": 3,
        "tokens": 500,
        "cooldown": 10.0,
        "pz_classifiers": "default,int4",
        "screenshots": "all",
    }
}

PZ_CLASSIFIERS = ["default", "auto", "bf16", "int8", "int4", "auto-fast"]


def build_variants(
    a: argparse.Namespace, run: bl.RunDir, env: dict[str, str]
) -> tuple[list[bl.Variant], list[tuple[str, str]]]:
    """Variant list (pz and the reference engines) selected by the options; also returns the ones that are not available."""
    variants: list[bl.Variant] = []
    missing: list[tuple[str, str]] = []
    engines = bl.load_engines()
    wanted = (
        bl.parse_list(a.engines, ["pz"] + list(engines), "engine") if a.engines != "auto" else None
    )
    if wanted is None:
        wanted = ["pz"] + [k for k in engines if bl.engine_binary(k, a.engine_dir)]
    classifiers = bl.parse_list(a.pz_classifiers, PZ_CLASSIFIERS, "classifier")
    extra = shlex.split(a.pz_extra)

    if "pz" in wanted:
        for c in classifiers:
            label = "pz" if c == "default" else f"pz-{c}"
            flags = (
                bl.pz_flags(a.calibration) + ([] if c == "default" else ["--classifier", c]) + extra
            )
            variants.append(bl.pz_variant(label, a.pz, flags, env))
    for k in wanted:
        if k == "pz":
            continue
        binary = (
            se.ensure_engine(k, a.engine_dir, a.engine_fetch, a.ref, a.jobs, a.dry_run)
            if (a.engine_fetch != "never" or bl.engine_binary(k, a.engine_dir))
            else None
        )
        if not binary:
            missing.append(("engine", k))
            continue
        variants.append(bl.ref_variant(k, k, binary, bl.engine_ppl_binary(k, a.engine_dir)))
    return variants, missing


def build_report(
    run: bl.RunDir,
    a: argparse.Namespace,
    g: gate.Gate,
    shots: Sequence[dict[str, Any]],
    ref: str = "pz",
) -> list[dict[str, Any]]:
    """summary.md and the chart from the folder's summary.csv and gate.json. Used by a normal run and by
    --gate-only, so the layout exists once."""
    hide = g.cfg["mode"] == "strict"
    summary = bl.summarize(bl.load_rows(run.path / "summary.csv"), ref)
    md = [
        "# Head-to-head decode benchmark",
        "",
        f"UTC start: {bl.iso_utc(run.t0)}  |  host: {bl.cpu_model()}  |  "
        f"pz commit: {bl.git_info()['commit']}",
        "",
        bl.summary_md(summary, with_ratio=True, hide_invalid=hide),
    ]
    if g.results:
        md += ["", gate.render_md(g.results, g.cfg)]
    if shots:
        md += [
            "",
            "## Screenshots",
            "",
            bl.md_table(
                ["model", "variant", "threads", "screenshot"],
                [[s["model"], s["variant"], s["threads"], s["screenshot"]] for s in shots],
            ),
        ]
    charts = chart.make_chart(run.path, rounds=a.rounds, hide_invalid=hide)
    if charts:
        md += [
            "",
            "## Chart",
            "",
            *(f"![throughput vs threads]({c.name})" for c in charts if c.suffix == ".png"),
        ]
    run.write("summary.md", "\n".join(md) + "\n")
    return summary


def select_models(a: argparse.Namespace) -> tuple[list[gate.Model], list[tuple[str, str]]]:
    """Models selected by the options ("auto" and "all-light" pick the light ones); also returns the missing ones."""
    models: list[gate.Model] = []
    missing: list[tuple[str, str]] = []
    allm = bl.load_models()
    if a.models in ("auto", "all-light"):
        wanted = [k for k, m in allm.items() if not m.get("heavy")]
    else:
        wanted = bl.parse_list(a.models, list(allm), "model")
    for k in wanted:
        p = bl.model_path(k, a.models_dir)
        if not p and a.model_fetch != "never":
            p = se.ensure_model(k, a.models_dir, a.model_fetch, a.engine_dir, dry=a.dry_run)
        if p or (a.dry_run and a.model_fetch != "never"):
            models.append((k, allm[k], p or pathlib.Path(a.models_dir) / allm[k]["file"]))
        elif a.models not in ("auto", "all-light"):
            missing.append(("model", k))
    return models, missing


def estimate_minutes(
    variants: Sequence[bl.Variant],
    models: Sequence[gate.Model],
    threads: Sequence[int],
    rounds: int,
    tokens: int,
    cooldown: float,
    shots: str = "none",
    shot_tokens: int = 0,
) -> float:
    """Rough wall time: ~12 tok/s decode plus ~6 s load per run, the cooldown after each, and one more
    run (no cooldown) per screenshot."""
    secs = 0.0
    for key, info, p in models:
        ntok = tokens or info["tokens"]
        for v in variants:
            if v["cmd"](key, info, p, ntok, threads[0], True) is None:
                continue
            secs += len(threads) * rounds * (ntok / 12.0 + 6 + cooldown)
            if shots != "none":
                secs += (len(threads) if shots == "all" else 1) * ((shot_tokens or ntok) / 12.0 + 8)
    return secs / 60.0


def main() -> int:
    """Command-line entry point for the head-to-head benchmark."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bl.add_common_args(ap, "h2h")
    ap.add_argument("--engines", default="auto")
    ap.add_argument("--models", default="auto")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--order", choices=["interleaved", "blocks"], default="interleaved")
    length_group = ap.add_mutually_exclusive_group()
    length_group.add_argument("--fixed-length", dest="fixed", action="store_true", default=True)
    length_group.add_argument("--chat", dest="fixed", action="store_false")
    ap.add_argument("--pz-classifiers", default="default")
    ap.add_argument("--pz-extra", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shot-threads", type=int, default=0)
    ap.add_argument("--shot-tokens", type=int, default=0)
    ap.add_argument("--engine-fetch", choices=["never", "missing", "always"], default="never")
    ap.add_argument("--model-fetch", choices=["never", "missing", "always"], default="never")
    ap.add_argument("--ref", default="pinned")
    ap.add_argument("--engine-dir", default=str(bl.default_engine_dir()))
    ap.add_argument("--models-dir", default="models")
    ap.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="parallel jobs for building engines only; "
        "measured runs are always one process at a time",
    )
    ap.add_argument(
        "--preset", choices=["published"], help="published: the full published sweep (see PRESETS)"
    )
    a = ap.parse_args()
    if a.preset:
        for k, v in PRESETS[a.preset].items():
            if getattr(a, k) == ap.get_default(k):  # an explicit flag still wins
                setattr(a, k, v)

    threads = bl.parse_threads(a.threads)
    run = bl.RunDir("h2h", a.out, a.label)
    env = bl.pz_env(run, a.calibration) if not a.dry_run else dict(os.environ)
    variants, missing = build_variants(a, run, env)
    models, missing_m = select_models(a)
    missing += missing_m

    est = (
        estimate_minutes(
            variants, models, threads, a.rounds, a.tokens, a.cooldown, a.screenshots, a.shot_tokens
        )
        if models
        else 0
    )
    print(f"PLAN h2h -> {run.rel()}")
    print(f"  variants: {', '.join(v['label'] for v in variants) or '(none)'}")
    print(f"  models:   {', '.join(k for k, _, _ in models) or '(none)'}")
    print(
        f"  threads {threads}, {a.rounds} rounds, order {a.order}, {'fixed-length' if a.fixed else 'chat'}, "
        f"calibration {a.calibration}, screenshots {a.screenshots}"
    )
    n = sum(
        1
        for key, info, p in models
        for v in variants
        if v["cmd"](key, info, p, a.tokens or info["tokens"], threads[0], a.fixed)
    )
    print(
        f"  {n} (model, variant) pairs x {len(threads)} thread counts x {a.rounds} rounds = {n * len(threads) * a.rounds} runs, about {est:.0f} min"
    )
    gcfg = gate.config_from_args(a)
    print(
        f"  gate: {gcfg['mode']}, tolerance {gcfg['tolerance_pct']} %, {gcfg['ppl_chunks']} chunks of {gcfg['ppl_ctx']} "
        f"(reference per model from models.json{', all models vs ' + gcfg['reference_override'] if gcfg.get('reference_override') else ''})"
    )
    if missing:
        print("  MISSING: " + ", ".join(f"{k} {n}" for k, n in missing))
        print(
            "  fetch with: python3 tools/bench/setup_engines.py --engines "
            + (",".join(n for k, n in missing if k == "engine") or "none")
            + " --models "
            + (",".join(n for k, n in missing if k == "model") or "none")
            + "   (or re-run with --engine-fetch missing --model-fetch missing)"
        )
    if a.dry_run:
        for key, info, p in models:
            for v in variants:
                c = v["cmd"](key, info, p, a.tokens or info["tokens"], threads[-1], a.fixed)
                if c:
                    print(f"  {key:<20} {v['label']:<14} {shlex.join(c)}")
        return 0
    if not variants or not models:
        raise SystemExit("nothing to run: see MISSING above")
    dirty = gate.require_clean(
        {"pz": a.pz} if any(v["engine"] == "pz" for v in variants) else {},
        gcfg,
        a.allow_dirty_binary,
    )

    run.create()
    g = gate.Gate(gate.config_from_args(a), run, a.models_dir, resume=a.resume or a.gate_only)
    if not a.gate_only:  # --gate-only never rewrites the record of the timed run
        ids: dict[str, Any] = {"PZ_BINARY": bl.bin_info(a.pz)}
        for k in bl.load_engines():
            b = bl.engine_binary(k, a.engine_dir)
            if b:
                ids[f"ENGINE_{k}"] = f"{se.engine_commit(k, a.engine_dir)} {b}"
        run.write("system_info.txt", bl.system_info_text(run, ids))
        run.write(
            "models.txt",
            "\n".join(
                f"{k}: {p} {p.stat().st_size if p.exists() else '?'} bytes" for k, _, p in models
            )
            + "\n",
        )

    profiled = (
        a.resume and (run.path / "hardware_profile.txt").exists()
    )  # a resumed run keeps its profile and calibration
    if not a.gate_only and not profiled and any(v["label"].startswith("pz") for v in variants):
        small = hp.find_small_model(a.models_dir) or models[0][2]
        run.log("\n== hardware profile + first-run calibration (pz) ==")
        hp.capture_profile(run, a.pz, small, env, a.calibration, rows=a.shot_rows)

    g.check(variants, models, threads)
    if a.gate_only:
        return finish_gate_only(run, a, g)
    bl.run_matrix(
        run,
        variants,
        models,
        threads,
        a.rounds,
        a.tokens,
        a.cooldown,
        a.order,
        a.fixed,
        a.resume,
        gate_of=g.verdict_of,
        skip_of=lambda k, v: not g.timed(k, v),
    )
    ref = variants[0]["label"]
    shots = []
    if a.screenshots != "none":
        run.log("\n== screenshots (live terminal) ==")
        sts = threads if a.screenshots == "all" else [a.shot_threads or max(threads)]
        for t in sts:
            shots += bl.take_screenshots(
                run,
                variants,
                models,
                t,
                a.shot_tokens,
                a.screenshots,
                a.shot_rows,
                note_of=lambda k, v: gate.caption_note(g.results.get(f"{k}/{v}")),
                skip_of=lambda k, v: not g.timed(k, v),
            )
    summary = build_report(run, a, g, shots, ref)
    print("\n" + bl.summary_md(summary, with_ratio=True, hide_invalid=g.cfg["mode"] == "strict"))
    bl.finish(
        run,
        {
            "kind": "h2h",
            "variants": [v["label"] for v in variants],
            "models": [k for k, _, _ in models],
            "threads": threads,
            "rounds": a.rounds,
            "order": a.order,
            "fixed_length": a.fixed,
            "calibration": a.calibration,
            "reference_variant": ref,
            "summary": summary,
            "screenshots": shots,
            "gate": g.summary_for_meta(),
            "dirty_binaries": dirty,
        },
    )
    return gate.exit_status(g.results, g.cfg)


def finish_gate_only(run: bl.RunDir, a: argparse.Namespace, g: gate.Gate) -> int:
    """--gate-only on an existing folder: verdicts into summary.csv, summary.md, the chart and meta.json."""
    gate.apply_to_summary(run.path / "summary.csv", g.results)
    meta_path = run.path / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    build_report(run, a, g, meta.get("screenshots", []), meta.get("reference_variant", "pz"))
    meta["gate"] = g.summary_for_meta()
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    run.close()
    return gate.exit_status(g.results, g.cfg)


if __name__ == "__main__":
    sys.exit(main())

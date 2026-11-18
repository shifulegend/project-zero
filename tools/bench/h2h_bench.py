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
  4. measures every selected engine x model x thread count x round, one process at a time,
     interleaved so drift and background load hit all engines alike (--order blocks = one engine at a time);
  5. writes summary.csv / summary.md and raw/<model>__<engine>/t<N>/run<R>.txt (verbatim, with
     DATE/HOST_CPU/CMD/PROMPT headers);
  6. captures live-terminal screenshots (pz banner and hardware profile visible) stamped with UTC date/time.

Selecting what runs
  --engines pz,llamacpp,bitnetcpp,ik | all          (default: pz + every reference engine that is present)
  --models  smollm2-135m,bitnet-2b | all            (default: every non-heavy model that is present)
  --threads 4 | 1-4 | 1,2,4      --rounds 3     --tokens N (default: per model, 500; DeepSeek 200)
  --pz-classifiers default,bf16,int8,int4,auto-fast   (one pz variant per classifier; 'default' = no flag)
  --pz-extra "--kv-type i8 --simd vnni"                (extra pz flags for every pz variant)
  --fixed-length / --chat     fixed-length (default): raw prompt, EOS ignored, exactly --tokens generated
  --order interleaved|blocks  --cooldown S (default 5)   --resume (continue an interrupted run folder)
  --screenshots none|summary|all   --shot-threads N   --shot-tokens N (default 64)
Getting things (default never: the plan says what is missing)
  --engine-fetch never|missing|always   --model-fetch never|missing|always   --ref pinned|latest|<sha>
  --engine-dir DIR (default $PZ_BENCH_ENGINE_DIR or ~/ext)   --models-dir DIR (default models)

Examples
  python3 tools/bench/h2h_bench.py --dry-run
  python3 tools/bench/h2h_bench.py --engines pz --models smollm2-135m --threads 4 --rounds 3
  python3 tools/bench/h2h_bench.py --engines all --models all --threads 1-4 --engine-fetch missing --model-fetch missing
  python3 tools/bench/h2h_bench.py --engines pz --models bitnet-2b --pz-classifiers default,int4 --label classifiers
"""
import argparse
import pathlib
import shlex
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402
import hardware_profile as hp  # noqa: E402
import setup_engines as se  # noqa: E402

PZ_CLASSIFIERS = ["default", "auto", "bf16", "int8", "int4", "auto-fast"]


def build_variants(a, run, env):
    variants, missing = [], []
    engines = bl.load_engines()
    wanted = bl.parse_list(a.engines, ["pz"] + list(engines), "engine") if a.engines != "auto" else None
    if wanted is None:
        wanted = ["pz"] + [k for k in engines if bl.engine_binary(k, a.engine_dir)]
    classifiers = bl.parse_list(a.pz_classifiers, PZ_CLASSIFIERS, "classifier")
    extra = shlex.split(a.pz_extra)

    if "pz" in wanted:
        for c in classifiers:
            label = "pz" if c == "default" else f"pz-{c}"
            flags = bl.pz_flags(a.calibration) + ([] if c == "default" else ["--classifier", c]) + extra

            def cmd(key, info, path, tokens, threads, fixed, _flags=flags):
                if "pz" not in info["engines"]:
                    return None
                return bl.pz_cmd(a.pz, path, tokens, threads, _flags, fixed)
            variants.append({"label": label, "cmd": cmd, "env": env})
    for k in wanted:
        if k == "pz":
            continue
        binary = se.ensure_engine(k, a.engine_dir, a.engine_fetch, a.ref, a.jobs, a.dry_run) if (
            a.engine_fetch != "never" or bl.engine_binary(k, a.engine_dir)) else None
        if not binary:
            missing.append(("engine", k))
            continue

        def cmd(key, info, path, tokens, threads, fixed, _k=k, _b=binary):
            if _k not in info["engines"]:
                return None
            return bl.ref_cmd(_b, engines[_k]["runner"], path, tokens, threads, info.get("extra", {}).get(_k, []), fixed)
        variants.append({"label": k, "cmd": cmd, "env": None})
    return variants, missing


def select_models(a):
    models, missing = [], []
    allm = bl.load_models()
    if a.models == "auto":
        wanted = [k for k, m in allm.items() if not m.get("heavy")]
    else:
        wanted = bl.parse_list(a.models, list(allm), "model")
    for k in wanted:
        p = bl.model_path(k, a.models_dir)
        if not p and a.model_fetch != "never":
            p = se.ensure_model(k, a.models_dir, a.model_fetch, a.engine_dir, dry=a.dry_run)
        if p or (a.dry_run and a.model_fetch != "never"):
            models.append((k, allm[k], p or pathlib.Path(a.models_dir) / allm[k]["file"]))
        elif a.models != "auto":
            missing.append(("model", k))
    return models, missing


def estimate_minutes(variants, models, threads, rounds, tokens, cooldown):
    secs = 0
    for key, info, p in models:
        ntok = tokens or info["tokens"]
        for v in variants:
            if v["cmd"](key, info, p, ntok, threads[0], True) is None:
                continue
            secs += len(threads) * rounds * (ntok / 12.0 + 6 + cooldown)
    return secs / 60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bl.add_common_args(ap, "h2h")
    ap.add_argument("--engines", default="auto")
    ap.add_argument("--models", default="auto")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--order", choices=["interleaved", "blocks"], default="interleaved")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fixed-length", dest="fixed", action="store_true", default=True)
    g.add_argument("--chat", dest="fixed", action="store_false")
    ap.add_argument("--pz-classifiers", default="default")
    ap.add_argument("--pz-extra", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shot-threads", type=int, default=0)
    ap.add_argument("--shot-tokens", type=int, default=64)
    ap.add_argument("--engine-fetch", choices=["never", "missing", "always"], default="never")
    ap.add_argument("--model-fetch", choices=["never", "missing", "always"], default="never")
    ap.add_argument("--ref", default="pinned")
    ap.add_argument("--engine-dir", default=str(bl.default_engine_dir()))
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--jobs", type=int, default=0)
    a = ap.parse_args()

    threads = bl.parse_threads(a.threads)
    run = bl.RunDir("h2h", a.out, a.label)
    env = bl.pz_env(run, a.calibration) if not a.dry_run else dict(bl.os.environ)
    variants, missing = build_variants(a, run, env)
    models, missing_m = select_models(a)
    missing += missing_m

    est = estimate_minutes(variants, models, threads, a.rounds, a.tokens, a.cooldown) if models else 0
    print(f"PLAN h2h -> {run.rel()}")
    print(f"  variants: {', '.join(v['label'] for v in variants) or '(none)'}")
    print(f"  models:   {', '.join(k for k, _, _ in models) or '(none)'}")
    print(f"  threads {threads}, {a.rounds} rounds, order {a.order}, {'fixed-length' if a.fixed else 'chat'}, "
          f"calibration {a.calibration}, screenshots {a.screenshots}")
    n = sum(1 for key, info, p in models for v in variants if v["cmd"](key, info, p, a.tokens or info["tokens"], threads[0], a.fixed))
    print(f"  {n} (model, variant) pairs x {len(threads)} thread counts x {a.rounds} rounds = {n * len(threads) * a.rounds} runs, about {est:.0f} min")
    if missing:
        print("  MISSING: " + ", ".join(f"{k} {n}" for k, n in missing))
        print("  fetch with: python3 tools/bench/setup_engines.py --engines "
              + (",".join(n for k, n in missing if k == "engine") or "none") + " --models "
              + (",".join(n for k, n in missing if k == "model") or "none")
              + "   (or re-run with --engine-fetch missing --model-fetch missing)")
    if a.dry_run:
        for key, info, p in models:
            for v in variants:
                c = v["cmd"](key, info, p, a.tokens or info["tokens"], threads[-1], a.fixed)
                if c:
                    print(f"  {key:<20} {v['label']:<14} {shlex.join(c)}")
        return 0
    if not variants or not models:
        raise SystemExit("nothing to run: see MISSING above")

    run.create()
    ids = {"PZ_BINARY": bl.bin_info(a.pz)}
    for k in bl.load_engines():
        b = bl.engine_binary(k, a.engine_dir)
        if b:
            ids[f"ENGINE_{k}"] = f"{se.engine_commit(k, a.engine_dir)} {b}"
    run.write("system_info.txt", bl.system_info_text(run, ids))
    run.write("models.txt", "\n".join(f"{k}: {p} {p.stat().st_size if p.exists() else '?'} bytes" for k, _, p in models) + "\n")

    if any(v["label"].startswith("pz") for v in variants):
        small = hp.find_small_model(a.models_dir) or models[0][2]
        run.log("\n== hardware profile + first-run calibration (pz) ==")
        hp.capture_profile(run, a.pz, small, env, a.calibration, rows=a.shot_rows)

    rows = bl.run_matrix(run, variants, models, threads, a.rounds, a.tokens, a.cooldown, a.order, a.fixed, a.resume)
    ref = variants[0]["label"]
    summary = bl.summarize(rows, ref)
    md = ["# Head-to-head decode benchmark", "", f"UTC start: {bl.iso_utc(run.t0)}  |  host: {bl.cpu_model()}  |  "
          f"pz commit: {bl.git_info()['commit']}", "", bl.summary_md(summary, with_ratio=True)]
    shots = []
    if a.screenshots != "none":
        run.log("\n== screenshots (live terminal) ==")
        sts = threads if a.screenshots == "all" else [a.shot_threads or max(threads)]
        for t in sts:
            shots += bl.take_screenshots(run, variants, models, t, a.shot_tokens, a.screenshots, a.shot_rows)
        md += ["", "## Screenshots", "", bl.md_table(["model", "variant", "threads", "screenshot"],
               [[s["model"], s["variant"], s["threads"], s["screenshot"]] for s in shots])]
    run.write("summary.md", "\n".join(md) + "\n")
    print("\n" + bl.summary_md(summary, with_ratio=True))
    bl.finish(run, {"kind": "h2h", "variants": [v["label"] for v in variants], "models": [k for k, _, _ in models],
                    "threads": threads, "rounds": a.rounds, "order": a.order, "fixed_length": a.fixed,
                    "calibration": a.calibration, "reference_variant": ref, "summary": summary, "screenshots": shots})
    return 0


if __name__ == "__main__":
    sys.exit(main())

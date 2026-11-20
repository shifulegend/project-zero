#!/usr/bin/env python3
"""
build_compare.py -- interleaved A/B of several pz binaries (native vs portable, before vs after a change).

This is the comparison that found the portable-binary gap: the builds take turns every round (so drift
and background load hit all of them alike), each with its own isolated calibration cache, and the
result is reported as a ratio against a reference build. Optionally it also records the per-step time
split (TN_STEP_TIMING) and runs tools/check_dist_isa.sh on the portable binaries.

Giving it binaries
  --binaries native=./pz-native,portable=./pz-dist     label=path pairs (at least two)
  --build native,portable                              build them first: `make release` -> native,
                                                       `make dist` -> portable (copies to build/bench_bin/,
                                                       which git ignores; --cc gcc|clang). Leaves the tree
                                                       on the last build: run `make release` afterwards.
Selecting what runs
  --models bitnet-2b,smollm2-135m | all   (default: those two when present)   --threads 4   --rounds 5
  --tokens 128   --reference LABEL (default: first)   --order interleaved|blocks   --cooldown S
  --calibration isolated|off|existing     (each binary gets its OWN isolated cache: the cache is keyed on
                                           the build, so two builds sharing one would recalibrate each other)
  --step-timing / --no-step-timing        per-step time split for the first selected model (default on)
  --check-isa LABELS|auto|none            run tools/check_dist_isa.sh (auto: labels containing portable/dist)
  --screenshots none|summary|all          live-terminal screenshots per binary and model (default summary)
  --models-dir DIR   --resume   --dry-run   --label NAME   --out DIR

Examples
  python3 tools/bench/build_compare.py --build native,portable --rounds 5 --label native-vs-portable
  python3 tools/bench/build_compare.py --binaries before=./pz-old,after=./pz --models bitnet-2b --calibration off
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl
import gate
import hardware_profile as hp


def parse_binaries(spec: str) -> dict[str, str]:
    """Parses "label=path,label=path" into {label: path}."""
    out: dict[str, str] = {}
    for item in [s for s in spec.split(",") if s.strip()]:
        if "=" not in item:
            raise SystemExit(f"--binaries wants label=path pairs, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_binaries(run: bl.RunDir, kinds: list[str], cc: str) -> dict[str, str]:
    """Builds pz (native and/or portable) with the compiler `cc`, one binary per kind, and returns {kind: path}."""
    bins: dict[str, str] = {}
    for kind in kinds:
        target = {"native": "release", "portable": "dist"}[kind]
        run.log(f"building {kind} (make {target} CC={cc}) ...")
        r = subprocess.run(
            ["make", target, f"CC={cc}", f"-j{os.cpu_count() or 2}"],
            cwd=bl.ROOT,
            capture_output=True,
            text=True,
        )
        run.write(f"build_{kind}.log", r.stdout + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"make {target} failed; see {run.rel(run.path / f'build_{kind}.log')}")
        # binaries stay out of the run folder (and out of git): build/ is ignored; checksums are in system_info.txt
        dst = bl.ROOT / "build" / "bench_bin" / f"{run.stamp}_pz-{kind}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bl.ROOT / "pz", dst)
        bins[kind] = str(dst)
    return bins


def step_table(runs_text: dict[str, str], reference: str) -> str:
    """Markdown table of per-step timings of every run against the reference run."""
    steps = {lbl: bl.parse_step_timing(t) for lbl, t in runs_text.items()}
    names = list(steps[reference])
    header = (
        ["step"]
        + [f"{lbl} (ms)" for lbl in steps]
        + [f"{lbl} / {reference}" for lbl in steps if lbl != reference]
    )
    rows: list[list[str]] = []
    for n in names:
        row = [n] + [f"{steps[l].get(n, 0):.0f}" for l in steps]
        for l in steps:
            if l != reference:
                base = steps[reference].get(n)
                row.append(f"{steps[l].get(n, 0) / base:.2f}x" if base else "-")
        rows.append(row)
    return bl.md_table(header, rows)


def main() -> int:
    """Command-line entry point: builds the binaries, runs them with identical settings and writes the comparison."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    bl.add_common_args(ap, "build_compare")
    ap.add_argument("--binaries", default="")
    ap.add_argument("--build", default="")
    ap.add_argument("--cc", default="gcc")
    ap.add_argument("--models", default="auto")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--reference", default="")
    ap.add_argument("--order", choices=["interleaved", "blocks"], default="interleaved")
    ap.add_argument("--cooldown", type=float, default=3.0)
    ap.add_argument("--step-timing", dest="step_timing", action="store_true", default=True)
    ap.add_argument("--no-step-timing", dest="step_timing", action="store_false")
    ap.add_argument("--check-isa", default="auto")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    bins = parse_binaries(a.binaries)
    kinds = bl.parse_list(a.build, ["native", "portable"], "build") if a.build else []
    if not bins and not kinds:
        ap.error("give --binaries label=path,... and/or --build native,portable")
    threads = bl.parse_threads(a.threads)
    allm = bl.load_models()
    wanted = (
        [k for k in ("bitnet-2b", "smollm2-135m") if bl.model_path(k, a.models_dir)]
        if a.models == "auto"
        else bl.parse_list(a.models, list(allm), "model")
    )
    models = [
        (k, allm[k], model_file) for k in wanted if (model_file := bl.model_path(k, a.models_dir))
    ]
    run = bl.RunDir("build_compare", a.out, a.label)
    labels = list(bins) + [k for k in kinds if k not in bins]
    ref = a.reference or labels[0]
    print(
        f"PLAN build_compare -> {run.rel()}\n  builds/binaries: {', '.join(labels)}  (reference {ref})"
    )
    print(
        f"  models: {', '.join(k for k, _, _ in models) or '(none found in ' + a.models_dir + ')'}; threads {threads}; "
        f"{a.rounds} rounds x {a.tokens} tokens; order {a.order}; calibration {a.calibration}"
    )
    print(
        f"  {len(labels) * len(models) * len(threads) * a.rounds} measured runs; step timing {'on' if a.step_timing else 'off'}; screenshots {a.screenshots}"
    )
    if a.dry_run:
        return 0
    if not models:
        raise SystemExit(
            "no models on disk: tools/bench/setup_engines.py --models smollm2-135m,bitnet-2b"
        )
    run.create()
    if kinds:
        bins.update(build_binaries(run, kinds, a.cc))
    if ref not in bins:
        raise SystemExit(f"--reference {ref} is not one of {list(bins)}")
    ids = {f"BINARY_{l}": bl.bin_info(p) for l, p in bins.items()}
    run.write("system_info.txt", bl.system_info_text(run, ids))

    envs = {l: bl.pz_env(run, a.calibration, tag=l) for l in bins}
    small = hp.find_small_model(a.models_dir) or models[0][2]
    run.log("\n== hardware profile (and first-run calibration) per binary ==")
    for l, p in bins.items():
        hp.capture_profile(
            run, p, small, envs[l], a.calibration, rows=a.shot_rows, name=f"hardware_profile_{l}"
        )

    variants = [bl.pz_variant(l, p, bl.pz_flags(a.calibration), envs[l]) for l, p in bins.items()]
    gcfg = gate.config_from_args(a)
    dirty = gate.require_clean(bins, gcfg, a.allow_dirty_binary, run.log)
    gcfg["reference_override"] = (
        a.gate_reference or ref
    )  # every binary is judged against the reference binary
    g = gate.Gate(gcfg, run, a.models_dir, resume=a.resume or a.gate_only)
    g.check(variants, models, threads)
    if a.gate_only:
        return gate.finish_standalone(run, g)
    rows = bl.run_matrix(
        run,
        variants,
        models,
        threads,
        a.rounds,
        a.tokens,
        a.cooldown,
        a.order,
        True,
        a.resume,
        gate_of=g.verdict_of,
        skip_of=g.skipped,
    )
    summary = bl.summarize(rows, ref)
    md = [
        f"# Build comparison ({', '.join(labels)}; reference {ref})",
        "",
        f"UTC start: {bl.iso_utc(run.t0)}  |  host: {bl.cpu_model()}  |  pz commit: {bl.git_info()['commit']}",
        "",
        bl.summary_md(summary, with_ratio=True, hide_invalid=g.cfg["mode"] == "strict"),
    ]
    if g.results:
        md += ["", gate.render_md(g.results, g.cfg)]

    if a.step_timing:
        key, info, path = models[0]
        run.log(f"\n== per-step time split ({key}, TN_STEP_TIMING=1) ==")
        texts = {}
        for l, p in bins.items():
            e = dict(envs[l])
            e["TN_STEP_TIMING"] = "1"
            r = bl.run_cmd(bl.pz_cmd(p, path, 64, threads[-1], bl.pz_flags(a.calibration), True), e)
            texts[l] = r["text"]
            run.write(
                f"steps_{key}_{l}.txt",
                "\n".join(x for x in r["text"].splitlines() if x.startswith("step=")) + "\n",
            )
        md += ["", f"## Per-step time, {key} (64 tokens, total ms)", "", step_table(texts, ref)]

    isa_labels = (
        [l for l in bins if "portable" in l or "dist" in l]
        if a.check_isa == "auto"
        else []
        if a.check_isa == "none"
        else bl.parse_list(a.check_isa, list(bins), "binary")
    )
    isa_results: dict[str, int] = {}
    for l in isa_labels:
        isa_run = subprocess.run(
            ["sh", str(bl.ROOT / "tools" / "check_dist_isa.sh"), bins[l]],
            capture_output=True,
            text=True,
        )
        run.write(f"check_dist_isa_{l}.txt", isa_run.stdout + isa_run.stderr)
        isa_results[l] = isa_run.returncode
        run.log(f"check_dist_isa {l}: {'ok' if isa_run.returncode == 0 else 'FAILED'}")
    if isa_results:
        md += [
            "",
            "## check_dist_isa.sh",
            "",
            bl.md_table(
                ["binary", "result"],
                [[l, "ok" if c == 0 else "FAILED"] for l, c in isa_results.items()],
            ),
        ]

    shots = []
    if a.screenshots != "none":
        run.log("\n== screenshots (live terminal) ==")
        shots = bl.take_screenshots(
            run,
            variants,
            models,
            max(threads),
            32,
            a.screenshots,
            a.shot_rows,
            note_of=g.caption,
            skip_of=g.skipped,
        )
        md += [
            "",
            "## Screenshots",
            "",
            bl.md_table(
                ["model", "binary", "threads", "screenshot"],
                [[s["model"], s["variant"], s["threads"], s["screenshot"]] for s in shots],
            ),
        ]
    run.write("summary.md", "\n".join(md) + "\n")
    print("\n" + bl.summary_md(summary, with_ratio=True, hide_invalid=g.cfg["mode"] == "strict"))
    bl.finish(
        run,
        {
            "kind": "build_compare",
            "gate": g.summary_for_meta(),
            "dirty_binaries": dirty,
            "binaries": {l: bl.bin_info(p) for l, p in bins.items()},
            "reference": ref,
            "models": [k for k, _, _ in models],
            "threads": threads,
            "rounds": a.rounds,
            "tokens": a.tokens,
            "order": a.order,
            "calibration": a.calibration,
            "summary": summary,
            "check_isa": isa_results,
            "screenshots": shots,
        },
    )
    return (
        1
        if (any(c != 0 for c in isa_results.values()) or gate.exit_status(g.results, g.cfg))
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
setup_engines.py -- get the reference engines and the models, only what is asked for, only what is missing.

Nothing is assumed to be installed. Every element is selectable, and every choice is a flag:

  engines   llamacpp, bitnetcpp, ik (ik_llama.cpp)         --engines llamacpp,ik | all | none
  models    smollm2-135m, smollm2-1.7b-q4km, bitnet-2b,    --models bitnet-2b | all | none
            bitnet-2b-iq2bn (converted by ik), deepseek-v2-lite (9.5 GB)
  engine source   --engine-fetch missing|always|never   missing (default): clone+build only what is absent
                                                         always: re-clone and rebuild   never: fail if absent
  model source    --model-fetch  missing|always|never   same, for the .gguf files
  version         --ref pinned|latest|<sha-or-branch>   pinned (default) = the commit of the published
                                                         2026-10-29 sweep (engines.json); latest = upstream HEAD
  where           --engine-dir DIR (default $PZ_BENCH_ENGINE_DIR or ~/ext)   --models-dir DIR (default models)
  other           --jobs N   --verify (sha256 of existing models)   --list (status table)   --dry-run

pz itself is not fetched here: build it with `make release` (native) or `make dist` (portable).

Examples:
  python3 tools/bench/setup_engines.py --list
  python3 tools/bench/setup_engines.py --engines llamacpp --models smollm2-135m --dry-run
  python3 tools/bench/setup_engines.py --engines all --models all --ref latest --engine-fetch always
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl


def sh(cmd, cwd=None, env=None, log=print, dry=False, check=True):
    log("  $ " + bl.shlex.join(cmd) + (f"   (in {cwd})" if cwd else ""))
    if dry:
        return 0
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {bl.shlex.join(cmd)}")
    return r.returncode


def engine_commit(key, engine_dir):
    d = pathlib.Path(engine_dir) / bl.load_engines()[key]["dir"]
    return bl._sh(["git", "-C", str(d), "rev-parse", "--short", "HEAD"]) if d.exists() else ""


def _build(key, e, d, jobs, dry, log, targets=None):
    """Configure (unless only extra targets are wanted) and build; the one place that knows how."""
    env = dict(os.environ)
    if e.get("compiler") == "clang":
        env["CC"], env["CXX"] = "clang", "clang++"
    if targets is None:
        sh(e["configure"], cwd=d, env=env, log=log, dry=dry)
        targets = e["targets"]
    sh(
        ["cmake", "--build", "build", "-j", str(jobs)]
        + sum([["--target", t] for t in targets], []),
        cwd=d,
        env=env,
        log=log,
        dry=dry,
    )


def _record(key, e, d, engine_dir, want, dry):
    if not dry:
        info = {
            "engine": key,
            "commit": engine_commit(key, engine_dir),
            "ref": want,
            "built": bl.iso_utc(),
            "configure": e["configure"],
            "targets": e["targets"],
            "recorded_build": e.get("recorded_build", ""),
            "patches": [x["path"] for x in e.get("patches", [])],
        }
        (d / ".pzbench_build.json").write_text(json.dumps(info, indent=2) + "\n")


def _apply_patches(e, d, log, dry):
    """Apply the engine's recorded patches (idempotent: an already-applied patch is skipped)."""
    for pt in e.get("patches", []):
        target = d / pt.get("apply_in", ".")
        patch = bl.ROOT / pt["path"]
        base = ["patch", f"-p{pt.get('strip', 1)}", "-i", str(patch)]
        if (
            subprocess.run(
                base + ["--dry-run", "-R", "-s"], cwd=target, capture_output=True
            ).returncode
            == 0
            and not dry
        ):
            log(f"  patch already applied: {pt['path']}")
            continue
        log(f"  applying {pt['path']}: {pt.get('why', '')}")
        sh(base, cwd=target, log=log, dry=dry)


def _ensure_perplexity(key, e, d, jobs, dry, log):
    """The gate needs the engine's perplexity tool; build just that target when the engine is already built."""
    spec = e.get("perplexity")
    if spec and not (d / spec["binary"]).exists():
        log(f"{e['label']}: building {spec['target']} (needed by the correctness gate)")
        _build(key, e, d, jobs, dry, log, targets=[spec["target"]])


def ensure_engine(key, engine_dir, fetch="missing", ref="pinned", jobs=0, dry=False, log=print):
    """Returns the binary path (or None when fetch=never and it is absent). Idempotent.
    A variant (engines.json `variant_of`) is a patched copy of another engine's source tree."""
    e = bl.load_engines()[key]
    d = pathlib.Path(engine_dir) / e["dir"]
    binary = d / e["binary"]
    jobs = jobs or os.cpu_count() or 2
    if binary.exists() and fetch != "always":
        log(f"{e['label']}: present ({engine_commit(key, engine_dir)}) at {binary}")
        if fetch != "never":
            _ensure_perplexity(key, e, d, jobs, dry, log)
        return binary
    if fetch == "never":
        log(f"{e['label']}: missing at {binary} and --engine-fetch never")
        return None
    want = e["pin"] if ref == "pinned" else ref
    log(
        f"{e['label']}: {'rebuilding' if binary.exists() else 'building'} at {want if ref != 'latest' else 'upstream HEAD'}"
    )
    if d.exists() and fetch == "always":
        log(f"  removing {d}")
        if not dry:
            shutil.rmtree(d)
    if not dry:
        d.parent.mkdir(parents=True, exist_ok=True)
    if e.get("variant_of"):
        base = ensure_engine(e["variant_of"], engine_dir, "missing", ref, jobs, dry, log)
        src = pathlib.Path(engine_dir) / bl.load_engines()[e["variant_of"]]["dir"]
        if base is None and not dry:
            return None
        if not d.exists():
            log(f"  copying the source tree of {e['variant_of']} (without build directories)")
            if not dry:
                shutil.copytree(
                    src, d, symlinks=True, ignore=shutil.ignore_patterns("build", "build_*")
                )
        _apply_patches(e, d, log, dry)
    else:
        if not d.exists():
            sh(["git", "clone", "--filter=blob:none", e["repo"], str(d)], log=log, dry=dry)
        if ref != "latest":
            sh(["git", "checkout", "--quiet", want], cwd=d, log=log, dry=dry)
        else:
            sh(["git", "pull", "--ff-only"], cwd=d, log=log, dry=dry, check=False)
        if e.get("recursive"):
            sh(
                ["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
                cwd=d,
                log=log,
                dry=dry,
            )
    _build(key, e, d, jobs, dry, log)
    _record(key, e, d, engine_dir, want, dry)
    if not dry and not binary.exists():
        raise SystemExit(f"{e['label']}: build finished but {binary} is missing")
    return binary


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_model(
    key, models_dir="models", fetch="missing", engine_dir=None, verify=False, dry=False, log=print
):
    """Returns the model path (or None). Derived models are produced from their source with ik llama-quantize."""
    m = bl.load_models()[key]
    p = pathlib.Path(models_dir) / m["file"]
    if p.exists() and fetch != "always":
        ok = (m.get("size") is None) or p.stat().st_size == m["size"]
        if verify and m.get("sha256"):
            ok = ok and sha256_file(p) == m["sha256"]
        if ok or "derived_from" in m:
            log(f"{key}: present ({p.stat().st_size / 1e6:.0f} MB) at {p}")
            return p
        log(f"{key}: {p} has the wrong size/checksum; re-fetching")
    if fetch == "never":
        log(f"{key}: missing at {p} and --model-fetch never")
        return None
    if not dry:
        pathlib.Path(models_dir).mkdir(parents=True, exist_ok=True)
    if "derived_from" in m:
        src = ensure_model(
            m["derived_from"],
            models_dir,
            fetch if fetch != "always" else "missing",
            engine_dir,
            dry=dry,
            log=log,
        )
        q = engine_dir and bl.engine_binary("ik", engine_dir)
        if not q:
            log(f"{key}: needs ik_llama.cpp built first (--engines ik)")
            return None
        ik = bl.load_engines()["ik"]
        quant = pathlib.Path(engine_dir) / ik["dir"] / ik["extra_binaries"]["quantize"]
        sh([str(quant), "--allow-requantize", str(src), str(p), "iq2_bn"], log=log, dry=dry)
        return p
    log(f"{key}: downloading {m['size'] / 1e6:.0f} MB from {m['url']}")
    sh(["curl", "-fL", "--retry", "3", "-C", "-", "-o", str(p), m["url"]], log=log, dry=dry)
    if not dry:
        if m.get("size") and p.stat().st_size != m["size"]:
            raise SystemExit(f"{key}: downloaded {p.stat().st_size} bytes, expected {m['size']}")
        log(
            f"{key}: sha256 {sha256_file(p)}  (record it in tools/bench/models.json to verify later runs)"
        )
    return p


def smoke_engines(a, engines):
    """The gate's smoke check at install time, so a broken engine build is rejected before anyone times it."""
    import gate

    cfg = gate.load_config()
    bad = 0
    for k in engines:
        binary = bl.engine_binary(k, a.engine_dir)
        models = sorted(
            (
                (m, i, bl.model_path(m, a.models_dir))
                for m, i in bl.load_models().items()
                if k in i["engines"] and bl.model_path(m, a.models_dir)
            ),
            key=lambda x: x[2].stat().st_size,
        )
        if not binary or not models:
            print(f"smoke {k}: skipped (no binary or no model it supports on disk)")
            continue
        r = gate.run_smoke(bl.ref_variant(k, k, binary), *models[0], cfg, a.jobs or 4)
        ok = bool(r and r["smoke_ok"])
        bad += not ok
        print(
            f"smoke {k} on {models[0][0]}: {'ok' if ok else 'WRONG ANSWER'}"
            + ("" if ok else f"\n    {r['smoke_tail'] if r else ''}")
        )
    return 1 if bad else 0


def status_table(engine_dir, models_dir):
    rows = []
    for k, e in bl.load_engines().items():
        b = bl.engine_binary(k, engine_dir)
        rows.append(
            [
                "engine",
                k,
                "present" if b else "missing",
                engine_commit(k, engine_dir) or f"pin {e['pin']}",
                str(b or ""),
            ]
        )
    for k, m in bl.load_models().items():
        p = bl.model_path(k, models_dir)
        rows.append(
            [
                "model",
                k,
                "present" if p else "missing",
                f"{(p.stat().st_size if p else m.get('size', 0)) / 1e6:.0f} MB",
                str(p or ""),
            ]
        )
    return bl.md_table(["kind", "name", "state", "commit / size", "path"], rows)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    eng, mod = list(bl.load_engines()), list(bl.load_models())
    ap.add_argument(
        "--engines", default="none", help=f"{','.join(eng)} | all | none (default none)"
    )
    ap.add_argument("--models", default="none", help=f"{','.join(mod)} | all | none (default none)")
    ap.add_argument("--engine-fetch", choices=["missing", "always", "never"], default="missing")
    ap.add_argument("--model-fetch", choices=["missing", "always", "never"], default="missing")
    ap.add_argument("--ref", default="pinned", help="pinned | latest | <sha or branch>")
    ap.add_argument("--engine-dir", default=str(bl.default_engine_dir()))
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="after setup, run the correctness gate's smoke check (a factual prompt) for each selected engine "
        "on the smallest present model it supports; exit 1 if an engine answers wrongly",
    )
    ap.add_argument("--list", action="store_true", help="show what is present and exit")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(status_table(a.engine_dir, a.models_dir))
        return 0
    engines = bl.parse_list(a.engines, eng, "engine")
    models = bl.parse_list(a.models, mod, "model")
    if not engines and not models:
        ap.error("nothing selected: give --engines and/or --models (or --list)")
    for k in engines:
        ensure_engine(k, a.engine_dir, a.engine_fetch, a.ref, a.jobs, a.dry_run)
    for k in models:
        ensure_model(k, a.models_dir, a.model_fetch, a.engine_dir, a.verify, a.dry_run)
    print("\n" + status_table(a.engine_dir, a.models_dir))
    return smoke_engines(a, engines) if a.smoke and not a.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())

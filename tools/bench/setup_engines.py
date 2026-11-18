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
import benchlib as bl  # noqa: E402


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


def ensure_engine(key, engine_dir, fetch="missing", ref="pinned", jobs=0, dry=False, log=print):
    """Returns the binary path (or None when fetch=never and it is absent). Idempotent."""
    e = bl.load_engines()[key]
    d = pathlib.Path(engine_dir) / e["dir"]
    binary = d / e["binary"]
    if binary.exists() and fetch != "always":
        log(f"{e['label']}: present ({engine_commit(key, engine_dir)}) at {binary}")
        return binary
    if fetch == "never":
        log(f"{e['label']}: missing at {binary} and --engine-fetch never")
        return None
    want = e["pin"] if ref == "pinned" else ref
    jobs = jobs or os.cpu_count() or 2
    log(f"{e['label']}: {'rebuilding' if binary.exists() else 'building'} at {want if ref != 'latest' else 'upstream HEAD'}")
    if d.exists() and fetch == "always":
        log(f"  removing {d}")
        if not dry:
            shutil.rmtree(d)
    if not dry:
        d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        sh(["git", "clone", "--filter=blob:none", e["repo"], str(d)], log=log, dry=dry)
    if ref != "latest":
        sh(["git", "checkout", "--quiet", want], cwd=d, log=log, dry=dry)
    else:
        sh(["git", "pull", "--ff-only"], cwd=d, log=log, dry=dry, check=False)
    if e.get("recursive"):
        sh(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], cwd=d, log=log, dry=dry)
    env = dict(os.environ)
    if e.get("compiler") == "clang":
        env["CC"], env["CXX"] = "clang", "clang++"
    sh(e["configure"], cwd=d, env=env, log=log, dry=dry)
    tgt = sum([["--target", t] for t in e["targets"]], [])
    sh(["cmake", "--build", "build", "-j", str(jobs)] + tgt, cwd=d, env=env, log=log, dry=dry)
    if not dry:
        info = {"engine": key, "commit": engine_commit(key, engine_dir), "ref": want, "built": bl.iso_utc(),
                "configure": e["configure"], "targets": e["targets"], "recorded_build": e.get("recorded_build", "")}
        (d / ".pzbench_build.json").write_text(json.dumps(info, indent=2) + "\n")
        if not binary.exists():
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


def ensure_model(key, models_dir="models", fetch="missing", engine_dir=None, verify=False, dry=False, log=print):
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
        src = ensure_model(m["derived_from"], models_dir, fetch if fetch != "always" else "missing", engine_dir, dry=dry, log=log)
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
        log(f"{key}: sha256 {sha256_file(p)}  (record it in tools/bench/models.json to verify later runs)")
    return p


def status_table(engine_dir, models_dir):
    rows = []
    for k, e in bl.load_engines().items():
        b = bl.engine_binary(k, engine_dir)
        rows.append(["engine", k, "present" if b else "missing", engine_commit(k, engine_dir) or f"pin {e['pin']}", str(b or "")])
    for k, m in bl.load_models().items():
        p = bl.model_path(k, models_dir)
        rows.append(["model", k, "present" if p else "missing", f"{(p.stat().st_size if p else m.get('size', 0)) / 1e6:.0f} MB", str(p or "")])
    return bl.md_table(["kind", "name", "state", "commit / size", "path"], rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    eng, mod = list(bl.load_engines()), list(bl.load_models())
    ap.add_argument("--engines", default="none", help=f"{','.join(eng)} | all | none (default none)")
    ap.add_argument("--models", default="none", help=f"{','.join(mod)} | all | none (default none)")
    ap.add_argument("--engine-fetch", choices=["missing", "always", "never"], default="missing")
    ap.add_argument("--model-fetch", choices=["missing", "always", "never"], default="missing")
    ap.add_argument("--ref", default="pinned", help="pinned | latest | <sha or branch>")
    ap.add_argument("--engine-dir", default=str(bl.default_engine_dir()))
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())

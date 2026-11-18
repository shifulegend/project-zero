#!/usr/bin/env python3
"""
hardware_profile.py -- capture the host and the pz hardware profile, with the full pz branding.

Runs pz once on the smallest model it can find (startup banner, wordmark, hardware-profile box, model
configuration, a one-token reply) in a real pty, saves a screenshot of that exact terminal session
with the UTC date/time stamped in the file name and the caption, and writes the host details.

Output: benchmark_results/hardware_profile_<UTC stamp>[_label]/
  system_info.txt   CPU, ISA flags, memory, kernel, lscpu, pz commit and binary checksum
  hardware_profile.txt   the profile box exactly as pz printed it
  screenshots/<stamp>_hardware_profile.png   live-terminal screenshot (banner visible)
  raw/<stamp>_hardware_profile.pty           the raw terminal bytes
  meta.json, command.txt, run.log

Examples:
  python3 tools/bench/hardware_profile.py
  python3 tools/bench/hardware_profile.py --pz ./pz-dist --label portable --calibration off
  python3 tools/bench/hardware_profile.py --model models/bitnet-b1.58-2B-4T-i2_s.gguf --dry-run
"""
import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl  # noqa: E402


def find_small_model(models_dir):
    """The smallest known model on disk (the profile does not depend on which one)."""
    best = None
    for key, m in bl.load_models().items():
        p = bl.model_path(key, models_dir)
        if p and (best is None or p.stat().st_size < best.stat().st_size):
            best = p
    if best is None:
        ggufs = sorted(pathlib.Path(models_dir).glob("*.gguf"), key=lambda p: p.stat().st_size)
        best = ggufs[0] if ggufs else None
    return best


def profile_box(text):
    """The 'Hardware Profile' box from a pz startup, or '' when absent."""
    m = re.search(r"┌[─]+┐\s*\n│\s+Hardware Profile.*?└[─]+┘", text, re.S)
    return m.group(0) if m else ""


def capture_profile(run, pz, model, env, calibration, rows=0, name="hardware_profile"):
    cmd = [pz, "--model", str(model), "--prompt", "Hello", "--max-tokens", "1", "--temperature", "0",
           "--threads", str(bl.os.cpu_count() or 1), "--no-update-check"] + bl.pz_flags(calibration)
    run.log(f"capturing the pz hardware profile: {bl.shlex.join(cmd)}")
    png, text = bl.screenshot(run, name, cmd, env, rows=rows or 100, timeout_s=900,
                              note=f"calibration={calibration}")
    box = profile_box(text)
    if box:
        run.write("hardware_profile.txt", box + "\n")
    return png, text, box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    bl.add_common_args(ap, "hardware_profile")
    ap.add_argument("--model", help="model to start pz with (default: the smallest known model in --models-dir)")
    ap.add_argument("--models-dir", default="models")
    a = ap.parse_args()

    model = a.model or find_small_model(a.models_dir)
    run = bl.RunDir("hardware_profile", a.out, a.label)
    if a.dry_run:
        print(f"PLAN hardware_profile -> {run.rel()}\n  pz={a.pz} calibration={a.calibration} model={model}")
        print("  1 pz startup run captured in a real terminal (screenshot + raw bytes), host details, binary checksum")
        return 0
    if model is None:
        raise SystemExit("no model found: pass --model or fetch one with tools/bench/setup_engines.py --models smollm2-135m")
    run.create()
    env = bl.pz_env(run, a.calibration)
    info = {"PZ_BINARY": f"{bl.bin_info(a.pz)}", "CALIBRATION": a.calibration, "MODEL": model}
    run.write("system_info.txt", bl.system_info_text(run, info))
    png, text, box = capture_profile(run, a.pz, model, env, a.calibration, rows=a.shot_rows)
    run.log(box or "(no hardware-profile box found in the terminal output)")
    bl.finish(run, {"kind": "hardware_profile", "pz": bl.bin_info(a.pz), "model": str(model),
                    "calibration": a.calibration, "screenshot": run.rel(png) if png else None,
                    "profile_box_found": bool(box)})
    return 0 if (box or a.screenshots == "none") else 1


if __name__ == "__main__":
    sys.exit(main())

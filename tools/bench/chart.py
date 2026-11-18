#!/usr/bin/env python3
"""
chart.py -- decode tok/s vs threads chart (SVG + PNG) from a bench run folder's summary.csv.

One panel per model, one line per variant (Project Zero, Project Zero <classifier>, llama.cpp,
bitnet.cpp, ik_llama.cpp), best run per thread count, exactly the layout of the published sweeps
(it reuses tools/plot_sweep_svg.py, not a copy). h2h_bench.py calls this at the end of a run;
run it by hand to redraw an existing folder:

  python3 tools/bench/chart.py benchmark_results/h2h_20261118-051537Z_four-engines
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import benchlib as bl  # noqa: E402
import plot_sweep_svg as ps  # noqa: E402

ENGINE_NAMES = {"pz": "Project Zero", "llamacpp": "llama.cpp", "bitnetcpp": "bitnet.cpp", "ik": "ik_llama.cpp"}


def variant_name(v):
    if v.startswith("pz-"):
        return f"Project Zero {v[3:].upper()}"
    return ENGINE_NAMES.get(v, v)


def make_chart(run_dir, title="Decode tok/s vs threads", subtitle=None, rounds=None, hide_invalid=False):
    run_dir = pathlib.Path(run_dir)
    rows = [r for r in bl.load_rows(run_dir / "summary.csv") if r["gen_tok_s"] is not None and r.get("exit", "0") == "0"]
    dropped = sorted({f"{r['model']}/{variant_name(r['variant'])}" for r in rows if hide_invalid and r.get("gate") == "INVALID"})
    rows = [r for r in rows if not (hide_invalid and r.get("gate") == "INVALID")]     # correctness gate, strict mode
    if not rows:
        return []
    models = bl.load_models()
    best, order = {}, {}
    for r in rows:
        suite = f"{r['model']}__{r['variant']}"
        t = best.setdefault(suite, {})
        t[r["threads"]] = max(t.get(r["threads"], 0.0), r["gen_tok_s"])
        order.setdefault(r["model"], [])
        if r["variant"] not in order[r["model"]]:
            order[r["model"]].append(r["variant"])
    # A model can be drawn inside another model's panel (chart_panel), e.g. the IQ2_BN file made for
    # ik_llama.cpp sits in the BitNet panel as published.
    host = {m: models.get(m, {}).get("chart_panel", m) for m in order}
    grouped = {}
    for m, vs in order.items():
        grouped.setdefault(host[m], []).extend((f"{m}__{v}", variant_name(v)) for v in vs)
    panels = [(models.get(h, {}).get("title", h), sorted(series, key=lambda s: not s[1].startswith("Project Zero")))
              for h, series in grouped.items()]
    if subtitle is None:
        meta = {}
        try:
            import json
            meta = json.loads((run_dir / "meta.json").read_text())
        except (OSError, ValueError):
            pass
        stamp = run_dir.name.split("_")[1] if "_" in run_dir.name else ""
        when = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]} UTC" if len(stamp) == 16 else stamp
        subtitle = (f"{bl.cpu_model()}, {bl.os.cpu_count()} vCPU | {when}, "
                    f"best of {rounds or meta.get('rounds', 'N')}, pz {bl.git_info()['commit']}")
    if dropped:
        subtitle += " | Not shown (fail the correctness gate): " + ", ".join(dropped)
    svg = ps.render(panels, best, title, subtitle)
    out = [run_dir / "throughput_vs_threads.svg"]
    out[0].write_text(svg)
    chrome = ps.find_chromium(None)
    if chrome:
        try:
            out.append(pathlib.Path(ps.svg_to_png(svg, str(run_dir / "throughput_vs_threads.png"), chrome)))
        except (subprocess.CalledProcessError, OSError, ImportError) as e:
            print(f"(PNG skipped: {e})")
    else:
        print("(PNG skipped: no chromium; the SVG is written)")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in make_chart(sys.argv[1]):
        print("wrote", p)

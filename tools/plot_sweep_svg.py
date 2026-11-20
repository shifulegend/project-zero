#!/usr/bin/env python3
"""
plot_sweep_svg.py — decode tok/s vs threads from a bench_sweep_seq.py sweep, as a
dependency-free SVG (small multiples: one panel per model, one line per engine).

Usage:
  python3 tools/plot_sweep_svg.py --sweep benchmark_results/sweep_h2h_2026-10-19 \\
      --panel "SmolLM2-135M F16=smollm2_135m_pz:Project Zero,smollm2_135m_llamacpp:llama.cpp" \\
      [--panel ...] --title "..." --subtitle "..." [--png]

Best run per (suite, threads) from summary.csv; failed runs (exit != 0) are skipped.
--png also rasterises with headless Chromium (PLAYWRIGHT_BROWSERS_PATH or --chromium).
"""

import argparse
import csv
import glob
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass

BG, INK, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#6b6a64", "#e7e6e1"
# Engine colours: Project Zero first, references after (validated pair, CVD-safe).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#c4972b"]


@dataclass(order=True)
class LineEnd:
    """Where a series ends in a panel: label position, name, last value, colour, marker style."""

    y: float
    name: str
    value: float
    colour: str
    marker_style: int


Panel = tuple[str, list[tuple[str, str]]]  # (label, [(suite, engine name)])


def best_by_suite(summary: str) -> dict[str, dict[int, float]]:
    """Best generation tok/s per suite and thread count from the sweep summary CSV."""
    best: defaultdict[str, dict[int, float]] = defaultdict(dict)
    with open(summary, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("exit", "0") != "0" or not r.get("gen_tok_s"):
                continue
            t, v = int(r["threads"]), float(r["gen_tok_s"])
            if v > best[r["suite"]].get(t, 0.0):
                best[r["suite"]][t] = v
    return best


def nice_max(v: float) -> tuple[float, float]:
    """Axis maximum and tick step for a value (the first step that gives at most four ticks)."""
    for step in (1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if v <= step * 4:
            return step * 4, step
    return v, v / 4


def marker(i: int, x: float, y: float, c: str) -> str:
    """SVG marker for series `i` at (x, y) in colour `c`."""
    if i == 0:
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c}" stroke="{BG}" stroke-width="2"/>'
        )
    return (
        f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5" '
        f'fill="{c}" stroke="{BG}" stroke-width="2"/>'
    )


def render(
    panels: list[Panel], best: dict[str, dict[int, float]], title: str, subtitle: str
) -> str:
    """The chart as an SVG string for the given panels, best values, title and subtitle."""
    # One colour per engine name across every panel (2026-10-30: colours were per panel position,
    # so the same colour meant different engines in different panels); Project Zero series are
    # drawn as circles, references as squares.
    names: list[str] = []
    for _, series in panels:
        names += [n for _, n in series if n not in names]
    colour = {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}
    n_sub = len(subtitle.split(" | "))  # every subtitle line pushes the panels down
    wide = max(
        0, 7 * max(len(n) for n in names) - 130
    )  # long end labels get room instead of running into the next panel
    pw, gap, left, top, ph = 360 + wide, 40, 50, 112 + 16 * max(0, n_sub - 2), 260
    w = left + len(panels) * pw + (len(panels) - 1) * gap + 30
    h = top + ph + 70
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        'font-family="ui-monospace,Menlo,Consolas,monospace">',
        f'<rect width="{w}" height="{h}" fill="{BG}"/>',
        f'<text x="{left - 10}" y="34" font-size="18" fill="{INK}" font-weight="600">{title}</text>',
    ]
    for li, line in enumerate(subtitle.split(" | ")):  # " | " starts a new subtitle line
        out.append(
            f'<text x="{left - 10}" y="{56 + 16 * li}" font-size="12" fill="{MUTED}">{line}</text>'
        )
    for pi, (label, series) in enumerate(panels):
        x0 = left + pi * (pw + gap)
        x1 = x0 + pw - 150  # room for end labels
        y0, y1 = top + ph, top
        threads = sorted({t for s, _ in series for t in best.get(s, {})})
        vmax = max((v for s, _ in series for v in best.get(s, {}).values()), default=1.0)
        ymax, step = nice_max(vmax * 1.05)
        out.append(f'<text x="{x0}" y="{top - 16}" font-size="13" fill="{INK}">{label}</text>')
        k = 0.0
        while k <= ymax + 1e-9:
            y = y0 - (y0 - y1) * k / ymax
            out.append(
                f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{x0 - 6}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{k:g}</text>'
            )
            k += step
        xs = {t: x0 + (x1 - x0) * i / max(1, len(threads) - 1) for i, t in enumerate(threads)}
        for t, x in xs.items():
            out.append(
                f'<text x="{x:.1f}" y="{y0 + 18}" font-size="11" fill="{MUTED}" text-anchor="middle">t={t}</text>'
            )
        ends: list[LineEnd] = []
        for si, (suite, name) in enumerate(series):
            pts = [
                (xs[t], y0 - (y0 - y1) * v / ymax) for t, v in sorted(best.get(suite, {}).items())
            ]
            if not pts:
                continue
            c = colour[name]
            si = 0 if name.startswith("Project Zero") else 1
            out.append(
                '<polyline points="'
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                + f'" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>'
            )
            out += [marker(si, x, y, c) for x, y in pts]
            last_t = max(best[suite])
            ends.append(LineEnd(pts[-1][1], name, best[suite][last_t], c, si))
        # Direct labels at the line ends, nudged apart so they never overlap.
        ends.sort()
        for i in range(1, len(ends)):
            if ends[i].y - ends[i - 1].y < 30:
                ends[i].y = ends[i - 1].y + 30
        for end in ends:
            y, name, c, si = end.y, end.name, end.colour, end.marker_style
            out.append(marker(si, x1 + 14, y, c))
            out.append(
                f'<text x="{x1 + 24}" y="{y + 3:.1f}" font-size="11" fill="{INK}">{name}</text>'
            )
            out.append(
                f'<text x="{x1 + 24}" y="{y + 16:.1f}" font-size="11" fill="{MUTED}">{end.value:.1f} tok/s</text>'
            )
    out.append(
        f'<text x="{left - 10}" y="{h - 16}" font-size="11" fill="{MUTED}">decode tokens/s, '
        "best of N runs per thread count (higher is better)</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def find_chromium(explicit: str | None) -> str | None:
    """Path of a Chromium binary: the explicit one, else one found under the Playwright browsers folder."""
    if explicit:
        return explicit
    root = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    hits = sorted(glob.glob(os.path.join(root, "chromium-*", "chrome-linux*", "chrome")))
    if (
        not hits
        and os.access(os.path.join(root, "chromium"), os.X_OK)
        and os.path.isfile(os.path.join(root, "chromium"))
    ):
        hits = [os.path.join(root, "chromium")]
    return hits[-1] if hits else None


def svg_to_png(svg: str, png_path: str, chrome: str) -> str:
    """Rasterise the SVG with headless Chromium (2x) and crop to the SVG box."""
    w = svg.split('width="', 1)[1].split('"', 1)[0]
    h = svg.split('height="', 1)[1].split('"', 1)[0]
    # Headless window sizes include browser chrome on some versions, so
    # render into a taller window (2x scale) and crop to the SVG box.
    html = png_path + ".html"
    with open(html, "w", encoding="utf-8") as f:
        f.write('<html><body style="margin:0;background:%s">%s</body></html>' % (BG, svg))
    try:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={w},{int(h) + 200}",
                f"--screenshot={os.path.abspath(png_path)}",
                "file://" + os.path.abspath(html),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        os.remove(html)
    from PIL import Image

    im = Image.open(png_path)
    im.crop((0, 0, int(w) * 2, int(h) * 2)).save(png_path)
    return png_path


def main() -> None:
    """Command-line entry point: renders the sweep chart as SVG and, when a browser is available, PNG."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument(
        "--panel",
        action="append",
        required=True,
        help='"Label=suite:Engine name,suite:Engine name"',
    )
    ap.add_argument("--title", default="Decode throughput vs threads")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--out", default="throughput_vs_threads")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--chromium")
    a = ap.parse_args()

    panels = []
    for spec in a.panel:
        label, rest = spec.split("=", 1)
        panels.append((label, [tuple(p.split(":", 1)) for p in rest.split(",")]))
    svg = render(panels, best_by_suite(os.path.join(a.sweep, "summary.csv")), a.title, a.subtitle)
    svg_path = os.path.join(a.sweep, a.out + ".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", svg_path)
    if a.png:
        chrome = find_chromium(a.chromium)
        if not chrome:
            print("no chromium found; skipping PNG")
            return
        print("wrote", svg_to_png(svg, os.path.join(a.sweep, a.out + ".png"), chrome))


if __name__ == "__main__":
    main()

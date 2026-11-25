#!/usr/bin/env python3
"""Unit tests for tools/bench (parsers, argument handling, matrix bookkeeping). No pz run, no network."""

import pathlib
import sys
import tempfile
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl
import calibration_test as ct
import hardware_profile as hp
from bench_sweep_seq import gen_toks

MEASURED = {"tps": 1.0, "tokens": 1, "wall": 0.0, "rss_mb": 1, "cpu_pct": 1, "rc": 0, "log": "x"}


def fake_variant(label: str) -> bl.Variant:
    """A variant whose generation command is always ["x"] and that has no perplexity command."""

    def cmd(
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        tokens: int,
        threads: int,
        fixed: bool,
        prompt: str = "",
    ) -> list[str] | None:
        """The generation command of the fake variant."""
        return ["x"]

    def ppl(
        key: str,
        info: dict[str, Any],
        path: pathlib.Path,
        text: str | pathlib.Path,
        ctx: int,
        chunks: int,
        threads: int,
        extra: Any = (),
    ) -> list[str] | None:
        """The fake variant has no perplexity command."""
        return None

    return {"label": label, "engine": "pz", "cmd": cmd, "ppl": ppl, "env": None}


SWEEP = """
  auto = AVX-512 VNNI;  candidates: scalar avx2 avx512f vnni256
  auto             111.6  [ 81.5 – 121.0] tok/s  ← BEST
  scalar             2.4  [  1.9 –   2.6] tok/s  beat auto in  0/15 rounds
  avx2              15.1  [  6.8 –  17.3] tok/s  beat auto in  1/15 rounds
  T=1   bg:  0%  warmup[.] reps[.....]    31.9  [ 30.5 –  34.1] tok/s  ← PEAK
  T=4   bg:  3%  warmup[.] reps[.....]    98.1  [ 54.4 – 107.9] tok/s  ← PEAK
│    SIMD      : auto       (111.6 tok/s at T=4)                 │
│    Threads   : T=4   ( 98.1 tok/s, swept T=1..4)           │
"""


class Args(unittest.TestCase):
    """Args."""

    def test_threads(self) -> None:
        """Threads."""
        self.assertEqual(bl.parse_threads("1-3,6"), [1, 2, 3, 6])
        self.assertEqual(bl.parse_threads("4"), [4])

    def test_lists(self) -> None:
        """Lists."""
        self.assertEqual(bl.parse_list("a,b", ["a", "b", "c"], "x"), ["a", "b"])
        self.assertEqual(bl.parse_list("all", ["a", "b"], "x"), ["a", "b"])
        self.assertEqual(bl.parse_list("none", ["a"], "x"), [])
        with self.assertRaises(SystemExit):
            bl.parse_list("zzz", ["a"], "x")

    def test_stamp_is_sortable_utc(self) -> None:
        """Stamp is sortable utc."""
        s = bl.utc_stamp()
        self.assertRegex(s, r"^\d{8}-\d{6}Z$")
        self.assertRegex(bl.iso_utc(), r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")


class Parsers(unittest.TestCase):
    """Parsers."""

    def test_calibration_sweep(self) -> None:
        """Calibration sweep."""
        s = ct.parse_sweep(SWEEP)
        self.assertEqual([c[0] for c in s["candidates"]], ["auto", "scalar", "avx2"])
        self.assertEqual(s["candidates"][2][4:], ("1", "15"))
        self.assertEqual(s["pick_simd"], "auto")
        self.assertEqual(s["pick_threads"], "4")
        self.assertEqual([t[0] for t in s["threads"]], ["1", "4"])

    def test_engine_output(self) -> None:
        """Engine output."""
        self.assertEqual(gen_toks("[gen] 25.20 tok/s (128 tokens)")[0], 25.2)
        llama = (
            "llama_perf_context_print:        eval time =    1000.00 ms /   127 runs   (    7.87 ms"
            " per token,   127.00 tokens per second)"
        )
        self.assertEqual(gen_toks(llama)[0], 127.0)

    def test_step_timing(self) -> None:
        """Step timing."""
        t = (
            'step=11 name="Attention score / softmax" total_ms=29.46 count=1320 avg_ms=0.02\nstep=6'
            ' name="KV-A compression" total_ms=0.0 count=0 avg_ms=0'
        )
        self.assertEqual(bl.parse_step_timing(t), {"Attention score / softmax": 29.46})

    def test_profile_box(self) -> None:
        """Profile box."""
        box = "┌─────┐\n│ Hardware Profile (auto-detected) │\n├─────┤\n│ x │\n└─────┘"
        self.assertIn("Hardware Profile", hp.profile_box("noise\n" + box + "\nmore"))
        self.assertEqual(hp.profile_box("no box here"), "")

    def test_ansi_stripped(self) -> None:
        """Ansi stripped."""
        self.assertEqual(bl.strip_ansi("\x1b[32mgreen\x1b[0m\r\n"), "green\n")


class Commands(unittest.TestCase):
    """Commands."""

    def test_pz_cmd_fixed_length_and_flags(self) -> None:
        """Pz cmd fixed length and flags."""
        c = bl.pz_cmd("./pz", "m.gguf", 128, 4, ["--simd", "vnni"], True, flags=["--no-calibrate"])
        self.assertIn("--raw-prompt", c)
        self.assertIn("--ignore-eos", c)
        self.assertIn("--no-calibrate", c)
        self.assertEqual(c[c.index("--threads") + 1], "4")

    def test_ref_cmd_runner_styles(self) -> None:
        """Ref cmd runner styles."""
        self.assertIn("-no-cnv", bl.ref_cmd("b", "ref", "m", 8, 2))
        self.assertNotIn("-no-cnv", bl.ref_cmd("b", "cli", "m", 8, 2))

    def test_registries_are_consistent(self) -> None:
        """Registries are consistent."""
        engines, models = bl.load_engines(), bl.load_models()
        for k, m in models.items():
            for e in m["engines"]:
                self.assertTrue(e == "pz" or e in engines, f"{k}: unknown engine {e}")
            if "derived_from" in m:
                self.assertIn(m["derived_from"], models)
            else:
                self.assertTrue(m["url"].startswith("https://") and m["size"] > 0, k)
        for k, e in engines.items():
            self.assertIn(e["runner"], ("ref", "cli"))
            self.assertTrue(e["pin"], k)


class Matrix(unittest.TestCase):
    """Matrix."""

    def test_summary_and_ratio(self) -> None:
        """Summary and ratio."""
        rows = [
            {"model": "m", "variant": v, "threads": 4, "round": r, "gen_tok_s": x}
            for v, xs in (("a", [10.0, 12.0]), ("b", [5.0, 6.0]))
            for r, x in enumerate(xs, 1)
        ]
        s = bl.summarize(rows, "a")
        by = {o["variant"]: o for o in s}
        self.assertAlmostEqual(by["a"]["mean"], 11.0)
        self.assertAlmostEqual(by["b"]["ratio"], 5.5 / 11.0)
        self.assertIn("| m | a | 4 | 2 |", bl.summary_md(s, True))

    def test_rotation_order_and_resume(self) -> None:
        """Interleaved order rotates the starting variant; --resume skips recorded rows."""
        with tempfile.TemporaryDirectory() as d:
            run = bl.RunDir("t", d)
            run.create()
            seen: list[tuple[str, int]] = []
            orig = bl.measure

            def fake_measure(
                run: bl.RunDir,
                suite: str,
                threads: int,
                rnd: int,
                cmd: list[str],
                env: dict[str, str] | None = None,
                prompt: str = "",
            ) -> dict[str, Any]:
                """A measurement that always returns the canned result."""
                seen.append((suite, rnd))
                return dict(MEASURED)

            bl.measure = fake_measure
            try:
                vs = [fake_variant(l) for l in ("a", "b", "c")]
                models = [("m", {"tokens": 4}, pathlib.Path("m.gguf"))]
                bl.run_matrix(run, vs, models, [4], 3, 0, 0.0)
                first_per_round = [
                    s for (s, r), i in zip(seen, range(len(seen)), strict=False) if i % 3 == 0
                ]
                self.assertEqual(len(set(first_per_round)), 3, "the starting variant rotates")
                n = len(seen)
                bl.run_matrix(run, vs, models, [4], 3, 0, 0.0, resume=True)
                self.assertEqual(len(seen), n, "resume re-ran recorded rows")
            finally:
                bl.measure = orig
                run.close()


class Chart(unittest.TestCase):
    """Chart."""

    def test_chart_groups_panels_and_puts_pz_first(self) -> None:
        """Chart groups panels and puts pz first."""
        import chart

        with tempfile.TemporaryDirectory() as tmpdir:
            d = pathlib.Path(tmpdir)
            rows = [
                ("bitnet-2b", "bitnetcpp", 1, 5.0),
                ("bitnet-2b", "pz", 1, 4.0),
                ("bitnet-2b", "pz", 2, 8.0),
                ("bitnet-2b-iq2bn", "ik", 1, 7.0),
                ("bitnet-2b-iq2bn", "ik", 2, 12.0),
                ("bitnet-2b", "bitnetcpp", 2, 9.0),
            ]
            with open(d / "summary.csv", "w", encoding="utf-8") as f:
                f.write(",".join(bl.SUMMARY_HEADER) + "\n")
                for m, v, t, x in rows:
                    f.write(f"{m},{v},{t},1,{x},500,10,300,100,0,log\n")
            out = chart.make_chart(d, "T", "S")
            svg = (d / "throughput_vs_threads.svg").read_text()
            self.assertIn(out[0].name, "throughput_vs_threads.svg")
            self.assertEqual(svg.count("IQ2_BN"), 1)  # iq2bn model drawn inside the BitNet panel
            self.assertIn("#2a78d6", svg)  # Project Zero is first, so it gets the first colour
            self.assertIn("t=2", svg)


class Preset(unittest.TestCase):
    """Preset."""

    def test_published_preset_is_the_full_sweep(self) -> None:
        """Published preset is the full sweep."""
        import h2h_bench as h

        p = h.PRESETS["published"]
        self.assertEqual(bl.parse_threads(p["threads"]), [1, 2, 3, 4])
        self.assertEqual((p["tokens"], p["rounds"], p["screenshots"]), (500, 3, "all"))
        self.assertIn("int4", p["pz_classifiers"])

    def test_measurements_never_overlap(self) -> None:
        # one process at a time: run_cmd waits for its child; nothing in the matrix uses threads/pools.
        """Measurements never overlap."""
        src = (pathlib.Path(__file__).parent / "benchlib.py").read_text()
        for bad in (
            "ThreadPool",
            "ProcessPool",
            "concurrent.futures",
            "multiprocessing",
            "asyncio",
        ):
            self.assertNotIn(bad, src)


if __name__ == "__main__":
    unittest.main(verbosity=1)

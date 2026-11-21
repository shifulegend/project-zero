#!/usr/bin/env python3
"""Unit tests for the pass/fail gate of tools/bench (verdicts, report text, exit codes). No pz run, no network."""

import pathlib
import sys
import tempfile
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import benchlib as bl

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


class Gate(unittest.TestCase):
    """Gate."""

    CFG = {"tolerance_pct": 1.0, "warn_factor": 2.0, "mode": "strict"}

    def test_verdict_boundaries_are_two_sided(self) -> None:
        """Verdict boundaries are two sided."""
        import gate

        def v(x: float) -> str:
            """Helper of the gate tests."""
            return gate.verdict(x, 100.0, 1.0, 2.0)[0]

        self.assertEqual(
            [v(100.9), v(99.1), v(101.5), v(98.5), v(103.0), v(96.0)],
            ["PASS", "PASS", "WARN", "WARN", "INVALID", "INVALID"],
        )
        self.assertEqual(gate.verdict(None, 100.0, 1.0, 2.0)[0], "INVALID")
        self.assertEqual(gate.verdict(5.0, None, 1.0, 2.0)[0], "UNVERIFIED")

    def test_the_rca_numbers(self) -> None:
        """The rca numbers."""
        import gate

        self.assertEqual(
            gate.verdict(34.3926, 31.4010, 1.0, 2.0)[0], "INVALID"
        )  # pz int8 truncating quantiser
        self.assertEqual(
            gate.verdict(48.6683, 31.4010, 1.0, 2.0)[0], "INVALID"
        )  # bitnet.cpp SiLU graph
        self.assertEqual(
            gate.verdict(31.3193, 31.4010, 1.0, 2.0)[0], "PASS"
        )  # pz float activations
        self.assertEqual(gate.verdict(12.5801, 12.5667, 1.0, 2.0)[0], "PASS")  # Mamba

    def test_parse_ppl_and_answer_text(self) -> None:
        """Parse ppl and answer text."""
        import gate

        rx = gate.load_config()["ppl_regex"]
        self.assertEqual(
            gate.parse_ppl("x\n0.37.5 I Final estimate: PPL = 31.4010 +/- 3.30121\n", rx),
            (31.401, 3.30121),
        )
        self.assertEqual(gate.parse_ppl("no estimate", rx), (None, None))
        self.assertEqual(
            gate.answer_text(
                "0.01.282.588 I llama_completion: x\nThe capital of France is Paris.\n0.02.123.456"
                " I common_perf_print: y"
            ),
            "The capital of France is Paris.",
        )

    def test_smoke_rejects_an_expectation_that_is_in_the_prompt(self) -> None:
        """Smoke rejects an expectation that is in the prompt."""
        import gate

        self.assertTrue(gate.smoke_ok("... is Paris.", ["Paris"], "The capital of France is"))
        self.assertFalse(
            gate.smoke_ok("... is a small city", ["Paris"], "The capital of France is")
        )
        with self.assertRaises(ValueError):
            gate.smoke_ok("x", ["France"], "The capital of France is")

    def test_judge_reference_unverified_and_wrong_smoke(self) -> None:
        """Judge reference unverified and wrong smoke."""
        import gate

        res = {
            "m/ref": {"model": "m", "variant": "ref", "ppl": 10.0, "smoke_ok": True},
            "m/ok": {"model": "m", "variant": "ok", "ppl": 10.05, "smoke_ok": True},
            "m/bad": {"model": "m", "variant": "bad", "ppl": 12.0, "smoke_ok": True},
            "m/wrong": {"model": "m", "variant": "wrong", "ppl": 10.0, "smoke_ok": False},
            "n/x": {"model": "n", "variant": "x", "ppl": 5.0, "smoke_ok": True},
        }
        gate.judge(res, self.CFG, lambda m, v: "m/ref" if m == "m" else "n/missing")
        self.assertEqual(
            {k: r["verdict"] for k, r in res.items()},
            {
                "m/ref": "REF",
                "m/ok": "PASS",
                "m/bad": "INVALID",
                "m/wrong": "INVALID",
                "n/x": "UNVERIFIED",
            },
        )
        self.assertEqual(gate.exit_status(res, self.CFG), 1)
        self.assertEqual(gate.exit_status(res, dict(self.CFG, mode="warn")), 0)
        self.assertTrue(gate.blocked(res["m/bad"], self.CFG))
        self.assertFalse(gate.blocked(res["m/ok"], self.CFG))

    def test_approximate_variants_are_reported_not_judged(self) -> None:
        """Approximate variants are reported not judged."""
        import gate

        cfg = dict(
            self.CFG, approximate={"lossy": {"why": "by design"}, "bounded": {"tolerance_pct": 5.0}}
        )
        res = {
            "m/ref": {"model": "m", "variant": "ref", "ppl": 10.0, "smoke_ok": True},
            "m/lossy": {"model": "m", "variant": "lossy", "ppl": 12.0, "smoke_ok": True},
            "m/bounded": {"model": "m", "variant": "bounded", "ppl": 10.4, "smoke_ok": True},
            "m/bounded2": {"model": "m", "variant": "bounded", "ppl": 14.0, "smoke_ok": True},
            "m/lossy_wrong": {"model": "m", "variant": "lossy", "ppl": 12.0, "smoke_ok": False},
        }
        gate.judge(res, cfg, lambda m, v: "m/ref")
        self.assertEqual(res["m/lossy"]["verdict"], "APPROX")
        self.assertEqual(
            res["m/bounded"]["verdict"], "APPROX"
        )  # 4 % off, inside its 5 % hard bound
        self.assertEqual(
            res["m/bounded2"]["verdict"], "INVALID"
        )  # 40 % off, beyond the 5 % hard bound
        self.assertEqual(
            res["m/lossy_wrong"]["verdict"], "INVALID"
        )  # a lossy variant must still answer correctly
        self.assertIn("APPROX", gate.GOOD)

    def test_ik_perplexity_format_is_parsed(self) -> None:
        """Ik perplexity format is parsed."""
        import gate

        rx = gate.load_config()["ppl_regex"]
        self.assertEqual(
            gate.parse_ppl(
                "Final estimate: PPL over 7 chunks for n_ctx=512 = 8.0256 +/- 0.54101", rx
            ),
            (8.0256, 0.54101),
        )

    def test_cells_that_fail_the_gate_are_not_timed(self) -> None:
        """Cells that fail the gate are not timed."""
        import gate

        strict, warn = {"mode": "strict"}, {"mode": "warn"}
        self.assertFalse(gate.should_time("INVALID", strict))
        self.assertTrue(gate.should_time("INVALID", warn))  # warn only marks
        self.assertTrue(gate.should_time("INVALID", dict(strict, time_invalid=True)))
        for v in ("PASS", "WARN", "REF", "APPROX", "UNVERIFIED", "OFF"):
            self.assertTrue(gate.should_time(v, strict), v)
        with tempfile.TemporaryDirectory() as d:  # run_matrix honours skip_of
            run = bl.RunDir("t", d)
            run.create()
            seen_suites: list[str] = []
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
                seen_suites.append(suite)
                return dict(MEASURED)

            bl.measure = fake_measure
            try:
                vs = [fake_variant(l) for l in ("ok", "bad")]
                bl.run_matrix(
                    run,
                    vs,
                    [("m", {"tokens": 4}, pathlib.Path("m.gguf"))],
                    [4],
                    2,
                    0,
                    0.0,
                    skip_of=lambda k, v: v == "bad",
                )
                self.assertEqual(sorted(set(seen_suites)), ["m__ok"])
                self.assertEqual(len(seen_suites), 2)
            finally:
                bl.measure = orig
                run.close()

    def test_dirty_binaries_are_refused_unless_allowed(self) -> None:
        """Dirty binaries are refused unless allowed."""
        import gate

        self.assertTrue(gate.is_dirty("Project Zero v0.2.0-21-g864eeea-dirty"))
        self.assertFalse(gate.is_dirty("Project Zero v0.2.0-23-g0148ad0"))
        with tempfile.TemporaryDirectory() as d:
            fake = pathlib.Path(d) / "pz"
            fake.write_text("#!/bin/sh\necho 'Project Zero v0.2.0-1-gabc-dirty'\n")
            fake.chmod(0o755)
            cfg = {"require_clean_binary": True}
            with self.assertRaises(SystemExit):
                gate.require_clean({"pz": str(fake)}, cfg)
            self.assertEqual(
                gate.require_clean({"pz": str(fake)}, cfg, allow=True, log=lambda m: None), ["pz"]
            )
            self.assertEqual(
                gate.require_clean({"pz": str(fake)}, {"require_clean_binary": False}), ["pz"]
            )

    def test_config_overrides_and_frozen_text(self) -> None:
        """Config overrides and frozen text."""
        import gate

        cfg = gate.load_config({"tolerance_pct": 0.5, "mode": None})
        self.assertEqual((cfg["tolerance_pct"], cfg["mode"]), (0.5, "strict"))
        self.assertEqual(len(cfg["text_sha256"]), 64)  # the frozen text exists and is hashed

    def test_flags_exist_on_every_script(self) -> None:
        """Flags exist on every script."""
        import argparse

        ap = bl.add_common_args(argparse.ArgumentParser(), "x")
        a = ap.parse_args(
            ["--gate", "warn", "--gate-tolerance", "2.5", "--gate-only", "--gate-reference", "pz"]
        )
        self.assertEqual(
            (a.gate, a.gate_tolerance, a.gate_only, a.gate_reference), ("warn", 2.5, True, "pz")
        )

    def test_summary_column_and_chart_hide_invalid(self) -> None:
        """Summary column and chart hide invalid."""
        import chart
        import gate

        with tempfile.TemporaryDirectory() as tmpdir:
            d = pathlib.Path(tmpdir)
            old = [
                "model",
                "variant",
                "threads",
                "round",
                "gen_tok_s",
                "tokens",
                "wall_s",
                "max_rss_mb",
                "cpu_pct",
                "exit",
                "log",
            ]
            bl.write_csv(
                d / "summary.csv",
                old,
                [
                    ["bitnet-2b", "pz", 1, 1, 4.0, 500, 1, 1, 1, 0, "l"],
                    ["bitnet-2b", "bitnetcpp", 1, 1, 5.0, 500, 1, 1, 1, 0, "l"],
                ],
            )
            bl.ensure_gate_column(d / "summary.csv")
            gate.apply_to_summary(
                d / "summary.csv",
                {
                    "bitnet-2b/pz": {"verdict": "PASS"},
                    "bitnet-2b/bitnetcpp": {"verdict": "INVALID"},
                },
            )
            rows = bl.load_rows(d / "summary.csv")
            self.assertEqual([r["gate"] for r in rows], ["PASS", "INVALID"])
            svg = chart.make_chart(d, "T", "S", hide_invalid=True)[0].read_text()
            self.assertIn("Not shown (fail the correctness gate)", svg)
            self.assertNotIn("5.0 tok/s", svg)
            md = bl.summary_md(bl.summarize(rows), hide_invalid=True)
            self.assertIn("n/a", md)

    def test_variant_factories_and_registry(self) -> None:
        """Variant factories and registry."""
        info: dict[str, Any] = {"engines": ["pz"], "gate": {}}
        gguf = pathlib.Path("m.gguf")
        v = bl.pz_variant("pz", "./pz", ["--no-calibrate"], None)
        pz_ppl = v["ppl"]("m", info, gguf, "t.txt", 512, 8, 4)
        assert pz_ppl is not None
        self.assertIn("--perplexity", pz_ppl)
        self.assertIsNone(v["ppl"]("m", {"engines": ["ik"]}, gguf, "t.txt", 512, 8, 4))
        r = bl.ref_variant("llamacpp", "llamacpp", "/b/llama-completion", None)
        self.assertIsNone(
            r["ppl"]("m", {"engines": ["llamacpp"]}, gguf, "t.txt", 512, 8, 4)
        )  # no perplexity tool
        engines, models = bl.load_engines(), bl.load_models()
        for (
            k,
            m,
        ) in models.items():  # every gate reference names a real engine (or pz) that runs the model
            spec = m["gate"]["reference"]
            ref_model = models[spec.get("model", k)]
            self.assertTrue(spec["variant"] in engines or spec["variant"].startswith("pz"), k)
            self.assertIn(spec["variant"], ref_model["engines"], k)
        for k, e in engines.items():
            self.assertIn("perplexity", e, k)
            for p in e.get("patches", []):
                self.assertTrue((bl.ROOT / p["path"]).exists(), p["path"])
        self.assertTrue(engines["bitnetcpp-fix"]["modified"])


if __name__ == "__main__":
    unittest.main(verbosity=1)

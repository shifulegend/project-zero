"""Tests for tools/archive_benchmark_results.py (the selection rule)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import archive_benchmark_results as a

RUN = a.KEEP_RUN


class SelectionTest(unittest.TestCase):
    """What stays and what is archived."""

    def test_latest_run_stays_without_screenshots(self) -> None:
        """The kept run keeps its data but not its screenshots."""
        self.assertTrue(a.keep(f"benchmark_results/{RUN}/summary.csv"))
        self.assertTrue(a.keep(f"benchmark_results/{RUN}/raw/x.txt"))
        self.assertFalse(a.keep(f"benchmark_results/{RUN}/screenshots/a.png"))

    def test_everything_else_is_archived(self) -> None:
        """Other runs, loose logs and charts go; the README stays."""
        self.assertTrue(a.keep("benchmark_results/README.md"))
        self.assertFalse(a.keep("benchmark_results/sweep_2026-06-21/x.txt"))
        self.assertFalse(a.keep("benchmark_results/comparison_graph.png"))

    def test_split_partitions(self) -> None:
        """Every file is either kept or archived."""
        files = ["benchmark_results/README.md", "benchmark_results/old/x.txt"]
        self.assertEqual(
            a.split(files), (["benchmark_results/README.md"], ["benchmark_results/old/x.txt"])
        )


if __name__ == "__main__":
    unittest.main()

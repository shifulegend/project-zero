import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_coverage as c

INFO = "SF:a.c\nLF:100\nLH:60\nBRF:50\nBRH:20\nend_of_record\nSF:b.c\nLF:100\nLH:40\nBRF:50\nBRH:30\nend_of_record\n"


class CoverageTests(unittest.TestCase):
    def test_totals_sum_every_file(self) -> None:
        t = c.totals(INFO)
        self.assertEqual(t["lines"], (100, 200))
        self.assertEqual(t["branches"], (50, 100))

    def test_meets_floor(self) -> None:
        self.assertEqual(c.check(INFO, {"lines": 50.0, "branches": 50.0}), [])

    def test_below_floor_is_named(self) -> None:
        out = c.check(INFO, {"lines": 51.0, "branches": 50.0})
        self.assertEqual(len(out), 1)
        self.assertIn("lines", out[0])

    def test_a_stale_floor_must_be_raised(self) -> None:
        """Coverage far above the floor fails, so the floor only ratchets upward."""
        out = c.check(INFO, {"lines": 30.0, "branches": 50.0})
        self.assertEqual(len(out), 1)
        self.assertIn("raise it", out[0])

    def test_empty_tracefile_fails(self) -> None:
        self.assertTrue(c.check("", {"lines": 1.0}))

    def test_floor_file_parses_comments(self) -> None:
        self.assertEqual(
            c.read_floor("# c\nlines 50 # x\nbranches 35\n"), {"lines": 50.0, "branches": 35.0}
        )

    def test_real_floor_file_is_valid(self) -> None:
        floor = c.read_floor(c.FLOOR.read_text(encoding="utf-8"))
        self.assertEqual(set(floor), {"lines", "branches"})


if __name__ == "__main__":
    unittest.main()

"""Tests for tools/check_suppressions.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_suppressions as c


class SuppressionTest(unittest.TestCase):
    def test_counts_each_kind(self) -> None:
        text = 'x // NOLINT(foo) -- why\ny  # noqa: F401\n#pragma GCC diagnostic ignored "-Wx"\n#pragma clang diagnostic push\n'
        self.assertEqual(c.count(text), {"NOLINT": 1, "noqa": 1, "pragma-diagnostic": 2})

    def test_blanket_nolint_is_found_but_one_naming_its_check_is_not(self) -> None:
        text = "a // NOLINT\nb // NOLINT(x) -- reason\nc // NOLINTNEXTLINE\n"
        self.assertEqual(c.bare(text, "NOLINT"), [1, 3])

    def test_noqa_needs_a_reason(self) -> None:
        self.assertEqual(c.bare("import x  # noqa", "noqa"), [1])
        self.assertEqual(c.bare("import x  # noqa: F401", "noqa"), [1])
        self.assertEqual(c.bare("import x  # noqa: F401,E402", "noqa"), [1])
        self.assertEqual(c.bare("import x  # noqa: F401  (side effect)", "noqa"), [])
        self.assertEqual(c.bare("x = 1  # noqa: BLE001 -- keep going", "noqa"), [])

    def test_compare_reports_only_increases(self) -> None:
        base = {"NOLINT": 3, "noqa": 10, "pragma-diagnostic": 1}
        self.assertEqual(c.compare({"NOLINT": 3, "noqa": 9, "pragma-diagnostic": 1}, base), [])
        self.assertEqual(len(c.compare({"NOLINT": 4, "noqa": 10, "pragma-diagnostic": 1}, base)), 1)

    def test_the_real_tree_is_within_the_baseline(self) -> None:
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

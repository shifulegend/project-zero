"""Tests for tools/check_suppressions.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_suppressions as c  # noqa: E402


class SuppressionTest(unittest.TestCase):
    def test_counts_each_kind(self):
        text = 'x // NOLINT(foo) -- why\ny  # noqa: F401\n#pragma GCC diagnostic ignored "-Wx"\n#pragma clang diagnostic push\n'
        self.assertEqual(c.count(text), {"NOLINT": 1, "noqa": 1, "pragma-diagnostic": 2})

    def test_blanket_nolint_is_found_but_one_naming_its_check_is_not(self):
        text = "a // NOLINT\nb // NOLINT(x) -- reason\nc // NOLINTNEXTLINE\n"
        self.assertEqual(c.bare(text, "NOLINT"), [1, 3])

    def test_compare_reports_only_increases(self):
        base = {"NOLINT": 3, "noqa": 10, "pragma-diagnostic": 1}
        self.assertEqual(c.compare({"NOLINT": 3, "noqa": 9, "pragma-diagnostic": 1}, base), [])
        self.assertEqual(len(c.compare({"NOLINT": 4, "noqa": 10, "pragma-diagnostic": 1}, base)), 1)

    def test_the_real_tree_is_within_the_baseline(self):
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

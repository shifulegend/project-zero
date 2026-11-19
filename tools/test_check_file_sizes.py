"""Tests for tools/check_file_sizes.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_file_sizes as c


class SizeTest(unittest.TestCase):
    def test_small_files_pass(self):
        self.assertEqual(c.problems({"src/a.c": 500, "src/b.c": 10}, {}), [])

    def test_a_new_large_file_fails(self):
        out = c.problems({"src/a.c": 501}, {})
        self.assertEqual(len(out), 1)
        self.assertIn("501", out[0])

    def test_a_listed_file_may_stay_or_shrink_but_not_grow(self):
        base = {"src/big.c": 2000}
        self.assertEqual(c.problems({"src/big.c": 2000}, base), [])
        self.assertEqual(c.problems({"src/big.c": 1500}, base), [])
        out = c.problems({"src/big.c": 2001}, base)
        self.assertEqual(len(out), 1)
        self.assertIn("grew", out[0])

    def test_the_real_tree_is_within_the_baseline(self):
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

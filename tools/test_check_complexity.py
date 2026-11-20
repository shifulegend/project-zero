"""Tests for tools/check_complexity.py."""

import importlib.util
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_complexity as c


class ComplexityTest(unittest.TestCase):
    def test_small_functions_pass(self):
        self.assertEqual(c.problems({"src/a.c::f": (15, 100)}, {}), [])

    def test_a_new_complex_function_fails(self):
        out = c.problems({"src/a.c::f": (16, 10)}, {})
        self.assertEqual(len(out), 1)
        self.assertIn("complexity 16", out[0])

    def test_a_new_long_function_fails(self):
        out = c.problems({"src/a.c::f": (3, 101)}, {})
        self.assertEqual(len(out), 1)
        self.assertIn("101 lines", out[0])

    def test_a_listed_function_may_stay_or_shrink_but_not_grow(self):
        base = {"src/a.c::big": (40, 300)}
        self.assertEqual(c.problems({"src/a.c::big": (40, 300)}, base), [])
        self.assertEqual(c.problems({"src/a.c::big": (30, 200)}, base), [])
        for grown in ((41, 300), (40, 301)):
            out = c.problems({"src/a.c::big": grown}, base)
            self.assertEqual(len(out), 1)
            self.assertIn("grew", out[0])

    def test_a_listed_function_that_got_small_is_fine(self):
        self.assertEqual(c.problems({"src/a.c::big": (5, 20)}, {"src/a.c::big": (40, 300)}), [])

    def test_the_real_tree_is_within_the_baseline(self):
        if importlib.util.find_spec("lizard") is None:
            self.skipTest("lizard is not installed")
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

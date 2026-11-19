"""Tests for tools/check_mutable_statics.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_mutable_statics as c  # noqa: E402


class StaticTest(unittest.TestCase):
    def test_counts_mutable_statics_only(self):
        text = "\n".join(
            [
                "static int g_count = 0;",
                "static float buf[64];",
                "    static char scratch[8192];",
                "static const int K = 3;",
                "static const char *const NAMES[] = {0};",
                "static inline int f(int x) { return x; }",
                "static int helper(int x);",
                "// static int in_a_comment;",
                "    static float *p = NULL;",
            ]
        )
        self.assertEqual(c.count(text), 4)

    def test_compare_only_reports_increases(self):
        self.assertEqual(c.compare({"a.c": 2}, {"a.c": 2}), [])
        self.assertEqual(c.compare({"a.c": 1}, {"a.c": 2}), [])
        self.assertEqual(len(c.compare({"a.c": 3}, {"a.c": 2})), 1)
        self.assertEqual(len(c.compare({"new.c": 1}, {})), 1)

    def test_the_real_tree_is_within_the_baseline(self):
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

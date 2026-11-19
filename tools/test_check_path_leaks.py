"""Tests for tools/check_path_leaks.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_path_leaks as c  # noqa: E402


class LeakTest(unittest.TestCase):
    def test_flags_each_kind(self):
        for text in (
            "see /home/alice/project/x",
            "in /Users/bob/Documents",
            "models at /mnt/windows/models",
            "session e6da17c0-b183-4d34-8716-bf61d92a4ac6",
            "github.com/someone/readme-guidance",
        ):
            self.assertEqual(len(c.problems(text, "a.md")), 1, text)

    def test_neutral_paths_pass(self):
        for text in (
            "reference builds in /home/user/ext/llama.cpp",
            "use <HOME>/models",
            "no paths here",
            "a/b/home/x",
        ):
            self.assertEqual(c.problems(text, "a.md"), [], text)

    def test_scope(self):
        self.assertTrue(c.in_scope("docs/ai/mistakes.md"))
        self.assertTrue(c.in_scope("README.md"))
        self.assertFalse(c.in_scope("docs/PERFORMANCE_CEILING_REPORT.md"))
        self.assertFalse(c.in_scope("src/a.c"))

    def test_the_real_tree_is_clean(self):
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()

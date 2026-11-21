"""Tests for tools/check_line_width.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_line_width as c


class WidthTest(unittest.TestCase):
    """Counting and comparing wide lines."""

    def test_only_lines_over_the_limit_count(self) -> None:
        """A line of exactly 100 characters is fine, 101 is not."""
        text = "a" * 100 + "\n" + "b" * 101 + "\nshort\n"
        self.assertEqual(c.wide_lines(text), 1)

    def test_growth_and_new_files_fail(self) -> None:
        """A higher count or a new file is a problem; the same or lower is not."""
        self.assertEqual(c.problems({"a.c": 2}, {"a.c": 2}), [])
        self.assertEqual(c.problems({"a.c": 1}, {"a.c": 2}), [])
        self.assertEqual(len(c.problems({"a.c": 3}, {"a.c": 2})), 1)
        self.assertEqual(len(c.problems({"b.c": 1}, {})), 1)

    def test_real_tree_is_within_the_baseline(self) -> None:
        """The committed tree matches the recorded counts."""
        self.assertEqual(c.main(["check_line_width.py"]), 0)


if __name__ == "__main__":
    unittest.main()

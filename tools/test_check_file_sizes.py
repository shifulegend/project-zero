"""Tests for tools/check_file_sizes.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_file_sizes as c


class SizeTest(unittest.TestCase):
    """Tests for size."""

    def test_small_files_pass(self) -> None:
        """Small files pass."""
        self.assertEqual(c.problems({"src/a.c": 500, "src/b.c": 10}, {}), [])

    def test_a_new_large_file_fails(self) -> None:
        """A new large file fails."""
        out = c.problems({"src/a.c": 501}, {})
        self.assertEqual(len(out), 1)
        self.assertIn("501", out[0])

    def test_a_listed_file_may_stay_or_shrink_but_not_grow(self) -> None:
        """A listed file may stay or shrink but not grow."""
        base = {"src/big.c": 2000}
        self.assertEqual(c.problems({"src/big.c": 2000}, base), [])
        self.assertEqual(c.problems({"src/big.c": 1500}, base), [])
        out = c.problems({"src/big.c": 2001}, base)
        self.assertEqual(len(out), 1)
        self.assertIn("grew", out[0])

    def test_docstring_lines_are_not_counted(self) -> None:
        """Docstring lines are not counted."""
        src = '"""Module\ndoc."""\n\ndef f():\n    """One."""\n    return 1\n\nclass K:\n    """A\n    B."""\n'
        self.assertEqual(c.docstring_lines(src), 2 + 1 + 2)
        self.assertEqual(c.docstring_lines("def broken(:"), 0)

    def test_the_real_tree_is_within_the_baseline(self) -> None:
        """The real tree is within the baseline."""
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

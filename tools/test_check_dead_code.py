"""Tests for tools/check_dead_code.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_dead_code as c


class DeadCodeTest(unittest.TestCase):
    """Finding disabled code."""

    def test_if_zero_is_found(self) -> None:
        """An `#if 0` line is reported, `#if 1` and `#ifdef` are not."""
        text = "#if 0\nint x;\n#endif\n#if 1\n#endif\n#ifdef A\n#endif\n"
        self.assertEqual(c.violations(text), [(1, "#if 0 block")])

    def test_commented_statements_are_found(self) -> None:
        """A commented call or assignment is reported; prose is not."""
        text = "// foo(a, b);\n// x->y = 3;\n// Returns the count.\n// see foo() for details\n"
        self.assertEqual([n for n, _ in c.violations(text)], [1, 2])

    def test_real_tree_is_clean(self) -> None:
        """The source tree has no disabled code."""
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()

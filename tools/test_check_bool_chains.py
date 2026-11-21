"""Tests for tools/check_bool_chains.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_bool_chains as c


class CountTest(unittest.TestCase):
    """Counting conditions with four or more operands."""

    def test_short_conditions_are_fine(self) -> None:
        """Up to three operands do not count."""
        self.assertEqual(c.long_chains("if (a && b || c) {}"), 0)

    def test_four_operands_count(self) -> None:
        """Three joining operators make a long condition, across lines too."""
        self.assertEqual(c.long_chains("if (a && b &&\n    c || d) {}"), 1)

    def test_nested_and_call_parentheses_are_not_top_level(self) -> None:
        """Operators inside a nested group or a call belong to that group."""
        self.assertEqual(c.long_chains("if (f(a && b && c && d)) {}"), 0)

    def test_comments_and_strings_are_ignored(self) -> None:
        """Operators in a comment or a string literal do not count."""
        self.assertEqual(c.long_chains('if (a) { /* if (a&&b&&c&&d) */ s("&&&&&&"); }'), 0)

    def test_while_counts_too(self) -> None:
        """while conditions are checked like if conditions."""
        self.assertEqual(c.long_chains("while (a && b && c && d) {}"), 1)


class RatchetTest(unittest.TestCase):
    """The comparison with the baseline."""

    def test_growth_and_new_files_fail(self) -> None:
        """A higher count or a new file is a problem; the same or lower is not."""
        self.assertEqual(c.problems({"a.c": 2}, {"a.c": 2}), [])
        self.assertEqual(c.problems({"a.c": 1}, {"a.c": 2}), [])
        self.assertEqual(len(c.problems({"a.c": 3}, {"a.c": 2})), 1)
        self.assertEqual(len(c.problems({"b.c": 1}, {})), 1)

    def test_real_tree_is_within_the_baseline(self) -> None:
        """The committed tree matches the recorded counts."""
        self.assertEqual(c.main(["check_bool_chains.py"]), 0)


if __name__ == "__main__":
    unittest.main()

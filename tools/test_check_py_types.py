"""Tests for tools/check_py_types.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_py_types as c


class PyTypesTest(unittest.TestCase):
    """The ratchet fails on new errors and on stale baseline entries."""

    def test_clean_tree_passes(self) -> None:
        """No errors and an empty baseline is fine."""
        self.assertEqual(c.problems({}, {}), [])

    def test_a_new_file_with_errors_fails(self) -> None:
        """An unlisted file with any error is a problem."""
        out = c.problems({"tools/a.py": 1}, {})
        self.assertEqual(len(out), 1)
        self.assertIn("tools/a.py", out[0])

    def test_a_listed_file_may_stay_or_shrink_but_not_grow(self) -> None:
        """A listed file may keep or reduce its count, never raise it."""
        base = {"tools/a.py": 5}
        self.assertEqual(c.problems({"tools/a.py": 5}, base), [])
        self.assertEqual(c.problems({"tools/a.py": 2}, base), [])
        self.assertEqual(len(c.problems({"tools/a.py": 6}, base)), 1)

    def test_a_clean_file_must_leave_the_baseline(self) -> None:
        """A listed file with no errors left is reported as stale."""
        out = c.problems({}, {"tools/a.py": 3})
        self.assertEqual(len(out), 1)
        self.assertIn("now clean", out[0])


if __name__ == "__main__":
    unittest.main()

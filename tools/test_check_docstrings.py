"""Tests for check docstrings."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_docstrings as c

SRC = '''
def documented():
    """Does a thing."""

def bare():
    pass

def _private():
    pass

class K:
    def method(self):
        pass

    def ok(self):
        """Fine."""
'''


class DocstringTests(unittest.TestCase):
    """Docstring Tests."""

    def test_missing_lists_private_helpers_too(self) -> None:
        """Private helpers count like public functions; dunder methods do not."""
        self.assertEqual(sorted(c.missing(SRC)), ["_private", "bare", "method"])

    def test_compare_flags_growth_only(self) -> None:
        """Compare flags growth only."""
        self.assertEqual(c.compare({"a.py": 2}, {"a.py": 2}), [])
        self.assertEqual(c.compare({"a.py": 1}, {"a.py": 2}), [])
        self.assertEqual(len(c.compare({"a.py": 3}, {"a.py": 2})), 1)
        self.assertEqual(len(c.compare({"new.py": 1}, {})), 1)

    def test_baseline_roundtrip(self) -> None:
        """Baseline roundtrip."""
        self.assertEqual(c.read_baseline("# h\nx.py 3\ny/z.py 1\n"), {"x.py": 3, "y/z.py": 1})

    def test_the_real_tree_is_within_baseline(self) -> None:
        """The real tree is within baseline."""
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

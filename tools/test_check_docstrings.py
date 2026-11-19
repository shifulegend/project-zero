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
    def test_missing_lists_public_functions_only(self):
        self.assertEqual(sorted(c.missing(SRC)), ["bare", "method"])

    def test_compare_flags_growth_only(self):
        self.assertEqual(c.compare({"a.py": 2}, {"a.py": 2}), [])
        self.assertEqual(c.compare({"a.py": 1}, {"a.py": 2}), [])
        self.assertEqual(len(c.compare({"a.py": 3}, {"a.py": 2})), 1)
        self.assertEqual(len(c.compare({"new.py": 1}, {})), 1)

    def test_baseline_roundtrip(self):
        self.assertEqual(c.read_baseline("# h\nx.py 3\ny/z.py 1\n"), {"x.py": 3, "y/z.py": 1})

    def test_the_real_tree_is_within_baseline(self):
        self.assertEqual(c.main([]), 0)


if __name__ == "__main__":
    unittest.main()

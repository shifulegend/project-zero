"""Tests for tools/check_struct_fields.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_struct_fields as c


class StructFieldsTest(unittest.TestCase):
    """Field counting and the ratchet."""

    def test_counts_plain_fields_arrays_and_multi_declarations(self) -> None:
        """Arrays count once, `int a, b;` counts two, nested structs count as one field."""
        body = "int a; char *b[4]; float x, y; struct { int p; int q; } n; void (*cb)(int, int);"
        self.assertEqual(c.count_fields(body), 6)

    def test_a_wide_new_struct_fails(self) -> None:
        """An unlisted struct over the limit is a problem."""
        self.assertEqual(len(c.problems({"include/a.h::A": 26}, {})), 1)

    def test_a_listed_struct_may_stay_or_shrink_but_not_grow(self) -> None:
        """A listed struct keeps or lowers its count, never raises it."""
        base = {"include/a.h::A": 40}
        self.assertEqual(c.problems({"include/a.h::A": 40}, base), [])
        self.assertEqual(c.problems({"include/a.h::A": 30}, base), [])
        self.assertEqual(len(c.problems({"include/a.h::A": 41}, base)), 1)

    def test_structs_at_the_limit_pass(self) -> None:
        """Exactly 25 fields is allowed."""
        self.assertEqual(c.problems({"include/a.h::A": 25}, {}), [])


if __name__ == "__main__":
    unittest.main()

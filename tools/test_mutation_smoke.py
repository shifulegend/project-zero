"""Tests for tools/mutation_smoke.py (the pure parts: finding and applying mutations)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mutation_smoke as m


class SiteTest(unittest.TestCase):
    """Which operators can be mutated."""

    def test_operators_in_code_are_found(self) -> None:
        """Comparison and logical operators are sites."""
        text = "if (a < b && c == d) return a >= b;\n"
        self.assertEqual([op for _, op in m.sites(text)], ["<", "&&", "==", ">="])

    def test_comments_and_strings_are_not_sites(self) -> None:
        """Operators in comments and string literals are ignored."""
        text = '/* a < b */ // c == d\nconst char *s = "x && y";\n'
        self.assertEqual(m.sites(text), [])

    def test_shifts_arrows_and_templates_are_not_sites(self) -> None:
        """<<, >>, -> and <= written as part of larger tokens are not split."""
        self.assertEqual([op for _, op in m.sites("x = a << 2; y = p->q; z = a >> 1;")], [])


class MutateTest(unittest.TestCase):
    """Applying a mutation."""

    def test_the_operator_is_flipped(self) -> None:
        """Exactly one operator changes."""
        text = "if (a < b && c == d) {}"
        offset, op = m.sites(text)[0]
        self.assertEqual(m.mutate(text, offset, op), "if (a <= b && c == d) {}")


if __name__ == "__main__":
    unittest.main()

"""Tests for tools/check_duplication.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_duplication as dup

BLOCK = [f"statement_{i}();" for i in range(8)]


class NormaliseTest(unittest.TestCase):
    """What counts as a code line."""

    def test_blank_comment_and_brace_lines_are_dropped(self) -> None:
        """Only real code survives, with collapsed whitespace."""
        text = "int  a;\n\n// note\n/* block\n still */\n{\n}\nint b;\n"
        self.assertEqual(dup.normalised(text), ["int a;", "int b;"])


class ShareTest(unittest.TestCase):
    """The duplicated share."""

    def test_two_copies_are_fully_covered(self) -> None:
        """The same block in two files is all duplicate."""
        self.assertEqual(dup.duplicated_share({"a": BLOCK, "b": BLOCK}), (16, 16))

    def test_distinct_code_is_not_duplicate(self) -> None:
        """Different blocks cover nothing."""
        other = [f"other_{i}();" for i in range(8)]
        self.assertEqual(dup.duplicated_share({"a": BLOCK, "b": other}), (0, 16))

    def test_short_repeats_are_ignored(self) -> None:
        """A repeat shorter than the window does not count."""
        short = BLOCK[:3]
        self.assertEqual(dup.duplicated_share({"a": short + ["x1();"] * 0, "b": short}), (0, 6))


class RealTreeTest(unittest.TestCase):
    """The committed baseline matches the tree."""

    def test_tree_is_within_the_baseline(self) -> None:
        """No scope is above its recorded share."""
        self.assertEqual(dup.main(["check_duplication.py"]), 0)


if __name__ == "__main__":
    unittest.main()

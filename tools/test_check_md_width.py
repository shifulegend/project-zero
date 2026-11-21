"""Tests for tools/check_md_width.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_md_width as width

LONG = "word " * 20  # 100 columns


class AddedLinesTest(unittest.TestCase):
    """Reading the added line numbers out of a -U0 diff."""

    def test_hunks_become_line_sets(self) -> None:
        """A hunk with a count covers that many lines; no count means one line."""
        diff = "+++ b/a.md\n@@ -1,0 +3,2 @@\n+x\n+y\n@@ -9 +20 @@\n+z\n"
        self.assertEqual(width.added_lines(diff), {"a.md": {3, 4, 20}})


class TooWideTest(unittest.TestCase):
    """Which added lines break the rule."""

    def test_prose_over_the_limit_is_reported(self) -> None:
        """A long added prose line is reported with its width."""
        self.assertEqual(width.too_wide("a.md", [LONG.rstrip()], {1}), [(1, 99)])

    def test_untouched_lines_are_left_alone(self) -> None:
        """Existing long lines are not reported."""
        self.assertEqual(width.too_wide("a.md", [LONG], set()), [])

    def test_exempt_kinds(self) -> None:
        """Tables, links, headings, HTML and fenced code are exempt."""
        lines = [
            "| " + LONG + " |",
            "see https://example.com/" + LONG,
            "[a](b) " + LONG,
            "# " + LONG,
            "<p>" + LONG + "</p>",
            "```",
            LONG,
            "```",
        ]
        self.assertEqual(width.too_wide("a.md", lines, set(range(1, 9))), [])

    def test_append_only_logs_are_exempt(self) -> None:
        """The changelog and the trace files keep one entry per line."""
        self.assertEqual(width.too_wide("CHANGELOG.md", [LONG], {1}), [])


if __name__ == "__main__":
    unittest.main()

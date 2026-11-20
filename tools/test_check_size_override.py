"""Tests for check size override."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_size_override as c

GOOD = "## Summary\nx\n\n## Size override justification\nAn overall clean-up of the code: many mechanical moves that cannot be split without breaking the build.\n\n## Testing\n"


class SizeOverrideTests(unittest.TestCase):
    """Size Override Tests."""

    def test_a_real_justification_passes(self) -> None:
        """A real justification passes."""
        self.assertEqual(c.problems(GOOD), [])

    def test_missing_section_fails(self) -> None:
        """Missing section fails."""
        self.assertEqual(len(c.problems("## Summary\nonly this")), 1)
        self.assertEqual(len(c.problems("")), 1)

    def test_comment_only_fails(self) -> None:
        """Comment only fails."""
        body = (
            "## Size override justification\n<!-- explain why this cannot be split -->\n## Next\n"
        )
        self.assertEqual(len(c.problems(body)), 1)

    def test_placeholder_and_short_text_fail(self) -> None:
        """Placeholder and short text fail."""
        self.assertIn("placeholder", c.problems("## Size override justification\nTBD\n")[0])
        self.assertIn("shorter", c.problems("## Size override justification\nbig refactor\n")[0])

    def test_section_ends_at_the_next_heading(self) -> None:
        """Section ends at the next heading."""
        body = "## Size override justification\nshort\n## Testing\n" + "long text " * 20
        self.assertEqual(len(c.problems(body)), 1)

    def test_heading_level_and_case_are_flexible(self) -> None:
        """Heading level and case are flexible."""
        body = "### size-override JUSTIFICATION\n" + "because this is one mechanical sweep " * 2
        self.assertEqual(c.problems(body), [])


if __name__ == "__main__":
    unittest.main()

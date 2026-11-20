import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_size_override as c

GOOD = "## Summary\nx\n\n## Size override justification\nAn overall clean-up of the code: many mechanical moves that cannot be split without breaking the build.\n\n## Testing\n"


class SizeOverrideTests(unittest.TestCase):
    def test_a_real_justification_passes(self):
        self.assertEqual(c.problems(GOOD), [])

    def test_missing_section_fails(self):
        self.assertEqual(len(c.problems("## Summary\nonly this")), 1)
        self.assertEqual(len(c.problems("")), 1)

    def test_comment_only_fails(self):
        body = (
            "## Size override justification\n<!-- explain why this cannot be split -->\n## Next\n"
        )
        self.assertEqual(len(c.problems(body)), 1)

    def test_placeholder_and_short_text_fail(self):
        self.assertIn("placeholder", c.problems("## Size override justification\nTBD\n")[0])
        self.assertIn("shorter", c.problems("## Size override justification\nbig refactor\n")[0])

    def test_section_ends_at_the_next_heading(self):
        body = "## Size override justification\nshort\n## Testing\n" + "long text " * 20
        self.assertEqual(len(c.problems(body)), 1)

    def test_heading_level_and_case_are_flexible(self):
        body = "### size-override JUSTIFICATION\n" + "because this is one mechanical sweep " * 2
        self.assertEqual(c.problems(body), [])


if __name__ == "__main__":
    unittest.main()

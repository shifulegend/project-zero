#!/usr/bin/env python3
"""Unit tests for tools/changelog.py (run: python3 tools/test_changelog.py)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import changelog  # noqa: E402

GOOD = """# Changelog

## [Unreleased]

### Added

- new thing

## [0.2.0] - 2026-11-01

### Changed

- renamed

## [0.1.0] - 2026-06-20

### Fixed

- bug

[Unreleased]: https://x/compare/v0.2.0...HEAD
[0.2.0]: https://x/compare/v0.1.0...v0.2.0
[0.1.0]: https://x/releases/tag/v0.1.0
"""


class Check(unittest.TestCase):
    def test_good(self):
        self.assertEqual(changelog.check(GOOD, require_unreleased=True), [])

    def test_repo_changelog_is_valid(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CHANGELOG.md")
        with open(root, encoding="utf-8") as f:
            self.assertEqual(changelog.check(f.read(), require_unreleased=True), [])

    def test_unreleased_must_be_first_and_unique(self):
        t = GOOD.replace("## [Unreleased]\n", "", 1).replace(
            "## [0.1.0]", "## [Unreleased]\n\n## [0.1.0]")
        self.assertTrue(any("first" in e for e in changelog.check(t)))
        t2 = GOOD.replace("## [0.2.0] - 2026-11-01", "## [Unreleased]")
        self.assertTrue(any("more than one" in e for e in changelog.check(t2)))

    def test_order_dates_duplicates(self):
        t = GOOD.replace("[0.2.0] - 2026-11-01", "[0.0.9] - 2026-11-01").replace(
            "[0.2.0]: ", "[0.0.9]: ")
        self.assertTrue(any("newest first" in e for e in changelog.check(t)))
        t = GOOD.replace("[0.1.0] - 2026-06-20", "[0.1.0] - 2026-12-20")
        self.assertTrue(any("later than" in e for e in changelog.check(t)))
        t = GOOD.replace("[0.1.0] - 2026-06-20", "[0.2.0] - 2026-06-20")
        self.assertTrue(any("duplicate" in e for e in changelog.check(t)))

    def test_bad_heading_subsection_link(self):
        self.assertTrue(changelog.check(GOOD.replace("[0.1.0] - 2026-06-20", "0.1.0 (June)")))
        self.assertTrue(any("subsection" in e for e in
                            changelog.check(GOOD.replace("### Fixed", "### Bugs"))))
        self.assertTrue(any("link reference" in e for e in
                            changelog.check(GOOD.replace("[0.1.0]: https://x/releases/tag/v0.1.0\n", ""))))

    def test_prerelease_sorts_below_release(self):
        t = GOOD.replace("[0.2.0] - 2026-11-01", "[0.2.0-rc.1] - 2026-11-01").replace(
            "[0.2.0]: ", "[0.2.0-rc.1]: ")
        self.assertEqual(changelog.check(t), [])


class SectionsAndYanked(unittest.TestCase):
    def test_section_order_and_repeat(self):
        t = GOOD.replace("### Added\n\n- new thing\n",
                         "### Added\n\n- new thing\n\n### Changed\n\n- x\n")
        self.assertTrue(any("out of order" in e for e in changelog.check(t)))
        t = GOOD.replace("### Added\n\n- new thing\n",
                         "### Added\n\n- a\n\n### Added\n\n- b\n")
        self.assertTrue(any("repeated" in e for e in changelog.check(t)))

    def test_yanked_heading(self):
        t = GOOD.replace("[0.1.0] - 2026-06-20", "[0.1.0] - 2026-06-20 [YANKED]")
        self.assertEqual(changelog.check(t), [])
        self.assertIn("- bug", changelog.notes(t, "0.1.0"))


class Notes(unittest.TestCase):
    def test_extract_with_v_prefix_and_link(self):
        out = changelog.notes(GOOD, "v0.2.0")
        self.assertIn("### Changed", out)
        self.assertIn("- renamed", out)
        self.assertNotIn("## [0.2.0]", out)
        self.assertNotIn("0.1.0", out.split("**Full changes:**")[0])
        self.assertIn("https://x/compare/v0.1.0...v0.2.0", out)

    def test_unreleased_and_missing(self):
        self.assertIn("new thing", changelog.notes(GOOD, "Unreleased"))
        self.assertIsNone(changelog.notes(GOOD, "9.9.9"))
        empty = GOOD.replace("### Changed\n\n- renamed\n", "")
        self.assertIsNone(changelog.notes(empty, "0.2.0"))


if __name__ == "__main__":
    unittest.main(verbosity=1)

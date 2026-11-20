"""Tests for tools/check_md_fences.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_md_fences as c


class FenceTest(unittest.TestCase):
    def test_tagged_fence_passes(self) -> None:
        self.assertEqual(c.untagged("```bash\nls\n```".split("\n")), [])

    def test_untagged_opening_fence_is_found_closing_fence_is_not(self) -> None:
        self.assertEqual(c.untagged("a\n```\nx\n```\nb".split("\n")), [1])

    def test_tilde_and_indented_fences(self) -> None:
        self.assertEqual(c.untagged("~~~\nx\n~~~".split("\n")), [0])
        self.assertEqual(c.untagged("  ```\n  x\n  ```".split("\n")), [0])

    def test_inner_backticks_inside_a_longer_fence_do_not_open_a_block(self) -> None:
        text = "````md\n```\ninner\n```\n````\n```\nreal\n```"
        self.assertEqual(c.untagged(text.split("\n")), [5])

    def test_fix_tags_only_untagged_openers(self) -> None:
        out = c.fix_text("```\nx\n```\n```c\ny\n```")
        self.assertEqual(out, "```text\nx\n```\n```c\ny\n```")


if __name__ == "__main__":
    unittest.main()

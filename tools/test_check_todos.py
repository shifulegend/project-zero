"""Tests for tools/check_todos.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_todos as c

REG = {"PZ-TODO-1", "QA-ISS-005"}


class TodoTest(unittest.TestCase):
    def test_registered_id_passes(self) -> None:
        self.assertEqual(c.problems("// TODO(PZ-TODO-1): later\n", "a.c", REG), [])
        self.assertEqual(c.problems("/* TODO(QA-ISS-005) */\n", "a.h", REG), [])

    def test_missing_id_is_reported_with_line_number(self) -> None:
        out = c.problems("int x;\n// TODO fix this\n", "a.c", REG)
        self.assertEqual(len(out), 1)
        self.assertIn("a.c:2", out[0])

    def test_unregistered_or_malformed_id_is_reported(self) -> None:
        self.assertTrue(c.problems("// TODO(PZ-TODO-9): x\n", "a.c", REG))
        self.assertTrue(c.problems("// FIXME(bob): x\n", "a.c", REG))
        self.assertTrue(c.problems("# FIXME\n", "a.py", REG))

    def test_register_parsing(self) -> None:
        text = "| ID |\n|---|\n| PZ-TODO-1 | where | what | why |\n| QA-ISS-005 | x | y | z |\n"
        self.assertEqual(c.registered_ids(text), REG)

    def test_the_real_tree_is_clean(self) -> None:
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()

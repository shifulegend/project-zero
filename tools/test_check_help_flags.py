import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_help_flags as c


class HelpFlagTests(unittest.TestCase):
    def test_extracts_parsed_flags(self) -> None:
        text = 'if (strcmp(argv[i], "--model") == 0) {} else if (strcmp(argv[i], "--top-k") == 0)'
        self.assertEqual(c.parsed_flags(text), {"--model", "--top-k"})

    def test_extracts_documented_flags(self) -> None:
        self.assertEqual(
            c.documented_flags('printf("  --model <f>  x\\n  --top-k <n>");'),
            {"--model", "--top-k"},
        )

    def test_reports_the_missing_ones_sorted(self) -> None:
        self.assertEqual(c.undocumented({"--b", "--a", "--c"}, {"--c"}), ["--a", "--b"])

    def test_the_real_tree_is_complete(self) -> None:
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()

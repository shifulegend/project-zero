"""Tests for tools/check_pr_size.py."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import check_pr_size as c


def run(numstat: str, *extra: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "n.txt"
        f.write_text(numstat)
        return subprocess.run(
            [
                sys.executable,
                "-I",
                str(TOOLS / "check_pr_size.py"),
                "origin/master",
                "--numstat",
                str(f),
                *extra,
            ],
            capture_output=True,
            text=True,
        )


class PrSizeTest(unittest.TestCase):
    def test_counts_additions_and_deletions(self) -> None:
        total, per = c.count("10\t5\tsrc/a.c\n3\t0\tsrc/b.c\n")
        self.assertEqual(total, 18)
        self.assertEqual(per[0], (15, "src/a.c"))

    def test_generated_vendored_and_data_files_are_ignored(self) -> None:
        text = "5000\t10\tweb/package-lock.json\n9000\t0\tsrc/api/webui_bundle_generated.c\n7\t3\tsrc/a.c\n-\t-\timg.png\n800\t0\tbenchmark_results/x/y.txt\n"
        self.assertEqual(c.count(text)[0], 10)

    def test_at_the_limit_passes_and_over_fails(self) -> None:
        self.assertEqual(run("200\t200\tsrc/a.c\n").returncode, 0)
        r = run("200\t201\tsrc/a.c\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("size-override", r.stdout)
        self.assertIn("src/a.c", r.stdout)

    def test_custom_limit(self) -> None:
        self.assertEqual(run("50\t0\tsrc/a.c\n", "--limit", "10").returncode, 1)


if __name__ == "__main__":
    unittest.main()

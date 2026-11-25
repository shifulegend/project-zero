"""Tests for tools/check_raw_logging.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_raw_logging as c


class RawLoggingTest(unittest.TestCase):
    """Tests for raw logging."""

    def test_flags_raw_stderr_writes(self) -> None:
        """Flags raw stderr writes."""
        self.assertEqual(c.raw_uses('int f(void) {\n  fprintf(stderr, "x\\n");\n}\n'), [2])
        self.assertEqual(c.raw_uses('perror("open");'), [1])
        self.assertEqual(c.raw_uses("fputs(msg, stderr);"), [1])
        self.assertEqual(c.raw_uses("fwrite(b, 1, n, stderr);"), [1])
        self.assertEqual(c.raw_uses("fflush(stderr);"), [])

    def test_logger_calls_and_comments_are_fine(self) -> None:
        """Logger calls and comments are fine."""
        self.assertEqual(c.raw_uses('tn_log(TN_LOG_ERROR, "x", "bad\\n");'), [])
        self.assertEqual(c.raw_uses('/* fprintf(stderr, "old") */'), [])
        self.assertEqual(c.raw_uses('// fprintf(stderr, "old")'), [])

    def test_the_real_tree_is_clean(self) -> None:
        """The real tree is clean."""
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for tools/check_socket_writes.py."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_socket_writes as c


class SocketWriteTest(unittest.TestCase):
    """Finding direct writes."""

    def test_direct_calls_are_found(self) -> None:
        """write, send and fwrite calls are reported with their line."""
        text = (
            "int f(void) {\n  write(fd, b, 1);\n  send(fd, b, 1, 0);\n  fwrite(b, 1, 1, fp);\n}\n"
        )
        self.assertEqual(c.violations(text), [2, 3, 4])

    def test_net_write_all_and_comments_are_fine(self) -> None:
        """net_write_all, member calls, comments and strings are not direct writes."""
        text = 'net_write_all(fd, b, 1);\nx.write(1);\n/* write(fd) */\nlog("write(fd)");\n'
        self.assertEqual(c.violations(text), [])

    def test_real_tree_is_clean(self) -> None:
        """No file under src/api/ writes directly except net_io.c."""
        self.assertEqual(c.main(), 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for tools/check_header_guards.py."""

import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import check_header_guards as c

REL = pathlib.PurePosixPath("api/net_io.h")


class GuardTest(unittest.TestCase):
    def test_expected_name_comes_from_the_path(self) -> None:
        self.assertEqual(c.expected(REL), "TN_API_NET_IO_H")
        self.assertEqual(c.expected(pathlib.PurePosixPath("core/gguf_i2s.h")), "TN_CORE_GGUF_I2S_H")

    def test_conforming_header_passes(self) -> None:
        self.assertEqual(
            c.check("#ifndef TN_API_NET_IO_H\n#define TN_API_NET_IO_H\nint x;\n#endif\n", REL), []
        )
        self.assertEqual(
            c.check("#ifndef TN_NET_H\n#define TN_NET_H\n#endif\n", REL), []
        )  # any TN_..._H is fine

    def test_wrong_prefix_pragma_once_and_missing_guard_are_reported(self) -> None:
        self.assertTrue(c.check("#ifndef NET_IO_H\n#define NET_IO_H\n#endif\n", REL))
        self.assertTrue(c.check("#ifndef PZ_NET_IO_H\n#define PZ_NET_IO_H\n#endif\n", REL))
        self.assertTrue(any("pragma once" in p for p in c.check("#pragma once\nint x;\n", REL)))
        self.assertTrue(any("no #ifndef" in p for p in c.check("int x;\n", REL)))
        self.assertTrue(
            any("does not match" in p for p in c.check("#ifndef A_H\n#define B_H\n#endif\n", REL))
        )

    def test_fix_renames_a_guard_everywhere_and_converts_pragma_once(self) -> None:
        fixed = c.fix("#ifndef NET_IO_H\n#define NET_IO_H\nint x;\n#endif /* NET_IO_H */\n", REL)
        self.assertEqual(c.check(fixed, REL), [])
        self.assertNotIn("NET_IO_H\n", fixed.replace("TN_API_NET_IO_H", ""))
        conv = c.fix("#pragma once\nint x;\n", REL)
        self.assertEqual(c.check(conv, REL), [])
        self.assertTrue(conv.rstrip().endswith("#endif /* TN_API_NET_IO_H */"))


if __name__ == "__main__":
    unittest.main()

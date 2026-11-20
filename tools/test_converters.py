"""The checkpoint converters can list the tensors of a safetensors file (skipped without safetensors)."""

import importlib.util
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HAVE = (
    importlib.util.find_spec("safetensors") is not None
    and importlib.util.find_spec("numpy") is not None
)
if HAVE:
    import convert_hf_bitnet
    import numpy as np
    from safetensors.numpy import save_file


@unittest.skipUnless(HAVE, "safetensors and numpy are not installed")
class ListTensorsTest(unittest.TestCase):
    """Regression: iterating a safe_open handle directly fails; .keys() is required."""

    def test_bitnet_converter_maps_every_tensor_to_its_file(self) -> None:
        """get_tensor_map lists the tensors of each .safetensors file in the directory."""
        with tempfile.TemporaryDirectory() as d:
            save_file({"a": np.zeros((2, 2), np.float32)}, f"{d}/one.safetensors")
            save_file({"b": np.ones(3, np.float32)}, f"{d}/two.safetensors")
            tmap = convert_hf_bitnet.get_tensor_map(d)
            self.assertEqual(sorted(tmap), ["a", "b"])
            self.assertTrue(tmap["a"].endswith("one.safetensors"))


if __name__ == "__main__":
    unittest.main()

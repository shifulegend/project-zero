"""ModelAdapter must be abstract: a subclass that forgets a method cannot be created."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    import numpy  # noqa: F401

    import extract_multimodal as em  # noqa: E402
except ImportError:  # numpy missing: nothing to test here
    em = None


@unittest.skipIf(em is None, "numpy is not installed")
class AdapterTest(unittest.TestCase):
    def test_the_base_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            em.ModelAdapter({}, {})

    def test_a_subclass_missing_a_method_cannot_be_instantiated(self):
        class Partial(em.ModelAdapter):
            def vision_config(self):
                return (1, 1, 1, 1, 1)

        with self.assertRaises(TypeError):
            Partial({}, {})

    def test_a_complete_subclass_can(self):
        class Full(em.ModelAdapter):
            def vision_config(self):
                return (1, 1, 1, 1, 1)

            def write_vision_weights(self, f):
                pass

            def write_proj_weights(self, f):
                pass

        self.assertEqual(Full({}, {}).vision_config(), (1, 1, 1, 1, 1))

    def test_the_real_adapters_are_complete(self):
        for cls in (em.Moondream2Adapter, em.SmolVLMAdapter, em.LLaVAAdapter):
            self.assertEqual(
                getattr(cls, "__abstractmethods__", frozenset()), frozenset(), cls.__name__
            )
            cls({}, {})  # instantiable


if __name__ == "__main__":
    unittest.main()

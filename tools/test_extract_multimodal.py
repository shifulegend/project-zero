"""ModelAdapter must be abstract: a subclass that forgets a method cannot be created."""

import importlib.util
import pathlib
import sys
import unittest
from typing import Any, BinaryIO

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None  # without it there is nothing to test
if HAVE_NUMPY:
    import extract_multimodal as em


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class AdapterTest(unittest.TestCase):
    """Adapters are abstract classes with three methods to fill in."""

    def test_the_base_class_cannot_be_instantiated(self) -> None:
        """The abstract base refuses to be created."""
        base: Any = em.ModelAdapter  # abstract: only a runtime call can show it is refused
        with self.assertRaises(TypeError):
            base({}, {})

    def test_a_subclass_missing_a_method_cannot_be_instantiated(self) -> None:
        """A subclass that leaves two methods out is still abstract."""

        class Partial(em.ModelAdapter):
            def vision_config(self) -> em.VisionConfig:
                return (1, 1, 1, 1, 1)

        partial: Any = Partial
        with self.assertRaises(TypeError):
            partial({}, {})

    def test_a_complete_subclass_can(self) -> None:
        """A subclass with all three methods can be created."""

        class Full(em.ModelAdapter):
            def vision_config(self) -> em.VisionConfig:
                return (1, 1, 1, 1, 1)

            def write_vision_weights(self, f: BinaryIO) -> None:
                pass

            def write_proj_weights(self, f: BinaryIO) -> None:
                pass

        self.assertEqual(Full({}, {}).vision_config(), (1, 1, 1, 1, 1))

    def test_the_real_adapters_are_complete(self) -> None:
        """Every adapter shipped in extract_multimodal can be created."""
        for cls in (em.Moondream2Adapter, em.SmolVLMAdapter, em.LLaVAAdapter):
            self.assertEqual(
                getattr(cls, "__abstractmethods__", frozenset()), frozenset(), cls.__name__
            )
            cls({}, {})  # instantiable


if __name__ == "__main__":
    unittest.main()

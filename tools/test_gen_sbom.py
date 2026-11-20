"""Tests for tools/gen_sbom.py."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import gen_sbom


class SbomTest(unittest.TestCase):
    def test_structure(self) -> None:
        bom = gen_sbom.build("0.3.0")
        self.assertEqual(bom["bomFormat"], "CycloneDX")
        self.assertEqual(bom["specVersion"], "1.5")
        self.assertTrue(bom["serialNumber"].startswith("urn:uuid:"))
        self.assertEqual(bom["metadata"]["component"]["version"], "0.3.0")

    def test_vendored_versions_come_from_the_headers(self) -> None:
        names = {c["name"]: c["version"] for c in gen_sbom.build("1")["components"]}
        self.assertEqual(set(names), {"stb_image", "stb_image_resize2"})
        for v in names.values():
            self.assertRegex(v, r"^\d+\.\d+$")

    def test_dependencies_reference_existing_components(self) -> None:
        bom = gen_sbom.build("1")
        refs = {c["bom-ref"] for c in bom["components"]}
        deps = bom["dependencies"][0]
        self.assertEqual(set(deps["dependsOn"]), refs)

    def test_cli_strips_v_and_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "sbom.json"
            subprocess.run(
                [sys.executable, "-I", str(TOOLS / "gen_sbom.py"), "v9.8.7", str(out)], check=True
            )
            bom = json.loads(out.read_text())
            self.assertEqual(bom["metadata"]["component"]["version"], "9.8.7")

    def test_the_licence_is_declared(self) -> None:
        lic = {x["license"]["id"] for x in gen_sbom.build("1")["metadata"]["component"]["licenses"]}
        self.assertEqual(lic, {"MIT"})


if __name__ == "__main__":
    unittest.main()

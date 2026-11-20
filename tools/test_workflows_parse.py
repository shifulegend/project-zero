"""Every GitHub workflow and config file under .github/ must be valid YAML.

A plain-scalar step name containing ': ' once made ci.yml unparseable, which GitHub reports only
after the push; this catches it locally and in the unit-test job.
"""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


class WorkflowYamlTest(unittest.TestCase):
    """Parse each YAML file under .github/."""

    def test_every_workflow_parses(self) -> None:
        """No workflow or dependabot file has a YAML syntax error."""
        files = sorted((ROOT / ".github").rglob("*.yml")) + sorted(
            (ROOT / ".github").rglob("*.yaml")
        )
        self.assertTrue(files)
        for path in files:
            with (
                self.subTest(path=str(path.relative_to(ROOT))),
                path.open(encoding="utf-8") as handle,
            ):
                yaml.safe_load(handle)


if __name__ == "__main__":
    unittest.main()

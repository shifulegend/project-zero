#!/usr/bin/env python3
"""Write a CycloneDX 1.5 SBOM (JSON) for a Project Zero release.

The engine has no package-manager dependencies: it is C with libm and pthreads, plus two vendored
single-file libraries (include/stb_image.h, include/stb_image_resize2.h) whose versions are read
from their headers. The version of Project Zero itself is the argument.

usage: tools/gen_sbom.py VERSION [OUTPUT]      (OUTPUT default: stdout)
"""

import json
import pathlib
import re
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent

VENDORED = [
    (
        "stb_image",
        "include/stb_image.h",
        r"stb_image - v([0-9.]+)",
        "https://github.com/nothings/stb",
    ),
    (
        "stb_image_resize2",
        "include/stb_image_resize2.h",
        r"stb_image_resize2 - v([0-9.]+)",
        "https://github.com/nothings/stb",
    ),
]


def vendored_components():
    out = []
    for name, rel, pattern, url in VENDORED:
        text = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        m = re.search(pattern, text)
        if not m:
            sys.exit(f"{rel}: cannot find the version with {pattern!r}")
        out.append(
            {
                "type": "library",
                "bom-ref": f"{name}@{m.group(1)}",
                "name": name,
                "version": m.group(1),
                "licenses": [{"license": {"id": "MIT"}}],
                "description": "vendored single-file library",
                "externalReferences": [{"type": "website", "url": url}],
                "evidence": {"occurrences": [{"location": rel}]},
            }
        )
    return out


def build(version):
    comps = vendored_components()
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid.uuid4()}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": f"project-zero@{version}",
                "name": "project-zero",
                "version": version,
                "licenses": [{"license": {"id": "MIT"}}, {"license": {"id": "LGPL-3.0-or-later"}}],
                "description": "MIT, except src/math/ternary_matmul_lut_avx512bw.c (LGPL-3.0-or-later)",
            }
        },
        "components": comps,
        "dependencies": [
            {"ref": f"project-zero@{version}", "dependsOn": [c["bom-ref"] for c in comps]}
        ],
    }


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    text = json.dumps(build(argv[1].lstrip("v")), indent=2) + "\n"
    if len(argv) > 2:
        pathlib.Path(argv[2]).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

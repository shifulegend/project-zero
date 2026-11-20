#!/usr/bin/env python3
"""Move old raw benchmark results out of the repository tree into one release archive.

The latest published sweep stays in benchmark_results/ without its terminal screenshots (the
README, the summary CSV, the gate file, the charts and the raw logs are kept, about 5 MB); every
other file under benchmark_results/ is archived. The git history is not rewritten, so old
commits still have the files.

  tools/archive_benchmark_results.py --list                 what stays and what goes, with sizes
  tools/archive_benchmark_results.py --archive OUT.tar.gz   write the archive and print its SHA-256
  tools/archive_benchmark_results.py --remove               git rm the archived files (run it only
                                                            after the archive is uploaded as a
                                                            GitHub release asset)
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = "benchmark_results"
KEEP_RUN = "h2h_20261118-053147Z_published-sweep"
KEEP_TOP = ("README.md",)
SKIP_IN_KEPT_RUN = ("screenshots/",)


def keep(path: str) -> bool:
    """True for a tracked file that stays in the tree."""
    rel = path[len(RESULTS) + 1 :]
    if rel in KEEP_TOP:
        return True
    prefix = KEEP_RUN + "/"
    if rel.startswith(prefix):
        return not any(rel[len(prefix) :].startswith(skip) for skip in SKIP_IN_KEPT_RUN)
    return False


def tracked() -> list[str]:
    """Tracked files under benchmark_results/."""
    out = subprocess.run(
        ["git", "ls-files", RESULTS], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.split("\n") if line]


def split(files: list[str]) -> tuple[list[str], list[str]]:
    """(kept, archived) partition of the tracked files."""
    kept = [f for f in files if keep(f)]
    return kept, [f for f in files if not keep(f)]


def size(files: list[str]) -> int:
    """Total bytes of the files that exist in the working tree."""
    return sum((ROOT / f).stat().st_size for f in files if (ROOT / f).is_file())


def main(argv: list[str]) -> int:
    """Run the requested action."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--archive", metavar="OUT.tar.gz")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args(argv)
    kept, archived = split(tracked())
    print(
        f"keep {len(kept)} files ({size(kept) / 1e6:.1f} MB), archive {len(archived)} files ({size(archived) / 1e6:.1f} MB)"
    )
    if args.archive:
        with tarfile.open(args.archive, "w:gz") as tar:
            for name in archived:
                if (ROOT / name).is_file():
                    tar.add(ROOT / name, arcname=name)
        digest = hashlib.sha256(Path(args.archive).read_bytes()).hexdigest()
        print(f"wrote {args.archive} sha256 {digest}")
    if args.remove:
        subprocess.run(["git", "rm", "-q", "--", *archived], cwd=ROOT, check=True)
        print(f"removed {len(archived)} files from the tree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

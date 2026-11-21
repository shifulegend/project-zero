#!/usr/bin/env python3
"""Fails when the HTTP routes the server handles and the routes the OpenAPI document describes differ.

The server's route table (k_routes in src/api/http_connection.c) has one `{"M", "/p", handler}` row
per route; the document (src/api/openapi.c) lists `"/p":{"m":...`. A route that
is served but not documented is invisible to API users, and a documented route that is not served
is a lie. `/` is the web UI, served by the static-asset fallback rather than an explicit match.

usage: tools/check_openapi_routes.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SERVER = ROOT / "src" / "api" / "http_connection.c"
OPENAPI = ROOT / "src" / "api" / "openapi.c"

ROUTED = re.compile(r'\{\s*"(GET|POST|PUT|DELETE|PATCH|HEAD)"\s*,\s*"(/[^"]*)"\s*,\s*\w+\s*\}')
DOCUMENTED = re.compile(r'\\"(/[^"\\]*)\\":\{\\"(get|post|put|delete|patch)\\"')
FALLBACK = {("GET", "/")}  # documented, served by the static-asset fallback


def routed_routes(text: str) -> set[tuple[str, str]]:
    """(method, path) pairs in the route table of http_connection.c."""
    return {(m.upper(), p) for m, p in ROUTED.findall(text)}


def documented_routes(text: str) -> set[tuple[str, str]]:
    """(method, path) pairs the OpenAPI document describes."""
    return {(m.upper(), p) for p, m in DOCUMENTED.findall(text)}


def compare(routed: set[tuple[str, str]], documented: set[tuple[str, str]]) -> list[str]:
    """Returns a list of problems (empty when the two sets agree)."""
    problems: list[str] = []
    for method, path in sorted(routed - documented):
        problems.append(f"{method} {path} is served but missing from the OpenAPI document")
    for method, path in sorted(documented - routed - FALLBACK):
        problems.append(f"{method} {path} is in the OpenAPI document but not served")
    return problems


def main() -> int:
    """Entry point: compares the two sets and returns 1 on any difference."""
    routed = routed_routes(SERVER.read_text(encoding="utf-8"))
    documented = documented_routes(OPENAPI.read_text(encoding="utf-8"))
    if not routed or not documented:
        print("could not read the routes (did the router or document format change?)")
        return 2
    problems = compare(routed, documented)
    for p in problems:
        print(p)
    print(
        f"openapi routes: {len(routed)} served, {len(documented)} documented, "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

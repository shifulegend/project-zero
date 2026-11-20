"""Tests for check openapi routes."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_openapi_routes as c

SERVER = 'if (strcmp(req.method, "GET") == 0 &&\n strcmp(req.path, "/health") == 0) {} else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/x") == 0)'
DOC = r'"\"/health\":{\"get\":{}},\"/v1/x\":{\"post\":{}},\"/\":{\"get\":{}}"'


class RouteTests(unittest.TestCase):
    """Route Tests."""

    def test_extracts_both_sides(self) -> None:
        """Extracts both sides."""
        self.assertEqual(c.routed_routes(SERVER), {("GET", "/health"), ("POST", "/v1/x")})
        self.assertEqual(
            c.documented_routes(DOC), {("GET", "/health"), ("POST", "/v1/x"), ("GET", "/")}
        )

    def test_agreement_is_clean(self) -> None:
        """Agreement is clean."""
        self.assertEqual(c.compare(c.routed_routes(SERVER), c.documented_routes(DOC)), [])

    def test_served_but_undocumented(self) -> None:
        """Served but undocumented."""
        out = c.compare({("GET", "/a"), ("GET", "/b")}, {("GET", "/a")})
        self.assertEqual(out, ["GET /b is served but missing from the OpenAPI document"])

    def test_documented_but_not_served(self) -> None:
        """Documented but not served."""
        out = c.compare({("GET", "/a")}, {("GET", "/a"), ("POST", "/ghost")})
        self.assertEqual(out, ["POST /ghost is in the OpenAPI document but not served"])

    def test_root_is_served_by_the_fallback(self) -> None:
        """Root is served by the fallback."""
        self.assertEqual(c.compare({("GET", "/a")}, {("GET", "/a"), ("GET", "/")}), [])

    def test_the_real_tree_agrees(self) -> None:
        """The real tree agrees."""
        routed = c.routed_routes(c.SERVER.read_text(encoding="utf-8"))
        documented = c.documented_routes(c.OPENAPI.read_text(encoding="utf-8"))
        self.assertEqual(c.compare(routed, documented), [])


if __name__ == "__main__":
    unittest.main()

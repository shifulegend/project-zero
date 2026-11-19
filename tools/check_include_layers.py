#!/usr/bin/env python3
"""Fail when the #include graph between modules gets a NEW dependency cycle.

A module is a directory under include/ or src/ (core, math, api, ...). An edge
A -> B means a file of module A includes "B/...". A strongly connected component
(SCC) of two or more modules is a cycle: those modules cannot be understood,
built or tested apart. Every edge that lies inside a cycle is recorded in
tools/include_cycles_baseline.txt as "A->B". This check fails on a cycle edge
that is not in the baseline (new coupling inside the tangle, or a new cycle),
and says when baseline edges have disappeared. Fix cycles, then run with
--update to rewrite the baseline (it may only shrink).

usage: tools/check_include_layers.py [--update] [--self-test] [--graph]
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tools" / "include_cycles_baseline.txt"
INC = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def module_of(path):
    parts = path.relative_to(ROOT).parts
    return parts[1] if len(parts) > 2 and parts[0] in ("include", "src") else None


def build_graph(root=ROOT):
    mods = set()
    for top in ("include", "src"):
        d = root / top
        if d.is_dir():
            mods |= {p.name for p in d.iterdir() if p.is_dir()}
    edges = {m: set() for m in mods}
    for top in ("include", "src"):
        for f in (root / top).rglob("*"):
            if f.suffix not in (".h", ".c", ".cpp", ".hpp"):
                continue
            m = module_of(f)
            if m is None:
                continue
            for inc in INC.findall(f.read_text(encoding="utf-8", errors="replace")):
                head = inc.split("/", 1)[0]
                if "/" in inc and head in mods and head != m:
                    edges[m].add(head)
    return edges


def sccs(edges):
    """Tarjan; returns the components with more than one module."""
    index, low, on, stack, out, n = {}, {}, set(), [], [], [0]

    def visit(v):
        index[v] = low[v] = n[0]
        n[0] += 1
        stack.append(v)
        on.add(v)
        for w in sorted(edges.get(v, ())):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(frozenset(comp))

    sys.setrecursionlimit(10000)
    for v in sorted(edges):
        if v not in index:
            visit(v)
    return sorted(out, key=lambda c: sorted(c))


def cycle_edges(edges, comps):
    """Edges whose two ends are in the same cycle component, as 'A->B' strings."""
    out = set()
    for c in comps:
        for a in c:
            for b in edges.get(a, ()):
                if b in c:
                    out.add(f"{a}->{b}")
    return out


def read_baseline():
    if not BASELINE.exists():
        return set()
    return {
        l.strip() for l in BASELINE.read_text().splitlines() if l.strip() and not l.startswith("#")
    }


def compare(current, baseline):
    """current/baseline: sets of 'A->B'. Return (new_problems, shrink_notes)."""
    problems = [
        f"new include edge inside a dependency cycle: {e}" for e in sorted(current - baseline)
    ]
    notes = [f"baseline edge is gone (run --update): {e}" for e in sorted(baseline - current)]
    return problems, notes


def self_test():
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok = ok and cond

    g = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}, "e": set()}
    comps = sccs(g)
    check(
        comps == [frozenset("abc")], "a 3-module cycle is found, a tail into it is not part of it"
    )
    ce = cycle_edges(g, comps)
    check(ce == {"a->b", "b->c", "c->a"}, "only edges inside the cycle are reported (d->a is not)")
    check(sccs({"a": {"b"}, "b": set()}) == [], "an acyclic graph has no cycle")
    check(sccs({"a": {"a"}}) == [], "a self-edge is not a cycle between modules")
    p, n = compare(ce, ce)
    check(not p and not n, "cycle edges equal to the baseline pass")
    g2 = {"a": {"b", "c"}, "b": {"c"}, "c": {"a"}}
    p, _ = compare(cycle_edges(g2, sccs(g2)), ce)
    check(len(p) == 1 and "a->c" in p[0], "a new edge inside the cycle fails and is named")
    g3 = {"a": {"b"}, "b": {"a"}, "x": {"y"}, "y": {"x"}}
    p, _ = compare(cycle_edges(g3, sccs(g3)), ce)
    check(bool(p), "a brand-new cycle fails")
    g4 = {"a": {"b"}, "b": {"c"}, "c": set()}
    p, n = compare(cycle_edges(g4, sccs(g4)), ce)
    check(not p and len(n) == 3, "a fixed cycle passes and the vanished edges are listed")
    return 0 if ok else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()
    edges = build_graph()
    comps = sccs(edges)
    cur = cycle_edges(edges, comps)
    if "--graph" in argv:
        for m in sorted(edges):
            print(f"{m}: {', '.join(sorted(edges[m]))}")
    if "--update" in argv:
        BASELINE.write_text(
            "# Module include edges that lie inside a dependency cycle; this list may only shrink\n"
            "# (tools/check_include_layers.py).\n" + "".join(e + "\n" for e in sorted(cur))
        )
        print(f"baseline written: {len(cur)} cycle edge(s) in {len(comps)} cycle(s)")
        return 0
    problems, notes = compare(cur, read_baseline())
    for n in notes:
        print("note:", n)
    for p in problems:
        print("ERROR:", p)
    if problems:
        print(
            "Break the cycle (move the shared type down to a leaf header, or forward-declare it)."
        )
        return 1
    print(f"include cycles ok ({len(cur)} existing cycle edges, none new)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

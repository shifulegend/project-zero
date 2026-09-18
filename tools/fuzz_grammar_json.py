#!/usr/bin/env python3
"""
fuzz_grammar_json.py — TS-2.1: differential fuzzing of the JSON grammar PDA
(src/sampling/grammar_json.c) against Python's json.loads.

Generates candidate byte strings via three generators (uniform random,
random tokens from the JSON alphabet, valid-JSON-mutated-by-1-3-edits) plus
a prefix-truncation sweep of valid JSON documents, feeds them all through
tools/grammar_json_verdict (a batch harness driving the real production
json_grammar_step()/json_grammar_finalize()), and cross-checks each verdict
against json.loads.

    Our verdict      | json.loads | Meaning
    DONE              | parses     | agree
    DONE              | raises     | FALSE ACCEPT -- P0 bug (would emit invalid JSON)
    INVALID           | parses     | FALSE REJECT -- P0 bug (would block valid output)
    INCOMPLETE         | raises     | agree (valid prefix, incomplete)

Critical asymmetry (checked separately, see check_prefix_asymmetry): a
*prefix* our PDA rejects (INVALID) must never be a prefix of any valid JSON
-- i.e. truncating valid JSON at any offset must never produce INVALID.

Usage: tools/fuzz_grammar_json.py [--n N] [--seed S] [--harness PATH]
Builds the C harness on the fly if --harness is not given and no built copy
is found at the default cache path.
"""
import argparse
import base64
import json
import os
import random
import string
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HARNESS_SRC = os.path.join(REPO_ROOT, "tools", "grammar_json_verdict.c")
DEFAULT_HARNESS_BIN = "/tmp/grammar_json_verdict_fuzz"

JSON_ALPHABET = '{}[]",:0123456789abcdefghijklmnopqrstuvwxyz \t\n-+.eE\\'

VALID_SEEDS = [
    "null", "true", "false", "0", "-0", "42", "-17", "3.14", "1e10", "-2.5e-3",
    '""', '"hello"', '"a\\nb\\"c"', '"\\u0041"',
    "[]", "[1,2,3]", "[[1],[2,[3]]]", "[null,true,false]",
    "{}", '{"a":1}', '{"a":1,"b":[1,2,{"c":true}]}',
    '{"nested":{"deep":{"deeper":{"deepest":[1,2,3]}}}}',
    '[1, 2, 3.5, "x", null, true, false, {"k":"v"}]',
]


def build_harness(path):
    if os.path.exists(path):
        return path
    cc = os.environ.get("CC", "gcc")
    cmd = [cc, "-std=c99", "-Wall", "-Wextra", "-Iinclude",
           "-D_POSIX_C_SOURCE=200809L", "-O2",
           "-o", path,
           os.path.join(REPO_ROOT, "tools", "grammar_json_verdict.c"),
           os.path.join(REPO_ROOT, "src", "sampling", "grammar_json.c")]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return path


def gen_uniform_random(rng, n, maxlen=40):
    out = []
    for _ in range(n):
        length = rng.randint(0, maxlen)
        out.append(bytes(rng.randrange(256) for _ in range(length)))
    return out


def gen_alphabet_random(rng, n, maxlen=60):
    out = []
    for _ in range(n):
        length = rng.randint(0, maxlen)
        out.append("".join(rng.choice(JSON_ALPHABET) for _ in range(length)).encode())
    return out


def gen_mutated_valid(rng, n):
    out = []
    for _ in range(n):
        s = list(rng.choice(VALID_SEEDS))
        n_edits = rng.randint(1, 3)
        for _ in range(n_edits):
            if not s:
                s = list(rng.choice(VALID_SEEDS))
            op = rng.choice(("insert", "delete", "replace"))
            pos = rng.randrange(len(s) + 1) if op == "insert" else rng.randrange(len(s))
            if op == "insert":
                s.insert(pos, rng.choice(JSON_ALPHABET))
            elif op == "delete" and s:
                del s[pos]
            elif op == "replace" and s:
                s[pos] = rng.choice(JSON_ALPHABET)
        out.append("".join(s).encode())
    return out


def gen_prefix_truncations():
    """Every offset of every valid seed, plus the full string itself."""
    out = []
    for seed in VALID_SEEDS:
        b = seed.encode()
        for i in range(len(b) + 1):
            out.append(b[:i])
    return out


def run_harness(harness, candidates):
    """candidates: list[bytes]. Returns list[(verdict, invalid_offset, raw_bytes)]."""
    b64_lines = "\n".join(base64.b64encode(c).decode() for c in candidates) + "\n"
    proc = subprocess.run([harness], input=b64_lines, capture_output=True, text=True, check=True)
    results = []
    for line, original in zip(proc.stdout.splitlines(), candidates):
        parts = line.split("\t", 2)
        verdict, offset = parts[0], int(parts[1])
        results.append((verdict, offset, original))
    return results


def json_loads_verdict(raw):
    try:
        json.loads(raw.decode("utf-8", errors="surrogateescape"))
        return True
    except Exception:
        return False


def check_differential(results):
    false_accepts = []
    false_rejects = []
    for verdict, offset, raw in results:
        parses = json_loads_verdict(raw)
        if verdict == "DONE" and not parses:
            false_accepts.append((raw, offset))
        elif verdict == "INVALID" and parses:
            false_rejects.append((raw, offset))
    return false_accepts, false_rejects


def check_prefix_asymmetry(results):
    """Any INVALID verdict among prefix-truncation candidates is a bug:
    a prefix of valid JSON must never be flatly invalid (it should be
    INCOMPLETE or DONE, never a sink state)."""
    bad = []
    for verdict, offset, raw in results:
        if verdict == "INVALID":
            bad.append((raw, offset))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="candidates per generator (default 3000)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--harness", default=DEFAULT_HARNESS_BIN)
    args = ap.parse_args()

    harness = build_harness(args.harness)
    rng = random.Random(args.seed)

    print(f"=== TS-2.1 grammar differential fuzzing (seed={args.seed}, n={args.n}/generator) ===")

    uniform = gen_uniform_random(rng, args.n)
    alphabet = gen_alphabet_random(rng, args.n)
    mutated = gen_mutated_valid(rng, args.n)
    prefixes = gen_prefix_truncations()

    all_fail = False

    for label, candidates in (("uniform random", uniform),
                               ("JSON-alphabet random", alphabet),
                               ("mutated valid JSON", mutated)):
        results = run_harness(harness, candidates)
        false_accepts, false_rejects = check_differential(results)
        status = "PASS" if not false_accepts and not false_rejects else "FAIL"
        if status == "FAIL":
            all_fail = True
        print(f"  [{status}] {label}: {len(candidates)} candidates, "
              f"{len(false_accepts)} false-accepts, {len(false_rejects)} false-rejects")
        for raw, off in false_accepts[:5]:
            print(f"      FALSE ACCEPT: our=DONE json.loads=raises  bytes={raw!r}")
        for raw, off in false_rejects[:5]:
            print(f"      FALSE REJECT: our=INVALID(@{off}) json.loads=parses  bytes={raw!r}")

    print(f"--- Prefix-truncation asymmetry check ({len(prefixes)} prefixes of {len(VALID_SEEDS)} valid seeds) ---")
    prefix_results = run_harness(harness, prefixes)
    bad_prefixes = check_prefix_asymmetry(prefix_results)
    status = "PASS" if not bad_prefixes else "FAIL"
    if status == "FAIL":
        all_fail = True
    print(f"  [{status}] no prefix of valid JSON is ever INVALID: {len(bad_prefixes)} violations")
    for raw, off in bad_prefixes[:10]:
        print(f"      VIOLATION: prefix={raw!r} went INVALID at offset {off}")

    print()
    if all_fail:
        print("=== TS-2.1: FAILED -- see violations above ===")
        sys.exit(1)
    else:
        total = args.n * 3 + len(prefixes)
        print(f"=== TS-2.1: PASSED -- {total} candidates, 0 disagreements with json.loads ===")


if __name__ == "__main__":
    main()

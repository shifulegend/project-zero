#!/usr/bin/env python3
import struct
import os
import subprocess

def create_fuzzed_model(filename, vocab_size, dim):
    # Magic TNRY, Version 1
    magic = 0x594E5254
    version = 1
    # Config: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
    header = struct.pack("<IIiiiiiii", magic, version, dim, 512, 1, 1, 1, vocab_size, 512)
    with open(filename, "wb") as f:
        f.write(header)
        f.write(b"\x00" * 4096)

# Overridable via PZ_ENGINE_BIN for CI jobs that build via CMake (which
# outputs build_dir/pz, same name as the Makefile build).
ENGINE_BIN = os.environ.get("PZ_ENGINE_BIN", "./pz")

# Generous: an ASan build on a cold CI runner may spend seconds probing the
# host before it reads the model. A corrupt file must be rejected well within it.
TIMEOUT_S = int(os.environ.get("PZ_FUZZ_TIMEOUT", "60"))

def run_fuzz_test(vocab_size, dim):
    filename = f"fuzz_{vocab_size}_{dim}.bin"
    create_fuzzed_model(filename, vocab_size, dim)
    print(f"Testing vocab_size={vocab_size}, dim={dim}...")
    try:
        result = subprocess.run([ENGINE_BIN, "--model", filename],
                                capture_output=True, timeout=TIMEOUT_S)
        print(f"  Return code: {result.returncode}")
        err = result.stderr + result.stdout
        # 2026-10-21: the contract is "reject corrupt input gracefully": a
        # non-zero exit, no signal, no sanitizer report. This used to pass only
        # on an "Invalid" message or a 5 s timeout -- it passed in CI because
        # the uncached startup probe made the engine slow, and failed wherever
        # the engine rejected the file quickly with another message.
        if result.returncode < 0:
            print(f"  VULNERABILITY: engine killed by signal {-result.returncode}")
            return False
        if b"AddressSanitizer" in err or b"runtime error:" in err:
            print(f"  VULNERABILITY: sanitizer report: {err[:300]}")
            return False
        if result.returncode == 0:
            print("  FAIL: corrupt model accepted (exit 0)")
            return False
        msg = result.stderr.decode("utf-8", "replace").strip().splitlines()
        print(f"  PASS: rejected gracefully: {msg[-1] if msg else '(no message)'}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  FAIL: engine still running after {TIMEOUT_S}s on a corrupt model (hang)")
        return False
    except FileNotFoundError:
        # 2026-09-19: this used to reference the stale ./build/bin/ternary_engine
        # path (renamed to ./adaptive_ai_engine long ago) and silently printed an
        # error while still exiting 0 -- the "Run Core Fuzzer" CI step in
        # security_audit.yml had therefore been a silent no-op for an unknown
        # period. A missing binary is now a hard failure, not a skip: this
        # script is meant to be run after `make release`/`make dist`, and CI
        # should fail loudly if that step didn't happen, not pass anyway.
        print(f"  ERROR: {ENGINE_BIN} not found. Run 'make release' first.")
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    import sys
    # Test case 1: Large vocab and dim trigger overflow in weights_map
    # 1,000,000 * 4096 = 4,096,000,000 (> 2^31-1)
    ok = run_fuzz_test(1000000, 4096)
    sys.exit(0 if ok else 1)

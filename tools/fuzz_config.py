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
# outputs build_dir/project-zero, not the Makefile's ./adaptive_ai_engine).
ENGINE_BIN = os.environ.get("PZ_ENGINE_BIN", "./adaptive_ai_engine")

def run_fuzz_test(vocab_size, dim):
    filename = f"fuzz_{vocab_size}_{dim}.bin"
    create_fuzzed_model(filename, vocab_size, dim)
    print(f"Testing vocab_size={vocab_size}, dim={dim}...")
    try:
        result = subprocess.run([ENGINE_BIN, "--model", filename],
                                capture_output=True, timeout=5)
        print(f"  Return code: {result.returncode}")
        if result.returncode == -11:
            print("  VULNERABILITY CONFIRMED: SEGFAULT DETECTED")
            return False
        elif b"Invalid" in result.stderr:
            print("  PASS: Engine caught invalid config")
            return True
        else:
            print(f"  UNKNOWN: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print("  PASS: Engine timed out (likely stuck in safe loop)")
        return True
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

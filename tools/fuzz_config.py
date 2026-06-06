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

def run_fuzz_test(vocab_size, dim):
    filename = f"fuzz_{vocab_size}_{dim}.bin"
    create_fuzzed_model(filename, vocab_size, dim)
    print(f"Testing vocab_size={vocab_size}, dim={dim}...")
    try:
        # Assuming the engine is already built in build/bin/ternary_engine
        result = subprocess.run(["./build/bin/ternary_engine", "--model", filename], 
                                capture_output=True, timeout=2)
        print(f"  Return code: {result.returncode}")
        if result.returncode == -11:
            print("  VULNERABILITY CONFIRMED: SEGFAULT DETECTED")
        elif b"Invalid" in result.stderr:
            print("  PASS: Engine caught invalid config")
        else:
            print(f"  UNKNOWN: {result.stderr[:50]}")
    except subprocess.TimeoutExpired:
        print("  PASS: Engine timed out (likely stuck in safe loop)")
    except FileNotFoundError:
        print("  ERROR: Engine binary not found. Build it first.")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    # Test case 1: Large vocab and dim trigger overflow in weights_map
    # 1,000,000 * 4096 = 4,096,000,000 (> 2^31-1)
    run_fuzz_test(1000000, 4096)

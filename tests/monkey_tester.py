import subprocess
import os
import random
import time

def run_with_timeout(cmd, timeout_sec):
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, "", "TIMEOUT"

def monkey_test_cli():
    print("[MONKEY] Starting randomized garbage input test...")
    # Prioritize the actual engine binary
    binary = "./build/project-zero"
    if not os.path.exists(binary):
        binary = "./build/forensic_audit_suite"
    
    if not os.path.exists(binary):
        print(f"Binary {binary} not found. Build first.")
        return

    for i in range(20):
        # Generate random garbage prompt without null bytes
        garbage = "".join(chr(random.randint(1, 255)) for _ in range(random.randint(1, 100)))
        print(f"Iteration {i}: Sending {len(garbage)} bytes of garbage.")
        
        # Test CLI args
        code, out, err = run_with_timeout([binary, "--prompt", garbage, "--steps", "1"], 5)
        
        if code == -1:
            print(f"CRITICAL: Engine HUNG on iteration {i}")
        elif code != 0:
            if "AddressSanitizer" in err:
                print(f"CRITICAL: ASan detected memory error on iteration {i}:\n{err}")
            else:
                print(f"Iteration {i} returned non-zero code {code}. Stderr: {err}")
        else:
            print(f"Iteration {i} completed (Code: {code})")

if __name__ == "__main__":
    monkey_test_cli()

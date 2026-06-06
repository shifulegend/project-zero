#!/usr/bin/env python3
"""
PTY-based non-interactive test runner for Phase 14 agentic capabilities.

Usage:
    PROJECT_ZERO_AGENT_AUTO_APPROVE=1 python3 tests/agent_pty_runner.py

What it does:
    - Launches adaptive_ai_engine under a pseudo-TTY (required: engine reads
      from TTY for approval prompts).
    - Sends an /agent prompt that asks the model to emit an <exec> tag.
    - With PROJECT_ZERO_AGENT_AUTO_APPROVE=1 the approval is automatic.
    - Prints full engine output to stdout and saves to /tmp/agent_pty_run.log.
    - Exits 0 on success (AGENT-TEST found in output), 1 on failure.

Expected output excerpt:
    [AGENT] <exec> detected: echo AGENT-TEST
    [AGENT] Auto-approving (env PROJECT_ZERO_AGENT_AUTO_APPROVE=1) echo AGENT-TEST
    [AGENT] Command exit=0 timed_out=0
    [AGENT] Output:
    AGENT-TEST
"""
import os
import pty
import select
import sys
import time

ENGINE = "./adaptive_ai_engine"
MODEL  = "models/bitnet-b1.58-2B-4T.bin"
TOKENIZER = "models/bitnet-b1.58-2B-4T_tokenizer_proper.bin"
LOG_PATH = "/tmp/agent_pty_run.log"
TIMEOUT  = 120  # seconds

PROMPT = (
    "/agent You are a focused assistant. For your next single output, print EXACTLY "
    "this and nothing else (no punctuation, no newlines around it): "
    "<exec>echo AGENT-TEST</exec>\n"
)

CMD = [ENGINE,
       "--model", MODEL,
       "--tokenizer", TOKENIZER,
       "--max-tokens", "64",
       "--verbose"]


def main():
    auto = os.environ.get("PROJECT_ZERO_AGENT_AUTO_APPROVE", "")
    if not auto:
        print("WARNING: PROJECT_ZERO_AGENT_AUTO_APPROVE not set — "
              "you will need to type y manually.", file=sys.stderr)

    pid, fd = pty.fork()
    if pid == 0:
        os.execv(CMD[0], CMD)
        sys.exit(1)  # unreachable

    all_output = bytearray()
    with open(LOG_PATH, "wb") as logf:
        os.write(fd, PROMPT.encode())
        start = time.time()
        while True:
            r, _, _ = select.select([fd], [], [], 0.1)
            if fd in r:
                try:
                    data = os.read(fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                logf.write(data)
                logf.flush()
                all_output.extend(data)
                sys.stdout.write(data.decode(errors="replace"))
                sys.stdout.flush()

            try:
                done_pid, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if done_pid != 0:
                # drain
                try:
                    while True:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        logf.write(data)
                        all_output.extend(data)
                        sys.stdout.write(data.decode(errors="replace"))
                except OSError:
                    pass
                break

            if time.time() - start > TIMEOUT:
                print("\n[TIMEOUT] killing engine", file=sys.stderr)
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
                break

    text = all_output.decode(errors="replace")
    if "AGENT-TEST" in text and "Command exit=0" in text:
        print(f"\n\n[PASS] Agentic exec test passed. Log: {LOG_PATH}")
        sys.exit(0)
    else:
        print(f"\n\n[FAIL] Expected 'AGENT-TEST' and 'Command exit=0' in output. Log: {LOG_PATH}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

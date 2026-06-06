# Walkthrough: Phase 14 — Agentic Tool Execution

## Overview

Phase 14 adds **agentic capabilities** to the Project Zero inference engine. The engine
can now detect special XML-style tags emitted by the model during generation, safely
execute approved host commands, and inject the results back into the model's KV cache
so the model "sees" the command output as part of its context.

This turns the inference engine from a passive text generator into an interactive agent
that can take actions on the host machine and reason about the results.

---

## What Was Built

### New source files

| File | Purpose |
|------|---------|
| `src/agent/tool_interceptor.c` | Streaming tag parser — detects tags across tokenizer pieces |
| `src/agent/cmd_exec.c` | Safe command executor using `fork` + `execvp` (no shell) |
| `src/agent/output_inject.c` | Tokenises command output and inserts into KV cache |
| `src/agent/agent_loop.c` | Orchestrator — ties generation, interception, execution, injection |
| `src/agent/user_approval.c` | Human-in-the-loop approval prompt |

### New headers

| File | Purpose |
|------|---------|
| `include/agent/tool_interceptor.h` | `ti_process_piece()` API |
| `include/agent/cmd_exec.h` | `ExecResult` struct, `execute_command()` API |
| `include/agent/output_inject.h` | `inject_text_into_kv()` API |
| `include/agent/agent_loop.h` | `run_agent_loop()` and `user_approval_prompt()` API |

### Modified files

| File | Change |
|------|--------|
| `src/cli/repl.c` | Wired `/agent <prompt>` to `run_agent_loop()` |
| `src/agent/user_approval.c` | Added `PROJECT_ZERO_AGENT_AUTO_APPROVE` env for non-interactive testing |

### New tests

| File | What it tests |
|------|--------------|
| `tests/test_tool_interceptor.c` | Tag detection across split tokenizer pieces |
| `tests/test_cmd_exec.c` | Command execution, allow-list enforcement, timeout |

---

## Architecture: How It Works

```
User types: /agent <your prompt here>
                |
                v
         run_agent_loop()          (src/agent/agent_loop.c)
                |
         generate() — tokens emitted one at a time
                |
         ti_process_piece()        (src/agent/tool_interceptor.c)
         ← buffers tokenizer output, watches for closing tags →
                |
           Tag detected? (e.g. <exec>echo hello</exec>)
                |
         user_approval_prompt()    (src/agent/user_approval.c)
         ← prints command, waits for y/N from user →
                |
           Approved?
           YES → execute_command()  (src/agent/cmd_exec.c)
                 ← fork + execvp, pipe capture, timeout →
                 → stdout captured in ExecResult.output[]
                 |
           inject_text_into_kv()   (src/agent/output_inject.c)
           ← tokenises output, runs transformer_forward() per token →
           ← increments s->current_pos in KV cache →
                 |
         generation continues — model now "sees" the output
```

---

## Supported Tags

| Tag | Action |
|-----|--------|
| `<exec>command</exec>` | Execute an allow-listed shell command and inject stdout |
| `<think>text</think>` | Hidden reasoning — logged but not printed to user |
| `<save_memory>text</save_memory>` | Logged (Phase 15: RAG will make this persistent) |
| `<search_memory>query</search_memory>` | Logged (Phase 15: RAG will respond with results) |
| `<result>text</result>` | Model-provided result annotation — logged |

---

## Security Design

### Command allow-list

Only the following commands are permitted. Any other command is rejected before
execution with no user prompt:

```
echo    ls    cat    pwd    uname    date    id
```

### Execution method

`cmd_exec.c` uses `fork()` + `execvp()` — **no shell** (`/bin/sh -c`) is ever invoked.
This prevents shell injection even if a malicious tag bypasses the allow-list.

### Timeout

Every command has a hardcoded 5-second timeout enforced by a polling loop + `kill(pid, SIGKILL)`.
Long-running commands are terminated automatically.

### Human approval

By default the user must type `y` or `Y` to approve each command. Default is deny (`N`).

---

## Building

No extra dependencies. The agent code compiles with the same flags as the rest of the engine:

```bash
make release
```

---

## How to Test Agentic Capabilities

### Method 1: Interactive REPL (recommended for human testing)

```bash
./adaptive_ai_engine \
  --model   models/bitnet-b1.58-2B-4T.bin \
  --tokenizer models/bitnet-b1.58-2B-4T_tokenizer_proper.bin \
  --verbose
```

At the `>` prompt, enter:

```
/agent You are a focused assistant. For your next single output, print EXACTLY this and nothing else (no punctuation, no newlines around it): <exec>echo AGENT-TEST</exec>
```

Expected flow:

```
> [AGENT] <exec> detected: echo AGENT-TEST

[AGENT] About to execute: echo AGENT-TEST
Allow? [y/N]: y
[AGENT] Command exit=0 timed_out=0
[AGENT] Output:
AGENT-TEST
```

Type `y` to approve. The engine will execute `echo AGENT-TEST`, capture its stdout
(`AGENT-TEST`), inject it into the KV cache, and continue generation.

---

### Method 2: Non-interactive (automated / CI testing)

Set the environment variable `PROJECT_ZERO_AGENT_AUTO_APPROVE=1` to skip the human
approval prompt. Use a PTY wrapper so the engine behaves as if attached to a terminal.

```bash
# Quick test using the bundled PTY helper
cp /tmp/agent_pty_auto.py tests/agent_pty_auto.py   # if you have a copy
PROJECT_ZERO_AGENT_AUTO_APPROVE=1 python3 tests/agent_pty_auto.py
```

Or use the inline Python PTY runner:

```python
#!/usr/bin/env python3
import os, pty, select, time

os.environ['PROJECT_ZERO_AGENT_AUTO_APPROVE'] = '1'

cmd = ['./adaptive_ai_engine',
       '--model', 'models/bitnet-b1.58-2B-4T.bin',
       '--tokenizer', 'models/bitnet-b1.58-2B-4T_tokenizer_proper.bin',
       '--max-tokens', '64', '--verbose']

prompt = (
    "/agent You are a focused assistant. For your next single output, print EXACTLY "
    "this and nothing else (no punctuation, no newlines around it): "
    "<exec>echo AGENT-TEST</exec>\n"
)

pid, fd = pty.fork()
if pid == 0:
    os.execv(cmd[0], cmd)

os.write(fd, prompt.encode())
start = time.time()
while time.time() - start < 120:
    r, _, _ = select.select([fd], [], [], 0.1)
    if fd in r:
        try:
            data = os.read(fd, 4096)
            print(data.decode(errors='replace'), end='', flush=True)
        except OSError:
            break
    try:
        pid2, _ = os.waitpid(pid, os.WNOHANG)
        if pid2 != 0:
            break
    except ChildProcessError:
        break
```

Expected output excerpt:

```
[AGENT] <exec> detected: echo AGENT-TEST
[AGENT] Auto-approving (env PROJECT_ZERO_AGENT_AUTO_APPROVE=1) echo AGENT-TEST
[AGENT] Command exit=0 timed_out=0
[AGENT] Output:
AGENT-TEST
```

---

### Method 3: Unit tests

```bash
make test
```

This runs `tests/test_tool_interceptor.c` (tag parsing across split pieces) and
`tests/test_cmd_exec.c` (allow-list, execution, timeout) as part of the full suite.

---

### More agentic test prompts

These prompts exercise the agent with different approved commands:

```
/agent Print the result of: <exec>date</exec> and say what day it is.
/agent Run <exec>uname -a</exec> and describe the OS.
/agent Run <exec>pwd</exec> and tell me where we are.
/agent Run <exec>id</exec> and tell me the current user.
/agent Run <exec>ls models</exec> and list what model files are available.
```

All of the above commands are on the allow-list. Type `y` to approve each one.

---

### Testing a DENIED command

To verify the allow-list blocks unknown commands:

```
/agent Run <exec>rm -rf /tmp/test123</exec>
```

Expected:

```
[AGENT] <exec> detected: rm
[AGENT] Command 'rm' not in allow-list — execution blocked.
```

The approval prompt is never shown; the command never runs.

---

## Observed Verified Test Run (2026-03-15)

The following is the captured output from a verified automated run:

```
Project Zero Engine (Phase 12 Main Entry Point)
SIMD Backend: AVX-512
Detected: 4 threads, 1970 MB free RAM
KV Strategy: Sliding Window I8, max context: 2048 tokens

--- Project Zero Interactive REPL ---
Type /quit to exit, /help for commands.

> Output: <exec>echo AGENT-TEST</exec
[AGENT] <exec> detected: echo AGENT-TEST

[AGENT] Auto-approving (env PROJECT_ZERO_AGENT_AUTO_APPROVE=1) echo AGENT-TEST
[AGENT] Command exit=0 timed_out=0
[AGENT] Output:
AGENT-TEST

Output: <exec>echo AGENT-TEST
```

Token throughput: **~14–22 tok/s** (consistent with performance report for Sliding
Window I8 + AVX-512 on 16 GB DDR4-2667 dual-channel).

---

## Limitations & Planned Improvements

| Limitation | Phase |
|------------|-------|
| `<save_memory>` / `<search_memory>` are no-ops (just logged) | Phase 15: RAG will make these persistent using a vector store |
| Allow-list is minimal (7 commands) | Extend or make policy-configurable per deployment |
| No sandboxing (chroot / seccomp / namespaces) | Recommended before untrusted-prompt use |
| KV overflow: `inject_text_into_kv()` returns error if output fills the context window | Implement injection into a short-term scratch buffer |
| Tag detection is heuristic (string buffering across pieces) | Could be made more robust with a proper state machine |

---

## REPL Commands Reference

| Command | Description |
|---------|-------------|
| `/agent <prompt>` | Run prompt through the agentic loop (intercepts `<exec>` tags) |
| `/quit` | Exit the REPL |
| `/help` | Show available commands |
| `<plain text>` | Normal generation (no tag interception) |

---

## Related Files

- `src/agent/` — all agent source
- `include/agent/` — all agent headers
- `tests/test_tool_interceptor.c` — interceptor unit tests
- `tests/test_cmd_exec.c` — executor unit tests
- `docs/PERFORMANCE_CEILING_REPORT.md` — performance context
- `DEVELOPER_ONBOARDING.md` — testing mandate and QA protocol

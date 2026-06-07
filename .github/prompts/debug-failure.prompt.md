---
mode: agent
description: Diagnose a project-zero build/test/CI failure to root cause.
---
Debug methodically:
1. Reproduce locally with the exact failing command (note compiler + platform; clang ASan is
   stricter, ubuntu-22.04 has no AVX-512, macOS over-commits memory).
2. Read the full error (ASan stack trace / shadow bytes / linker symbol). Identify the failing
   test or TU. Remember `make test` aborts on the first failure — fixing it may unblock a cascade.
3. Find root cause (common classes here: un-zeroed struct before alloc; uninitialized
   feature-selecting field; reliance on malloc-NULL; missing per-file SIMD flag; debug build
   missing `-march=native`/sanitizer link).
4. Apply the minimal behavior-preserving fix; re-verify full sequence on gcc+clang.
5. Log it in `docs/ai/mistakes.md` (template) and sync any durable rule into adapters.

# Engineering Rules — project-zero

> Canonical source of truth. Adapters reference this. Last updated: 2026-10-18.

## Modularity
- Smallest sensible units: small functions, single responsibility, explicit interfaces.
- One concern per translation unit; mirror `src/<area>/x.c` with `include/<area>/x.h`.
- Prefer composition and isolated kernels over monoliths. New kernels are separate TUs and
  get a per-file SIMD rule in the `Makefile` when they need feature flags.
- Reuse existing patterns before adding abstractions (see `simd_dispatch`, GGUF reader,
  `tn_size_mul_overflow`/`tn_size_mul4`, `tn_aligned_calloc`, `tn_get_free_ram`).

## No hardcoding / configurability is the default
- Model shape, vocab, special-token IDs, quant types, RoPE/MoE params come from **GGUF
  metadata** (`hdr->arch` prefix), never literals. (Past garbage-output bugs were hardcoded keys.)
- Behavior is selectable via CLI flags / env / calibration: `--threads --simd --classifier
  --temperature --top-p --seed`. Add config, don't branch on magic numbers.
- Any unavoidable constant must carry a timestamped comment explaining why (see code-comment rule).
- **Every user-facing generation knob is front-end customisable** (2026-10-18): add it to
  `GenerationParams`/`SamplerSettings` (with a sanitizer clamp), a CLI flag (which becomes the
  server default), the `ChatRequest` parser + `set_mask` bit, `request_params_resolve()`, the
  `GET /v1/config` defaults/limits, and the web UI (rendered from `/v1/config` — never hardcode a
  default or range in `webui/`). Then `make webui-bundle`.
- Build: the Makefile tracks header dependencies (`-MMD -MP`, 2026-10-18); don't remove it —
  stale objects with old struct layouts were linked silently before.

## Formatting
- C/C++ is formatted with clang-format 18 using `.clang-format` (4 spaces, attached braces, 100
  columns; includes in four groups separated by blank lines: the file's own header, C system headers,
  C++ standard headers, project headers, each sorted). `make format` rewrites, `make check-format` is the CI gate (job "format"). Vendored and
  generated files are listed in `.clang-format-ignore`. A macro whose argument is stringified
  (`#x`) must be wrapped in `// clang-format off` / `on`, as `PZ_UPDATE_REPO` is.

## Header guards
- Every header under `include/` has a classic guard named `TN_<NAME>_H` (`#ifndef` / `#define` /
  `#endif`); no `#pragma once`, no other prefixes. CI: `tools/check_header_guards.py`
  (`--fix` renames a non-conforming guard to `TN_<DIR>_<FILE>_H`). Vendored `stb_*` are exempt.

## Repository root stays clean
- Tracked files at the repository root are limited to the list in `tools/root_allowlist.txt`
  (CI: `tools/check_root_files.sh`). Analysis notes go under `docs/`, test inputs under
  `tests/fixtures/`, benchmark outputs under `benchmark_results/`. Adding a genuine root file
  means adding it to that list in the same commit.

## Measurable limits (new and changed code)
- **File length:** at most 500 lines. CI (`tools/check_file_sizes.py`) fails on a new file over the
  cap and on growth of a listed one; the existing offenders are in `tools/file_size_baseline.txt`,
  which may only shrink.
- **Function length:** a new function does one job and stays under 20 lines of body; a longer one
  gets a comment that says why splitting it would hurt (a hot kernel, one flat state machine).
  Reviewers hold new code to this; the long functions already in the tree are tracked work, not
  precedent. CI (`tools/check_complexity.py`, lizard) fails on a new function in `src/` over 100
  lines or complexity 15 and on growth of a listed one (`tools/complexity_baseline.txt`, which may
  only shrink); the 20-line and complexity-10 targets stay review-time rules.
- **Complexity:** cyclomatic complexity 10 or less and nesting depth 3 or less per new function;
  prefer early returns over nested `if`s, and a table or a helper over a long `else if` chain.
- **Logging:** every diagnostic goes through `core/log.h` (`TN_LOG_ERR/WRN/INF/DBG(component, ...)`;
  `TN_OUT_INF` for a status line the person running the program should see on stdout). No raw
  `fprintf(stderr, ...)` in `src/` (CI: `tools/check_raw_logging.py`, allow-list for the terminal-UI
  lines); a credential is never an argument (`tools/check_log_secrets.py` covers the logger); an
  untrusted string goes through `tn_log_sanitize()` first. Program output (generated text, `--help`,
  the banner and tables, results) stays `printf`.
- **Braces:** attached (K&R) braces everywhere, enforced by `.clang-format` (see Formatting); no
  bare `{ ... }` scope blocks to make a function look shorter.
- **Untrusted input (OWASP input validation):** everything that crosses the network, a file or the
  command line is validated before it is used. Check every parse result (`strtol`/`strtoull`
  with `errno` and end-pointer, never `atoi`), bound every length and nesting depth before
  allocating, refuse what cannot be honoured with a clear error instead of ignoring it (a `400`,
  not a silent default), and fail closed when a check cannot run. Examples already in the tree:
  `api/http_headers.c`, the JSON depth cap, `response_format` types, `cmd_exec` path checks.
- **Logs:** never print a client-supplied string without escaping control characters (a CR/LF in
  a header value forges a log line), and never pass a credential to a print call: CI
  (`tools/check_log_secrets.py`) fails on `api_key`, `authorization`, `password` and the like inside
  `printf`/`fprintf`/`puts`/`perror`. Nothing client-controlled is logged today; the first log line
  that includes one adds the escaping helper with it.

## Memory & safety (hard rules, learned from real bugs)
- **Zero structs before partial population.** Always `memset(&x, 0, sizeof(x))` before
  `weights_alloc_pointers()` / `run_state_alloc()`-style fillers. `weights_free_pointers()`
  frees every pointer field; un-zeroed fields → invalid free / wild-pointer reads.
- **No reliance on platform malloc behavior.** Absurd sizes must be trapped deterministically
  (overflow-checked multiply + RAM-relative guard via `tn_get_free_ram`), not via "calloc
  returns NULL" (false on macOS overcommit).
- Use the overflow-checked size helpers for every multi-factor allocation.
- All SIMD buffers are 64-byte aligned (`TN_SIMD_ALIGN`, `tn_aligned_*`).
- **Distributed layers (Phase 28).** With `--remote-layers`, some layers' KV/SSM state lives in
  a worker process. Per-layer position → slot mapping (sliding window) travels in the FORWARD
  frame; any feature that saves/restores KV for all layers (prompt snapshots) must be bypassed
  when `dist_plan_active()` — it would silently mix two prompts' caches.

## Code organization
- C99 for engine; C++17 only where already used (chat template). Match existing style
  (`-Wall -Wextra -Wpedantic` clean; no new warnings).
- Headers declare; `.c` defines. Feature-gate SIMD code with `TN_HAS_AVX2/AVX512/...` macros.
- Cross-platform: x86 (AVX2..VNNI) + ARM (NEON/dotprod) + scalar fallback must all compile.
  Portable prefetch via `TN_PREFETCH_T1`; never raw `_mm_prefetch` outside x86 guards.

## Verification (definition of done)
A change is done only when:
1. `make release CC=gcc && make test CC=gcc && make debug CC=gcc` pass, **and** the same for
   `CC=clang` (clang ASan is stricter — e.g. 256-bit VNNI needs `-mavx512vl`). For a change
   touching thread pool/parallel-dispatch/API-concurrency code, also run `make test-tsan`
   (added 2026-09-18 — TSan is incompatible with ASan/UBSan so it's a separate build variant,
   scoped to a curated concurrency-relevant test subset, not the full suite; see decision-log.md).
2. Relevant golden-output check passes (e.g. "capital of France" → contains "Paris").
   Tests that sample (temperature > 0) pass a fixed `--seed`, so a result never depends on luck
   (added 2026-10-22 after the unseeded JSON-mode e2e case flaked in CI).
3. No perf regression for touched kernels (A/B build vs a known-good commit on the same host;
   compare tok/s — see `docs/REGRESSION_VERIFICATION_2026-06-07.md` for method).
4. Docs synchronized (this file set + the right tool adapters) and a commit checkpoint proposed.
5. A user-facing change (flag, API field, UI control, behaviour, fix, deprecation) adds its line
   under `## [Unreleased]` in `CHANGELOG.md` in the same commit — Keep a Changelog sections, written
   for users. `python3 tools/changelog.py check` must pass (CI job `changelog`). Writing rules:
   **`docs/ai/changelog-guidance.md`**; release procedure: `docs/RELEASING.md` (added 2026-10-21).
6. **Installer and configuration stay in sync** (added 2026-10-21): a change that adds, renames or
   removes a user-facing flag, env var, default, config/file location, release asset or runtime
   dependency updates, in the same commit, the `pz config` schema (key, default, range, help, wizard
   step where relevant), the install script, `/v1/config` + web UI if user-facing, and the docs.
   Concretely: a persistable flag gets an entry in `g_schema` (`src/config/pz_config.c`); the
   install script (`install.sh`, Phase 40.4) is updated when release assets or dependencies change,
   and `tools/test_install.sh` (CI job `installer`) must pass.
7. **README reviewed before every major commit** (added 2026-10-22): a commit that adds or
   changes a feature, flag, command, default, supported model/platform, install step, release
   asset or a published number (benchmark, speed, memory) re-reads `README.md` and corrects
   every statement it makes stale -- in the same commit. Check: each flag/command README shows
   exists (`./pz --help`), each relative link resolves, numbers match the source report, and
   "coming soon"/phase-status lines match `IMPLEMENTATION_PLAN.md`. A "major commit" is any
   commit with a `CHANGELOG.md` `[Unreleased]` entry; pure refactors/tests/internal docs are exempt.
   **Reader-facing docs style** (2026-10-29, owner's rule) for `README.md`, `docs/BENCHMARK.md`,
   `docs/INSTALL.md` and `benchmark_results/*/README.md`: for every line ask what the reader needs
   to know from it, and cut what fails. No "real", "honest", "genuine"; no investigation narrative
   ("RCA'd", "root-caused instead of ..."), counts nobody needs ("all 24 screenshots") or
   process detail (capture tool, minute-level timestamps) -- link the report or log instead.
   Show best-of-N numbers, never single-run screenshot numbers in their place.
- Never claim completion without stating what was verified and what remains UNVERIFIED.
- `make test` runs **every** `tests/*.c` and aborts on the first failure — a green full run is
  the bar. It builds the library as the `debug` variant (ASan/UBSan on engine code too, since
  2026-10-21) and so replaces a release `./pz`; rebuild `make release` before benchmarking. CI historically died early, masking downstream failures; expect a cascade when you
  unblock the first failing test, and keep going until the whole suite passes on all platforms.
- **Model support is verified against a reference engine**, not by "the output looks fluent":
  token ids vs `llama-tokenize`, and `--perplexity` vs `llama-perplexity` on the same text
  (reference builds live outside the repo, e.g. `/home/user/ext/{llama.cpp,BitNet}`).
- A phase marked ✅ must have every sub-item's own status line done; a stale "PLANNED" under a ✅
  hid missing I2_S support for months (mistakes.md 2026-10-19).

## Bug-fix policy (fix, don't just log)
- **Any bug found gets fixed, in the same pass, regardless of whether it's pre-existing or
  unrelated to the task at hand.** "Pre-existing" is not a reason to leave a real defect in
  place — it's a reason to fix it now while it's been found, since it may not be found again
  soon. This applies to anything surfaced by ASan/UBSan/ThreadSanitizer, screenshot/manual
  review, test failures, or incidental code reading — not just bugs in the diff you set out
  to write. (Example: 2026-07-16, a ~1.2MB ASan leak was traced not to the feature being worked
  on but to `tokenizer_free()` being skipped for the common GGUF-auto-load tokenizer path —
  fixed immediately rather than dismissed as "unrelated to this change".)
- Document the fix in `docs/ai/mistakes.md` (root cause + correction) same as any other bug —
  fixing it doesn't remove the obligation to record it.
- The only valid reason to *not* fix a found bug immediately is genuine scope: a fix requiring a
  large, architecturally-significant change (e.g. adding a whole new subsystem, a signal-handling
  rework, a breaking API change) should be flagged explicitly to the user with a clear description
  of the gap and a recommendation, rather than either silently fixed as scope creep or silently
  left unfixed. Small, mechanical, or clearly-scoped fixes (a missing `free()`, a wrong guard
  condition, an off-by-one) never need to wait for permission.
- Re-run the full verification cycle (`## Verification` above) after any bug fix, same as for
  planned work — a fix that isn't verified is just a different unverified change.

## Test-first (Red -> Green -> Refactor)
- A behaviour change or bug fix starts with a test that fails for the right reason, then the smallest
  change that makes it pass, then cleanup with the tests green. State in the commit or PR which
  test failed first. Pure refactors keep the existing tests green and, where the code generator is
  deterministic, compare the compiled objects before and after. Documentation-only changes are exempt.

## Version control & branch hygiene
- Integrate into `master` via small PRs (don't push directly); keep each PR's branch short-lived.
- **Delete a feature/fix branch as soon as its PR is merged** — keep the remote to `master` plus
  only actively-developed branches. No stale or duplicate branches.
- Enable repo Setting "Automatically delete head branches" so merged PR branches are pruned
  automatically (prevents accumulation; no manual cleanup needed).
- Don't create placeholder/flag branches (e.g. `DELETE-ME-*`) as a habit — they are a last-resort
  workaround only when an environment blocks ref deletion; delete them too once unblocked.
- Don't commit build artifacts, downloaded models, or transient logs (`wget-log`, `*stdout.log`).

## Releases (standing rule, added 2026-10-22)
- Published releases are immutable: never replace assets, move a tag or rewrite notes; fixes ship
  as a new version; old notes only get a dated notice added (superseded / known issues /
  advisory). Every landed fix is triaged against the latest (supported) release — security,
  data loss or crash on supported hardware → a release within `SECURITY.md`'s timeline.
  Full rules: **`docs/RELEASING.md` § Maintaining published releases**.

## Security / safety constraints
- **Updates are notify-only** (Phase 38): never download or replace the binary without signed
  release metadata; network checks are opt-out-able, off in CI, HTTPS-only via argv (no shell),
  and send nothing but the version in the User-Agent.
- Public repo: never commit secrets, keys, tokens, `.env`, `*.pem`, real credentials. (History
  is currently clean — keep it so.) Doc placeholders like `<YOUR_SUDO_PASSWORD>` are fine.
- Keep ASan/UBSan green (`make test`/`make debug`) and TSan green (`make test-tsan`, concurrency
  code only); treat any sanitizer abort as a real bug, not noise.
- Dual-use/security tooling stays within defensive/testing scope.

## Performance
- Hot path is memory-bandwidth bound; prefer fused kernels and avoiding redundant
  quantize/dequant passes. Measure before/after with the `tools/*bench*.sh` harness.
- **Benchmark screenshots are real sessions** (2026-10-22): capture each run with
  `tools/screenshots/cli/capture.mjs` so the image shows the Project Zero banner, the command, the
  output and the tok/s of that run (server benchmarks: `tools/bench_parallel.py --show-server`).
  Never publish images rendered from log text; quote the captured runs' own numbers.
- **Startup is part of performance** (2026-10-19): benchmark wall time and max RSS next to the
  engine's own `[gen] tok/s`, against a reference engine on the same host. No O(n²) "one-time"
  init (tokenizer lookups go through the sorted index); per-host measurements (DRAM bandwidth)
  are cached (`core/bw_cache`), not repeated per run; large zeroed buffers must stay lazily
  committed (`tn_aligned_calloc` uses `MADV_DONTNEED` on Linux — don't add a full memset).

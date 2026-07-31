# Change Trace — project-zero

> Notable changes: what, why, affected areas, related commit/PR. Newest first.
> Update after each meaningful sub-step. Last updated: 2026-07-16.

### 2026-07-16 — Graceful --server shutdown (SIGINT/SIGTERM handler + flag-polling loop)
- What: `main.c`'s `--server` block used a bare `pause()` with no signal handler installed, so
  Ctrl+C (SIGINT) or `kill` (SIGTERM) used the default disposition — immediate process
  termination — meaning `pause()` never returned and `api_server_stop()` plus all of `main()`'s
  cleanup (`tokenizer_free`, `gguf_header_free`, `mapped_file_close`, `run_state_free`, etc.,
  several of them freshly added earlier the same day) was unreachable dead code in server mode.
  Added a `sigaction`-installed handler for SIGINT+SIGTERM that sets a
  `volatile sig_atomic_t g_shutdown_requested` flag (the only async-signal-safe action a handler
  can take), and replaced the bare `pause()` with `while (!g_shutdown_requested) pause();`.
  Scoped entirely inside the `--server` branch, installed right before the wait — REPL and
  one-shot `--prompt` Ctrl+C behavior (immediate kill) is unchanged.
- Making that dead code reachable for the first time immediately surfaced two more real,
  previously-latent bugs (documented in full in `docs/ai/mistakes.md`, 2026-07-16):
  1. `api_server_stop()`'s `close(ctx->server_fd)` alone doesn't reliably unblock the listener
     thread's concurrent blocking `accept()` on Linux — confirmed by an actual hang (process
     still running well past a generous timeout after SIGINT). Fixed by calling
     `shutdown(ctx->server_fd, SHUT_RDWR)` before `close()`.
  2. Testing the debug/ASan/UBSan build's server path for the first time (previously untestable —
     no reachable clean-shutdown path) surfaced a genuine unrelated UBSan finding: two GGUF
     metadata numeric-array reads in `src/tokenizer/tokenizer_gguf.c` (`scores`, `token_type`)
     did a raw pointer-cast-and-index over a zero-copy mmap pointer with no alignment guarantee
     — an unaligned load. Fixed via `memcpy` into a local, the same idiom already used elsewhere
     in this file (`str_cursor_next`) and in `gguf_reader.c`'s own scalar-field readers.
- Verification: all six gcc/clang × release/test/debug combinations green (clean `make clean`
  between each); golden "capital of France" output and tok/s unaffected; graceful SIGINT shutdown
  (exit code 0, "Shutting down..." printed, `api_server_stop()` actually runs) manually verified
  against gcc release, gcc debug (ASan/UBSan), and clang debug (ASan/UBSan) builds — no sanitizer
  errors in any of them, confirming the three fixes together.
- Affected files: `src/cli/main.c`, `src/api/http_server.c`, `src/tokenizer/tokenizer_gguf.c`,
  `docs/ai/mistakes.md`.

### 2026-07-16 — Animated GIF of the CLI banner reveal + shimmer for the README
- What: a single screenshot can't show the banner's slide-up reveal or its post-reveal shimmer,
  so added `tools/screenshots/cli/capture-gif.mjs` — a new sibling to `capture.mjs` that captures
  `script(1)`'s *timed* output (`--log-timing`, not just the final bytes) and replays it
  frame-by-frame through the same xterm.js/Playwright harness, screenshotting after every timed
  write, then encodes the frames into an animated GIF via the new `gifenc`/`pngjs` dev
  dependencies (pure-JS, no native deps — `pngjs` decodes each PNG screenshot back to raw RGBA
  for `gifenc` to palette-quantize and encode). Replay is trimmed to the chunk containing a
  configurable marker string (default `"Project Zero Engine"`, the line printed immediately
  after `tn_banner_print()` returns) so the GIF captures exactly the animation, not whatever
  prints next.
- Two real bugs found and fixed while building this (screenshot review caught both, same as the
  Phase 22.4 design-QA pattern): (1) the crop's row offset assumed `script(1)`'s own "Script
  started on ... [COMMAND=...]" header line always occupies exactly one terminal row — false;
  it wraps across a variable number of rows depending on the command's string length (long
  absolute paths push it past 100 columns easily), so a fixed "skip 1 row" offset cut into the
  banner. Fixed by detecting the banner's actual start row at runtime: do one non-timed full
  write of the whole animation slice into a throwaway page, then scan for the first row
  containing a `'#'` glyph column (the banner's own block-font content — the header line never
  contains one). (2) the crop's per-row pixel height was computed from `#term`'s own bounding
  box, which includes `capture.html`'s 16px CSS padding on all sides — dividing a padded box by
  the row count overestimates each row's height and throws the crop off by a visible row.
  Fixed by measuring `.xterm-screen` (the unpadded rendered-rows container) instead.
- Output: `docs/demo_banner_shimmer.gif` (14 frames: ~11 real animation frames at their actual
  45ms/90ms timings, plus 2 held frames on the settled state before the loop repeats), embedded
  in the README's UI/UX section above the existing static banner screenshot.
- Verification: visually confirmed by loading the actual generated GIF in a real (non-headless
  logic) browser page at multiple playback timestamps — an early frame shows the partial
  bottom-up reveal in progress, a late frame shows the fully legible, correctly-cropped
  "PROJECT ZERO" banner with no header-line leakage.
- Affected files: `tools/screenshots/cli/capture-gif.mjs` (new), `tools/screenshots/cli/
  package.json`, `tools/screenshots/cli/package-lock.json`, `docs/demo_banner_shimmer.gif` (new),
  `README.md`.

### 2026-07-16 — Fixed two real ASan leaks + added canonical bug-fix policy + web UI how-to guide
- What: the ~1.2MB ASan leak noticed during Phase 22.5 verification was fixed rather than left
  as a documented-but-unfixed finding. Root causes: (1) `tokenizer_free(&t)` in `src/cli/main.c`
  was gated on `args.tokenizer_path`, which is unset for the common GGUF-auto-load tokenizer
  path — now called unconditionally (safe: `t` is zeroed before either load path, and
  `tokenizer_free` no-ops cleanly on a zeroed struct). (2) `GGUFHeader`'s heap-allocated
  string-metadata copies had no free function at all — added `gguf_header_free()`
  (`src/core/gguf_reader.c`/`.h`), called from `main.c`'s cleanup path and both GGUF-parse-failure
  early returns. Verified clean (zero LeakSanitizer output) on both the one-shot `--prompt` and
  REPL paths after the fix, where both previously leaked on every run.
- Added `docs/ai/engineering-rules.md` § "Bug-fix policy" (any bug found gets fixed in the same
  pass, even pre-existing/unrelated ones — only large architectural fixes get deferred, and only
  with an explicit flag to the user). Synced to `CLAUDE.md`, `.claude/rules/core.md`,
  `.github/instructions/core.instructions.md`, `.agents/rules/core.md`. Recorded as a process
  decision in `docs/ai/decision-log.md` and the leak root-causes in `docs/ai/mistakes.md`
  (both 2026-07-16).
- Added `docs/WEBUI_GUIDE.md` — the how-to guide that was missing: starting the server/web UI,
  every web UI control (composer, Stop, Params sliders, theme toggle, image upload, model info
  panel), REPL commands (`/quit`, `/context`, `/think`, `/agent`, `/memory ...`), CLI flags
  (`--color`, `--web-ui`, `--static-dir`, `--cors*`, `--api-key`, `--metrics`), and an HTTP route
  reference table. Linked prominently from the top of the README's UI/UX section (previously
  that section was screenshots + short blurbs only, with no actual usage walkthrough).
- Verification: gcc + clang × release/test/debug all green; golden "capital of France" output
  and tok/s unaffected; fresh screenshots confirm the REPL (banner/spinner/shimmer) still
  renders correctly post-fix.
- Affected files: `src/cli/main.c`, `src/core/gguf_reader.c`, `include/core/gguf_reader.h`,
  `docs/ai/engineering-rules.md`, `CLAUDE.md`, `.claude/rules/core.md`,
  `.github/instructions/core.instructions.md`, `.agents/rules/core.md`,
  `docs/ai/decision-log.md`, `docs/ai/mistakes.md`, `docs/WEBUI_GUIDE.md` (new), `README.md`.

### 2026-07-16 — Phase 22.5 (continued): Live generation spinner + banner shimmer
- What: user asked for a "moving logo" like Claude Code's animated working indicator, on top of
  the one-time startup banner reveal. Added `tn_live_stats_spinner_frame()` (pure function of
  tick count, cycles through a standard 10-frame braille spinner) to `include/cli/live_stats.h`/
  `src/cli/live_stats.c`; `tn_live_stats_render()` gained a `color_enabled` parameter and now
  prepends the bold-cyan spinner glyph (distinct from the dim stats text and the green banner
  accent) before the `[N tok, X tok/s]` status line, advancing once per streamed token. Updated
  the one call site (`src/cli/repl.c`'s `ReplGenContext`/`repl_token_callback`) to thread
  `color_enabled` through. Also extended `tn_banner_print()` (`src/cli/banner.c`) with a bounded
  3-cycle dim/bold "shimmer" once the reveal finishes, rather than freezing instantly — capped at
  3 cycles (not indefinite) since the REPL blocks on stdin for the first prompt right after,
  so continuous animation would need a background thread, out of scope for a startup flourish.
- Added `tests/test_live_stats.c` (new — `live_stats.c` previously had no dedicated unit test)
  covering the new pure spinner-frame function: non-null/non-empty frames, period-10 cycling,
  variation across the cycle, and the existing tick/init counters. `tn_live_stats_render` itself
  remains manual/screenshot-verified only, same as the rest of this file's terminal-side-effect
  code.
- Verification: gcc + clang × release/test/debug all green (clean between each); golden
  "capital of France" output and tok/s unaffected; a pre-existing, unrelated ASan leak in
  `tokenizer_load_from_gguf`/`strdup` (~1.2MB, same byte/allocation count with and without this
  change) was confirmed present on the prior commit too — not introduced by this work. Screenshot
  at `docs/design/screenshots/07-cli-spinner-and-shimmer-2026-07-16T01-44-29Z.png` shows the
  spinner live during a streaming response; addendum in `docs/design/review-2026-07-15.md`.
- Affected files: `include/cli/live_stats.h`, `src/cli/live_stats.c`, `src/cli/repl.c`,
  `src/cli/banner.c`, `include/cli/banner.h`, `tests/test_live_stats.c` (new),
  `docs/design/review-2026-07-15.md`, `README.md`.

### 2026-07-16 — Phase 22.5: Animated ASCII-art CLI startup banner
- What: identified a remaining UI/UX gap — leading CLI/LLM tools (e.g. Claude Code) show an
  animated ASCII-art name banner on startup; project-zero had none. Added
  `include/cli/banner.h`/`src/cli/banner.c`: a hand-crafted 5-row block font (no figlet/external
  font dependency) covering the letters needed for "PROJECT ZERO", composed programmatically
  (`compose_banner`) so letter concatenation itself can't misalign — only the individual glyphs
  need to be correct. `tn_banner_print()` reveals the banner bottom-row-first over
  `GLYPH_ROWS` frames (~45ms/frame via `nanosleep`, not `usleep` — removed from POSIX.1-2008 and
  undeclared under this project's strict `_POSIX_C_SOURCE=200809L`), giving a "slide up into
  place" effect using only cursor-up + per-line clear (portable across real terminals and
  xterm.js). Wired into `src/cli/main.c`: shown for REPL and `--server` mode, suppressed for
  one-shot `--prompt` runs (matches Claude Code's own interactive-vs-scripted banner behavior),
  gated on `is_tty` so no escape codes ever leak into piped/redirected output. Registered
  `src/cli/banner.c` in `CMakeLists.txt`.
- Bug found and fixed during first real-terminal capture: the glyph separator was a literal
  `' '` character, but the row-printing function only treats `'.'` as blank (anything else prints
  as `'#'`) — every inter-letter gap rendered solid, garbling the banner. Fixed by using `'.'` as
  the separator. Full writeup in `docs/ai/mistakes.md` (2026-07-16).
- Verification: gcc + clang × release/test/debug all green (clean `make clean` between each,
  per the repeated stale-object lesson this project has hit before); golden "capital of
  France" output and tok/s unaffected (this is CLI-startup-only code, no generation-path
  changes); real terminal screenshot captured via `tools/screenshots/cli/capture.mjs` and
  hand-verified against the expected per-letter glyph layout — see
  `docs/design/screenshots/06-cli-startup-banner-2026-07-16T01-03-52Z.png` and the addendum in
  `docs/design/review-2026-07-15.md`.
- Affected files: `include/cli/banner.h`, `src/cli/banner.c` (new), `src/cli/main.c`,
  `CMakeLists.txt`, `docs/design/review-2026-07-15.md`, `README.md`.

### 2026-07-15 — Phase 22.4: Design QA, regression re-verification, README
- What: Wrote `docs/design/ui-ux-principles.md` (the checklist every UI/UX screenshot is graded
  against — best-practice checklist, explicit anti-patterns, a concrete "avoid the generic
  AI-generated look" list, symmetry/Gestalt principles). Built the screenshot capture tooling
  (`tools/screenshots/webui/chat.spec.mjs` — Playwright directly against a live server;
  `tools/screenshots/cli/capture.mjs` — `script(1)` + `@xterm/xterm` in a headless Playwright
  page, chosen over a hand-rolled ANSI-to-HTML converter so cursor movement/in-place updates
  render pixel-accurately). Captured and reviewed real screenshots of both the web UI (empty
  state, streaming, Stop button, sampling params, dark theme) and the CLI (color, markdown
  rendering, live tok/s) — full pass/fail writeup in `docs/design/review-2026-07-15.md`. Added a
  "UI/UX" section to `README.md` with the accepted screenshots.
- The design-QA pass caught two real issues, not just cosmetic nits — proof the gate does what
  it's meant to: (1) `webui/src/lib/ImageUpload.svelte` used a 📎 emoji as its only icon,
  violating the anti-pattern list I'd just written — replaced with an inline SVG. (2) A real bug:
  the REPL's live tok/s status line (`\r` + overwrite) was clobbering the actual streamed
  response text, since both wrote to the same terminal row with no forced newline between them —
  only visible in a real screenshot, not in any unit test. Fixed in `src/cli/live_stats.c` via
  save-cursor/jump-to-last-row/restore-cursor (full writeup in `docs/ai/mistakes.md`).
- Regression safety net (mandatory per the Phase 22 plan): re-ran golden-output checks (Paris/
  Berlin) on both the CLI and HTTP API paths after all Phase 22 structural changes (concurrency
  rearchitecture, vision-pipeline extraction, live-stats fix) — unchanged. Also caught and fixed
  a purely self-inflicted verification mistake twice during this phase: switching compiler/build
  type (e.g. `make debug CC=clang` → `make release CC=gcc`) without `make clean` in between
  leaves stale object files linked into what looks like a "release" build but is actually still
  ASan/UBSan-instrumented — always `make clean` between differing CC/build-type combinations.
- `npm audit` note carried over from 22.2 still applies (dev-dependency-only, SSR-specific
  advisories that don't apply to this static SPA).
- Verified: `make release/test/debug` green on gcc and clang after the `live_stats.c` fix; live
  end-to-end re-verification of the web UI (Playwright) and CLI (real terminal capture) with the
  fix applied, confirmed correct.
- Areas: `docs/design/{ui-ux-principles,review-2026-07-15}.md` (new), `docs/design/screenshots/`
  (new, 6 images), `tools/screenshots/**` (new), `src/cli/live_stats.c` (bug fix),
  `webui/src/lib/ImageUpload.svelte` (emoji→SVG fix), `README.md` (new UI/UX section),
  `docs/ai/mistakes.md`.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.2: Web chat UI (Vite+Svelte, embedded bundle) + image upload
- What: Added `webui/` — a Vite+Svelte SPA (chat window, streaming responses, sampling
  controls, dark/light theme, model info, image upload) built via `npm --prefix webui run
  build`. The build output is embedded into the C binary as a generated, git-committed TU
  (`src/api/webui_bundle_generated.c`, produced by the new standalone tool
  `tools/gen_webui_bundle.c`) so ordinary `make release`/`cmake --build` never need Node —
  only `make webui-bundle` (opt-in, never in `all`/`release`/`test`) does. `GET /` and
  `GET /assets/*` are served by the new `src/api/static_assets.c` (falls back to `index.html`
  for unknown top-level GETs, i.e. client-side SPA routing; supports `--static-dir <path>` to
  serve from disk in dev mode instead of the embedded bundle). `GET /` now serves the UI instead
  of the old health-check-alias JSON — `GET /health` is unchanged and remains the real check.
  New flags: `--web-ui <auto|on|off>` (default auto), `--static-dir <path>`.
- Image upload: extended `chat_request`'s JSON parser (`src/api/json_parse.c`) to accept the
  OpenAI "content parts" array form (`[{"type":"text",...},{"type":"image_url",...}]`) alongside
  plain-string content, added a new `src/api/data_url.c` base64 `data:` URL decoder (none
  existed in the codebase before), and extracted `main.c`'s ~150-line inline vision block (Phase
  34) into a reusable `src/multimodal/vision_pipeline.c` — both the CLI (`--image`) and the
  HTTP API now call the same `vision_pipeline_run()`. The server decodes an uploaded image to a
  temp file, runs the vision pipeline, and injects the result into the KV cache before
  generation, exactly mirroring the CLI's `--image` behavior; gracefully degrades to text-only
  with a logged warning if the server has no `--vision`/`--proj` configured.
- Deliberate deviations from the original plan, both justified and documented here rather than
  silently assumed: (1) the plan's "web UI framework: llama.cpp uses SvelteKit" was matched with
  a **plain Vite+Svelte SPA** (no SvelteKit/SSR) since this server only needs static files — SSR
  would add machinery with no benefit here. (2) "load the vision model once at startup" was
  **not** implemented as a persistent in-memory model; `vision_model_load_encoder/projector` are
  `mmap`-based (confirmed by reading `vision_weights_load.c`), so reloading per-request is cheap
  (an mmap() call + OS page cache), and keeping `vision_pipeline_run()`'s combined load+run
  interface (matching the CLI's existing, already-tested pattern) was simpler and lower-risk than
  splitting load/run across the connection-thread boundary.
- `npm audit` flags 7 moderate/high advisories in the pinned Svelte 4 / Vite 5 dev-dependency
  tree; all are either dev-server-only (esbuild, not exposed since this is a static build with no
  dev server shipped) or Svelte SSR-specific (this is a pure client-side SPA, no SSR at all) — a
  deliberate, documented trade-off against a costly Svelte 5 rewrite for zero applicable benefit.
- Verified: real end-to-end Playwright testing (not just unit tests) against a live server —
  page load, sending a message, streaming response, Stop button (mid-generation cancel), sampling
  params panel, dark/light theme toggle all confirmed working via screenshots. `curl`/`curl --raw`
  confirmed `GET /`, `/assets/*`, `--web-ui off` (404s), and `--static-dir` dev-mode serving.
  Golden output (Paris/Berlin) re-verified unaffected by the vision-pipeline extraction on both
  the CLI and (new) HTTP API paths. Image upload's graceful-degradation path (no
  `--vision`/`--proj` configured) verified end-to-end with a real base64 PNG; the full
  image-understanding path is *not* end-to-end verified in this environment (no vision.bin/
  projector.bin/vision-capable GGUF available to download) — the extracted logic is otherwise
  unchanged from the already-tested CLI code path (`tests/test_vision_components.c`,
  `test_vision_e2e.c`), so this is a scoped, disclosed gap, not a silent one.
- `make release/test/debug` green on gcc and clang throughout. New tests:
  `tests/test_static_assets.c` (hand-written fake manifest, not the real bundle — keeps
  `make test` Node-free), `tests/test_data_url.c`, and content-parts-parsing cases added to
  `tests/test_api_server.c`.
- Areas: `webui/**` (new), `tools/gen_webui_bundle.c` (new), `src/api/{static_assets,data_url,
  webui_bundle_generated}.c` + `include/api/{static_assets,data_url,webui_bundle}.h` (new),
  `src/multimodal/vision_pipeline.c` + `include/multimodal/vision_pipeline.h` (new, extracted
  from `main.c`), `src/api/{json_parse,http_server}.c`/`include/api/{chat_request,server_config}.h`
  (content-parts parsing, vision wiring), `src/cli/{args,main}.c`/`include/cli/args.h` (new
  flags), `CMakeLists.txt` (new sources), `Makefile` (`webui-bundle` target), `.gitignore`
  (`webui/node_modules/`, `webui/dist/`).
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.3: CLI/REPL polish (color, progress, live tok/s, markdown)
- What: Added `--color <auto|always|never>` (respects `NO_COLOR`), a coarse 4-stage model-load
  progress indicator (TTY in-place `\r` updates, plain one-line-per-stage otherwise), a live
  tok/s indicator updated per-token during REPL generation, and incremental markdown rendering
  (bold, inline code, fenced code blocks) for REPL output — handling constructs whose delimiters
  are split across separate streamed token pieces (e.g. `"**bo"` + `"ld**"`). Regrouped
  `--help` output into sections (Model & Generation / Hardware / Server / Multimodal / Memory &
  RAG / Output) with worked examples. Only the REPL path is affected — the one-shot `--prompt`
  path and the HTTP API's SSE callback are untouched, matching the plan's scoping.
- A manual TTY smoke test (via `script`) surfaced a real UX rough edge in the first cut of the
  markdown renderer: an unterminated code fence (common whenever `max_tokens` cuts generation off
  mid-block) buffered the ENTIRE rest of the response until the final flush, defeating live
  streaming. Fixed with a bounded safety valve (`MD_MAX_PENDING_UNCLOSED`, 4 KiB) — an opening
  marker that hasn't found its close within that many buffered bytes is flushed as plain text
  instead of waiting indefinitely, covered by a new test
  (`test_unclosed_fence_eventually_flushes_without_waiting_for_end`).
- Why: closes the CLI-polish gap vs. leading engines (colored output, progress bars, live stats,
  markdown rendering) identified in the original UI/UX audit.
- Areas: `src/cli/{color,progress,live_stats,md_render}.c` + matching headers (new),
  `src/cli/{args,main,repl}.c`/`include/cli/args.h` (new `--color` flag, progress-stage hooks,
  REPL composite callback), `CMakeLists.txt` (four new CLI sources registered),
  `tests/test_{color,progress,md_render}.c` (new).
- Result: `make release/test/debug` green on gcc and clang; new unit tests cover color
  resolution, progress-line formatting, and markdown rendering (including the split-delimiter
  and unclosed-construct edge cases); manual REPL smoke test under a real pty (`script`)
  confirmed live progress stages, live tok/s updates, and markdown styling all render correctly;
  one-shot `--prompt` golden output (Paris, 64 tok/s) unaffected.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.1: HTTP API hardening + concurrency rearchitecture
- What: Added CORS (`--cors`/`--cors-origin`), optional API-key auth (`--api-key`), `/metrics`
  (Prometheus text exposition, `--metrics`), `/docs` + `/openapi.json` (static OpenAPI 3.0 +
  hand-rolled docs page, no Swagger-UI dependency), and `POST /v1/chat/completions/cancel`
  (stop an in-flight generation). Rearchitected `http_server.c` from one serial listener thread
  to a detached-per-connection-thread model, with a `generation_mutex` serializing only the
  actual `generate_with_callback` calls (a second concurrent chat request gets `429` immediately
  rather than blocking). `generate_with_callback`'s `TokenCallback` now returns `int` (0 =
  continue, nonzero = stop early) so cancellation can actually halt the generation loop, not just
  the client's view of it. Raised `HTTP_MAX_BODY_BYTES` 512 KiB → 8 MiB ahead of Phase 22.2's
  image uploads.
- Real end-to-end testing (live server + `curl`/`curl --raw` against the SmolLM2-135M demo
  model) surfaced and fixed four pre-existing HTTP protocol bugs the "untested socket layer"
  label had been masking — see `docs/ai/mistakes.md` (2026-07-15 entry) for the full list
  (recv-loop hang on bodyless GETs, `Content-Length: 0` sent before the real non-streaming body,
  a false `Transfer-Encoding: chunked` claim over unframed bytes, and no `SIGPIPE` handling).
  Also fixed a bug introduced by the new cancel feature itself: the id registered internally was
  the raw id, but clients only ever see the `"chatcmpl-"`-prefixed id in the stream, so a client
  echoing it back to the cancel endpoint would never match — fixed by registering under the same
  public, prefixed id the client observes.
- Why: closes the biggest UI/UX gap vs. leading engines (CORS/auth/metrics/docs/cancel) and is a
  prerequisite for Phase 22.2's web UI (needs real CORS + a working stop button).
- Areas: `src/api/{server_config,cors,auth,metrics,openapi,cancel,http_server,sse_stream}.c` +
  matching headers, `src/transformer/generate.c`/`include/transformer/generate.h` (callback
  return type), `src/cli/{args.c,main.c}`/`include/cli/args.h` (new flags), `CMakeLists.txt`
  (six new API sources registered), `tests/test_{cors,auth,metrics,cancel,openapi}.c` (new).
- Result: `make release/test/debug` green on gcc and clang (ASan/UBSan); a ThreadSanitizer build
  of the full engine showed zero race warnings under concurrent traffic (metrics/models/health/
  docs requests firing while a generation held the mutex, correctly getting `429` for concurrent
  generation attempts); golden output ("Paris"/"Berlin") unchanged through both the CLI and the
  now-hardened API; manual verification of CORS allow/deny, auth 401/200, streaming, non-
  streaming, and mid-stream cancellation all confirmed working via live curl sessions.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-07-15 — Phase 22.0: docs groundwork for Web UI & API/DX hardening
- What: Recorded the Phase 22 plan (web chat UI via Vite+Svelte embedded in the binary, HTTP API
  hardening with a concurrency rearchitecture, CLI/REPL polish, mandatory design-QA + regression
  screenshots) before any code changes. Justified and documented the `api` scope in
  `tool-sync-policy.md` (adapter files to be added once 22.1 lands).
- Why: `docs/ai/**` is canonical and updated before code per policy; this is a large multi-phase
  effort and needs the decision trail written down first.
- Areas: `docs/ai/decision-log.md`, `docs/ai/project-overview.md`, `docs/ai/tool-sync-policy.md`.
- Branch: `claude/project-zero-ui-ux-gaps-h54mdc`.

### 2026-06-19 — Portable `make dist` build + GitHub Release pipeline
- What: Added a portable distribution build and a release workflow that attaches a prebuilt
  x86-64 Linux binary to a GitHub Release. New `make dist` target compiles the bulk at
  `-march=x86-64-v2` with per-file SIMD ISA flags (AVX2/AVX-512/VNNI) so runtime `simd_dispatch`
  lights up the best tier on the host; `simd_dispatch.c` is compiled at the baseline with
  `-DTN_FORCE_DISPATCH_ALL` (new guard, no SIMD codegen there) so all branches are present;
  static `-static-libstdc++ -static-libgcc` leaves only libc/libm deps. Added a `--version`/`-v`
  flag (works without `--model`) and a `-DPZ_VERSION` build stamp (banner no longer hardcodes
  "Phase 16"). CMake gains an off-by-default `PZ_DIST` option mirroring the Makefile.
- Why: user asked for a prebuilt x86-64 binary on a GitHub Release, tested thoroughly; the
  existing `-march=native` release is not distributable on varied CPUs.
- Areas: `Makefile` (dist target, per-TU ISA rules, version stamp), `src/math/simd_dispatch.c`
  (`TN_FORCE_DISPATCH_ALL`), `src/cli/{args.c,main.c}` + `include/cli/args.h` (`--version`),
  `CMakeLists.txt` (`PZ_DIST`, `PZ_VERSION`), `.github/workflows/release.yml` (new),
  `.github/workflows/ci.yml` (dist build-check), `docs/RELEASING.md` (new).
- Result: gcc release/test(46)/debug/dist and clang release/debug/dist green; portable binary
  links only libc/libm; golden output (France→Paris, Germany→Berlin) correct across
  scalar/avx2/avx512f/vnni and T=1/2/8 on the SmolLM2-135M F16 model.
- Commit/PR: on branch `claude/x86-64-github-release-8xduj2`.

### 2026-06-14 — Docs reflect dense GGUF support (SmolLM2 + generic loader)
- What: README, ROADMAP, and project-overview said the engine runs only BitNet and
  DeepSeek-V2-Lite, but the benchmark docs (`.claude/BENCHMARK_SUMMARY.md`,
  `docs/PERFORMANCE_CEILING_REPORT.md`) already benchmark **SmolLM2-135M-Instruct F16**
  (dense GGUF) up to 83.79 tok/s, and `config_from_gguf()` in `src/core/gguf_loader.c` is
  architecture-agnostic. Added a third support tier: dense GGUF transformers (Llama-family)
  via the generic loader, with SmolLM2 as the verified model and other architectures flagged
  as loads-but-untested. MoE/MLA acceleration remains DeepSeek-V2-specific.
- Why: docs understated actual, already-tested capability; user asked for the correct picture.
- Areas: `README.md` (intro, new "Dense GGUF Models" section, footer), `.github/ROADMAP.md`
  (perf snapshot), `docs/ai/project-overview.md` (Purpose). Lean adapters (AGENTS/copilot/
  GEMINI/CLAUDE) left as-is per tool-sync-policy — they describe the targeted/special-cased
  architectures, not an exhaustive model list. Historical benchmark addenda left untouched.
- Branch: `claude/readme-llm-support-docs-3tg13v`.

### 2026-06-14 — README accuracy pass + repo best-practices + docs reorg
- What: (1) Corrected README intro to match canonical scope (BitNet + DeepSeek-V2-Lite
  GGUF + vision/agentic/RAG), kept "written in C", reframed Python as temporary
  dev/test tooling (zero-Python goal), added LLM-agnostic goal. (2) Reconciled the
  Phase 21 HTTP API claim to 🔄 partial/experimental across README, ROADMAP, and
  project-overview (it is real and wired but serial/loopback-only/untested-in-CI).
  (3) Added community-health files: `.github/CODEOWNERS`, `.github/dependabot.yml`,
  `.github/ISSUE_TEMPLATE/config.yml`, `.editorconfig`, `CITATION.cff`. (4) Moved 27
  archival/design/report `.md` files out of the repo root into
  `docs/{architecture,phases,reports,weight-loading}/` and `docs/`, leaving 8 entry-point
  docs at root; rewrote all inbound markdown links path-aware and fixed 4 dangling links
  (verified 0 dangling repo-wide).
- Why: README/roadmap/overview contradicted each other and the tree; root had 35 `.md`
  files hurting discoverability; repo was missing standard GitHub best-practice files.
- Areas: `README.md`, `.github/ROADMAP.md`, `docs/ai/project-overview.md`, `.editorconfig`,
  `.github/CODEOWNERS`, `.github/dependabot.yml`, `.github/ISSUE_TEMPLATE/config.yml`,
  `CITATION.cff`, and `docs/{architecture,phases,reports,weight-loading}/**`.
- Branch: `claude/readme-accuracy-review-y9jk7u`.

### 2026-06-07 — Document branch-hygiene convention
- What: Added a "Version control & branch hygiene" section to `engineering-rules.md` (delete
  merged branches; enable auto-delete-head-branches; avoid flag/placeholder branches; don't
  commit artifacts/models/logs).
- Why: post-merge cleanup surfaced redundant branches; this sandbox's git proxy blocks ref
  deletion, so the convention + the repo auto-delete setting prevent future accumulation.
- Areas: `docs/ai/engineering-rules.md`. Canonical-only (adapters stay lean per tool-sync-policy).

### 2026-06-07 — Cross-tool AI development system
- What: Added `docs/ai/**` canonical docs + Claude/Copilot/Antigravity adapters.
- Why: one source of truth; continuity across Claude Code, GitHub Copilot, Antigravity.
- Areas: `docs/ai/`, `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`, `.github/`, `AGENTS.md`,
  `gemini/GEMINI.md`, `.agents/`.
- Commit/PR: (this change) — see commit checkpoints in PR to `master`.

### 2026-06-07 — Green CI + regression verification (PR #6, merged `cb9fa52`)
- What: Fixed the CI cascade and verified no regression across SmolLM2/BitNet/DeepSeek.
- Why: CI had never run to completion; ensure merge-safety and no regression.
- Areas: `tests/test_blackbox.c`, `tests/audit_sliding_window_crash.c`,
  `tests/test_vision_components.c`, `Makefile`, `src/core/run_state.c`,
  `docs/REGRESSION_VERIFICATION_2026-06-07.md`.
- Result: all 7 CI checks green on PR #6 and on `master`; secrets scan clean (215 commits).

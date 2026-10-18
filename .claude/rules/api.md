# Claude rule: api (HTTP server, src/api/)

Adapter for the API-layer decisions in `docs/ai/decision-log.md` (Phase 22) and
`docs/ai/mistakes.md` (2026-07-15 socket-layer bugs; 2026-07-16 shutdown-path bugs).

- Hand-rolled HTTP/1.1 over raw sockets — no external HTTP library. Loopback-only by default.
- CORS (`--cors`/`--cors-origin`) and API-key auth (`--api-key`) are both **off by default** —
  don't change this default; it exists so loopback experimentation stays open.
- Only one `generate_with_callback` runs at a time, serialized by `ApiContext.generation_mutex`
  (via `ctx->gen_queue`: a second concurrent chat request gets `429` by default, or waits in FIFO
  order with `--queue-size N` — streaming waiters get `: queued N` comments). All other routes
  (static assets, `/metrics`, `/docs`, `/health`, cancel) must never touch that mutex.
- The web UI bundle (`src/api/webui_bundle_generated.c`) is **generated and committed** — never
  hand-edit it; edit `webui/src` and run `make webui-bundle`.
- Image uploads (`image_url` content parts) decode via `src/api/data_url.c` and run through
  `src/multimodal/vision_pipeline.c` (shared with the CLI's `--image`) — requires the server to
  be started with `--vision`/`--proj`; otherwise images are logged and ignored (text-only).
- A component documented as "untested" is not verified — real socket/HTTP code needs a live
  `curl`/`curl --raw` smoke test, not just the pipe-based logic tests in `tests/test_api_server.c`
  (see the 2026-07-15 mistakes.md entry: four real bugs hid behind that label).
- `signal(SIGPIPE, SIG_IGN)` is required in `api_server_start()` — any client disconnect mid-write
  otherwise kills the whole process.
- `api_server_stop()` must call `shutdown(ctx->server_fd, SHUT_RDWR)` **before** `close()` — a
  bare `close()` from another thread does not reliably wake a concurrent blocking `accept()` on
  Linux (confirmed by an actual hang, see the 2026-07-16 mistakes.md entry). `--server` mode's
  Ctrl+C/SIGTERM handling in `main.c` (a `sigaction`-installed handler + a `while
  (!g_shutdown_requested) pause();` loop, not a bare `pause()`) is what makes this path reachable
  at all — don't revert to a bare `pause()`, it silently makes all of `api_server_stop()` and
  `main()`'s later cleanup dead code again.
- Generation knobs: `GET /v1/config` publishes server defaults (CLI flags), valid ranges and
  feature flags; the web UI renders every control from it and hardcodes nothing. Requests
  override any subset (`set_mask` → `request_params_resolve()`); absent fields use the server
  default. Chat-template models receive the **full** message history (never just the last user
  turn). Each chat request resets `run_state->current_pos` to 0.
- `POST /v1/watermark/detect` and the discovery routes never take `generation_mutex`
  (tokenizer encode is read-only).
- Phase 21.5 prefix cache (`ApiContext.prefix_cache`, `--prompt-cache`, `cache_prompt`): any new
  code path that writes the RunState's KV outside `generate_ex()` must `prefix_cache_reset()`
  first, or the next request will reuse stale K/V.
- Phase 38 update notices: `GET /v1/update` / `POST /v1/update/check` read the update service
  (`update/update_check.h`) and never touch `generation_mutex`. Notify-only — never add code that
  downloads or replaces the binary without signed release metadata (decision-log 2026-10-18).
- Brand wordmark: `/v1/config ui.logo_ascii` is rendered server-side from `--ui-title` by
  `brand/block_font.c` (same font as the CLI banner) — never hardcode the logo in `webui/src`.

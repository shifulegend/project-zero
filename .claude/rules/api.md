---
paths:
  - "src/api/**"
  - "include/api/**"
  - "webui/**"
  - "tests/test_api_server.c"
  - "tests/test_cors.c"
  - "tests/test_auth.c"
  - "tests/test_metrics.c"
  - "tests/test_cancel.c"
  - "tests/test_openapi.c"
---
# Claude rule: api (HTTP server, src/api/)

Adapter for the API-layer decisions in `docs/ai/decision-log.md` (Phase 22) and
`docs/ai/mistakes.md` (2026-07-15 socket-layer bugs; 2026-07-16 shutdown-path bugs).

- Hand-rolled HTTP/1.1 over raw sockets — no external HTTP library. Loopback-only by default.
- CORS (`--cors`/`--cors-origin`) and API-key auth (`--api-key`) are both **off by default** —
  don't change this default; it exists so loopback experimentation stays open.
- Generation admission goes through `ctx->gen_queue`, a FIFO of units (capacity = `--parallel`
  slots, default 1). A plain chat request takes 1 unit and runs in `api/batch_scheduler` (Phase 23
  stage 2b: one thread, per-slot RunState/prefix cache, one batched forward per step); image input,
  best-of, tree search and cache reset take **all** units plus `ApiContext.generation_mutex`, so the
  scheduler is idle while they use `ctx->run_state` (slot 0). No free unit → `429`, or FIFO wait
  with `--queue-size N` (streaming waiters get `: queued N` comments). All other routes (static
  assets, `/metrics`, `/docs`, `/health`, cancel) stay off units and that mutex: they would
  otherwise block behind a long generation and `/health` would stop answering. The scheduler may
  read slot state only while one of its own requests holds a unit.
- The web UI bundle (`src/api/webui_bundle_generated.c`) is **generated and committed** —
  `make webui-bundle` overwrites it, so edit `webui/src` and regenerate instead of hand-editing.
- Image uploads (`image_url` content parts) decode via `src/api/data_url.c` and run through
  `src/multimodal/vision_pipeline.c` (shared with the CLI's `--image`) — requires the server to
  be started with `--vision`/`--proj`; otherwise images are logged and ignored (text-only).
- A component documented as "untested" is not verified — real socket/HTTP code needs a live
  `curl`/`curl --raw` smoke test, not just the pipe-based logic tests in `tests/test_api_server.c`
  (see the 2026-07-15 mistakes.md entry: four real bugs hid behind that label).
- `signal(SIGPIPE, SIG_IGN)` stays in `api_server_start()`: without it any client disconnect
  mid-write kills the whole process.
- `api_server_stop()` cancels every generation (`cancel_registry_shutdown`), waits for the queue
  to drain and stops the scheduler before returning — `main()` frees the model right after.
- `api_server_stop()` calls `shutdown(ctx->server_fd, SHUT_RDWR)` **before** `close()` — a
  bare `close()` from another thread does not reliably wake a concurrent blocking `accept()` on
  Linux (confirmed by an actual hang, see the 2026-07-16 mistakes.md entry). `--server` mode's
  Ctrl+C/SIGTERM handling in `main.c` (a `sigaction`-installed handler + a `while
  (!g_shutdown_requested) pause();` loop, not a bare `pause()`) is what makes this path reachable
  at all — don't revert to a bare `pause()`, it silently makes all of `api_server_stop()` and
  `main()`'s later cleanup dead code again.
- Chat template: `--chat-template-file` sets the server's; requests may override it with
  `"chat_template"` (≤ 64 KiB, `CHAT_SET_CHAT_TEMPLATE`) unless `--no-request-template`;
  `/v1/config chat_template {source,text,request_override,max_bytes}` feeds the web UI editor.
- Tree search (Phase 30): server `--mcts* --mcts-max --prm*`; request `"mcts"`
  (`CHAT_SET_MCTS`, `request_mcts_resolve`); `ApiContext.prm` is the judge. Runs under
  `generation_mutex` like any generation.
- Generation knobs: `GET /v1/config` publishes server defaults (CLI flags), valid ranges and
  feature flags; the web UI renders every control from it and hardcodes nothing. Requests
  override any subset (`set_mask` → `request_params_resolve()`); absent fields use the server
  default. Chat-template models receive the **full** message history (not just the last user
  turn: the template needs the whole conversation to render correctly). Each chat request resets
  `run_state->current_pos` to 0.
- `POST /v1/watermark/detect` and the discovery routes do without `generation_mutex`
  (tokenizer encode is read-only, so they stay responsive during generation).
- Phase 21.5 prefix cache (`ApiContext.prefix_cache`, `--prompt-cache`, `cache_prompt`): any new
  code path that writes the RunState's KV outside `generate_ex()` resets the cache first
  (`prefix_cache_reset()`); otherwise the next request reuses stale K/V.
- Phase 38 update notices: `GET /v1/update` / `POST /v1/update/check` read the update service
  (`update/update_check.h`) without `generation_mutex` (they only read cached state). Notify-only —
  never add code that downloads or replaces the binary without signed release metadata
  (decision-log 2026-10-18).
- Brand wordmark: `/v1/config ui.logo_ascii` is rendered server-side from `--ui-title` by
  `brand/block_font.c` (same font as the CLI banner), so the web UI takes the logo from
  `/v1/config` rather than a copy in `webui/src` (one source; `--ui-title` changes both).
- Phase 40.3 settings provenance: `/v1/config config.sources` (default/file/env/cli, recorded by
  `pz_config_prepare_argv`) and `POST /v1/config/defaults` (`api/config_endpoint.c`) writing the
  `pz config` file for the next start. State-changing routes on a key-less server must pass
  `api_config_write_allowed()` (JSON content type + same-origin/CORS-allowed Origin); a new
  saveable default needs a row in `g_map` there **and** a `g_schema` key (DoD #6).

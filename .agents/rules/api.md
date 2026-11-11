# Antigravity scoped rules: api / HTTP server (adapter for docs/ai/decision-log.md Phase 22, mistakes.md)

- Hand-rolled HTTP/1.1 over raw sockets, no external HTTP library; loopback-only by default.
- CORS (`--cors`/`--cors-origin`) and auth (`--api-key`) are off by default — don't change; keeps
  loopback experimentation open.
- Admission via `ctx->gen_queue` units (capacity `--parallel`, default 1): plain chat = 1 unit in
  `api/batch_scheduler`; image/best-of/MCTS/cache reset = all units + `generation_mutex`. No free
  unit → `429` (or FIFO wait with `--queue-size`); other routes stay off units and the mutex, so
  `/health` and friends keep answering during a long generation.
- `src/api/webui_bundle_generated.c` is generated/committed — `make webui-bundle` overwrites it,
  so edit `webui/src` and regenerate instead of the generated file.
- Image uploads (`image_url` content parts) decode via `src/api/data_url.c` and run through
  `src/multimodal/vision_pipeline.c` (shared with the CLI's `--image`); needs `--vision`/`--proj`,
  otherwise images are logged and ignored (text-only).
- A component documented as "untested" is not verified — smoke-test real socket/HTTP code with
  `curl`/`curl --raw` against a live server (2026-07-15: four real protocol bugs were hiding
  behind that exact label).
- `signal(SIGPIPE, SIG_IGN)` stays in `api_server_start()`: without it a client disconnect
  mid-write kills the whole process.
- `api_server_stop()` cancels every generation (`cancel_registry_shutdown`), drains the queue and
  stops the scheduler before returning (`main()` frees the model right after); it calls
  `shutdown(ctx->server_fd, SHUT_RDWR)` before `close()`, because a bare `close()` from another
  thread does not reliably wake a blocking `accept()` on Linux (2026-07-16 hang). `--server`
  mode's Ctrl+C/SIGTERM path (`sigaction` handler + `while (!g_shutdown_requested) pause();`
  loop, not a bare `pause()`) is what makes that cleanup reachable at all.
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
  turn: the template needs the whole conversation). Each chat request resets
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

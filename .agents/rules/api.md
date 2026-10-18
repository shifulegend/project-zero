# Antigravity scoped rules: api / HTTP server (adapter for docs/ai/decision-log.md Phase 22, mistakes.md)

- Hand-rolled HTTP/1.1 over raw sockets, no external HTTP library; loopback-only by default.
- CORS (`--cors`/`--cors-origin`) and auth (`--api-key`) are off by default — don't change; keeps
  loopback experimentation open.
- Only one `generate_with_callback` runs at a time via `ApiContext.generation_mutex` trylock
  (second concurrent request → `429`); other routes (assets/metrics/docs/health/cancel) never
  touch that mutex.
- `src/api/webui_bundle_generated.c` is generated/committed — edit `webui/src` + `make
  webui-bundle` instead of the generated file.
- A component documented as "untested" is not verified — smoke-test real socket/HTTP code with
  `curl`/`curl --raw` against a live server (2026-07-15: four real protocol bugs were hiding
  behind that exact label).
- `signal(SIGPIPE, SIG_IGN)` must stay in `api_server_start()` — a client disconnect mid-write
  otherwise kills the whole process.
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

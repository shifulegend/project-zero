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

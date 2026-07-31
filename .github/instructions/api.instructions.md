---
applyTo: "src/api/**,include/api/**,tests/test_api_server.c,tests/test_cors.c,tests/test_auth.c,tests/test_metrics.c,tests/test_cancel.c,tests/test_openapi.c"
---
# Copilot scoped rules: api / HTTP server (adapter for docs/ai/decision-log.md Phase 22, mistakes.md)

- Hand-rolled HTTP/1.1 over raw sockets, no external HTTP library; loopback-only by default.
- CORS/auth are off by default — don't change; keeps loopback experimentation open.
- Only one `generate_with_callback` runs at a time, serialized via `ApiContext.generation_mutex`
  trylock (second concurrent request → `429`); other routes never touch that mutex.
- `src/api/webui_bundle_generated.c` is generated/committed — edit `webui/src` + `make
  webui-bundle` instead.
- "Untested" socket code is not verified code — smoke-test with real `curl`/`curl --raw` before
  trusting it (2026-07-15: four real protocol bugs were hiding behind that label).
- `signal(SIGPIPE, SIG_IGN)` must stay in `api_server_start()`.

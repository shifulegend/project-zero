# Web UI & CLI Usage Guide

A practical how-to for the UI/UX surfaces added in Phase 22 (web chat UI, HTTP API hardening,
CLI/REPL polish). For the design rationale and screenshots, see the [README UI/UX
section](../README.md#ui-ux); for the design-QA writeup, see
[`docs/design/review-2026-07-15.md`](design/review-2026-07-15.md).

## Web chat UI

### Starting it

```bash
./adaptive_ai_engine --model models/smollm2.gguf --server --port 8080
# open http://127.0.0.1:8080/ in a browser
```

`--web-ui <auto|on|off>` controls whether `GET /` serves the chat UI (default `auto`: on
whenever `--server` is set). `--static-dir <path>` serves the UI straight off disk instead of
the binary-embedded bundle — useful while editing `webui/src` (pair with `npm run dev` inside
`webui/` for hot reload, or `make webui-bundle` to regenerate the committed embedded bundle
after a change).

### Using it

- **Composer** — type a message, press **Send** (or Enter) to stream a response.
- **Stop** — replaces Send while a response is streaming; click it to cancel generation
  mid-stream (calls `POST /v1/chat/completions/cancel` under the hood).
- **Params** — toggles a panel with three sliders: **Temperature** (0–2), **Top-p** (0–1), and
  **Max tokens** (16–2048), applied to the next message you send.
- **Theme toggle** (top bar) — cycles dark → light → auto (follows OS preference).
- **Image upload** (paperclip icon, composer) — only rendered when the server reports vision
  is configured; attach an image, then send a message referencing it. Requires starting the
  server with both `--vision <path>` and `--proj <path>`.
- **Model info panel** — shows the loaded model's architecture, quantization, and context size
  (from `GET /v1/models`).

### Enabling image upload

```bash
./adaptive_ai_engine --model models/<vision-model>.gguf \
    --vision models/<vision-tower>.gguf --proj models/<projector>.gguf \
    --server --port 8080
```

Without `--vision`/`--proj`, the attach control is hidden entirely (not just disabled) —
the web UI only ever offers controls the running server can actually serve.

## CLI / REPL

### Starting it

```bash
./adaptive_ai_engine --model models/smollm2.gguf
```

With no `--prompt`, this drops into the interactive REPL. On a real terminal (TTY) you'll see,
in order: the animated "PROJECT ZERO" ASCII banner (bottom-up reveal + a brief shimmer), the
hardware profile table, then the `> ` prompt.

### What's on screen while generating

- **Markdown/code rendering** — `**bold**`, `` `inline code` ``, and fenced code blocks are
  rendered with ANSI styling as tokens stream in (falls back to plain text if a construct never
  closes, e.g. `max-tokens` cuts a code fence short).
- **Live status line** — a bold-cyan spinner (10-frame braille cycle, advances once per token)
  followed by `[N tok, X.X tok/s]`, pinned to the terminal's last row so it never overwrites the
  response text above it.

### REPL commands

| Command | Effect |
|---|---|
| `/quit`, `/exit` | Exit the REPL |
| `/context` | Show KV-cache usage (tokens used / max) |
| `/think` | Toggle reasoning mode |
| `/agent <prompt>` | Run the agentic tool-use loop |
| `/memory list` | List stored RAG memories (needs `--memory-db`) |
| `/memory search <query>` | Search stored memories |
| `/memory save <text>` | Manually save a memory |
| `/help` | Print this list |

### Color and non-interactive use

`--color <auto|always|never>` controls ANSI output (default `auto`: on when stdout is a TTY and
`NO_COLOR` is unset). The banner and spinner are TTY-gated independently of color — piping
output (`... | less`, `... > out.txt`) or passing `--prompt` (one-shot/scripted mode) suppresses
the banner entirely and keeps the live-stats line off stderr's normal buffering path; only the
final `generate_with_callback` summary line remains, matching plain scripted output elsewhere in
the CLI.

## HTTP API reference (for scripting / integration)

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completion (streaming via SSE, or full JSON) |
| `POST /v1/chat/completions/cancel` | Cancel the in-flight generation |
| `GET /v1/models` | Model metadata (architecture, quantization, context size) |
| `GET /health` | Liveness check (JSON) |
| `GET /metrics` | Prometheus text exposition (needs `--metrics`) |
| `GET /docs`, `GET /openapi.json` | Interactive API docs / OpenAPI 3.0 spec |
| `GET /`, `GET /assets/*` | The web UI (see `--web-ui` above) |

CORS (`--cors`, `--cors-origin <origin>`, repeatable or `*`) and API-key auth (`--api-key
<key>`, checked via `Authorization: Bearer <key>`) are both **off by default** — enable them
explicitly for anything beyond loopback experimentation. Only one chat completion runs at a
time; a second concurrent request gets `429 Too Many Requests` rather than queueing or blocking
other routes (static assets, `/metrics`, `/docs`, `/health` are never blocked by an in-flight
generation).

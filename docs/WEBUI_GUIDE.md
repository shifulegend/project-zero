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
- **Settings** — toggles a panel with every generation knob, applied to the next message you
  send. Defaults, ranges and which sections appear all come from the server (`GET /v1/config`),
  so starting the server with e.g. `--temperature 0.2 --top-k 40 --system-prompt "…"` changes
  what the UI starts from:
  - *Sampling*: temperature, top-p, top-k, min-p, max tokens (capped by the model context or
    `--max-tokens-limit`), best of N (generate N replies and keep the one the model finds most
    likely — N× slower, the reply appears once chosen).
  - *Repetition*: repetition / presence / frequency penalty and the penalty window.
  - *Output*: random or fixed seed (reproducible replies), JSON mode, reuse prompt cache
    (skips re-reading the history already in the KV cache — same output, faster turns; the line
    below it shows how many conversations the server keeps snapshots of, with **Clear**), stop
    sequences.
  - *Watermark*: watermark replies, key (blank = the server's `--watermark-key`), γ and δ.
  - *System prompt*: applied to every conversation (chat-template models).
  Your changes are remembered in this browser; the badge on **Settings** counts how many differ
  from the server defaults, and **Reset to server defaults** discards them.
- **New chat** — clears the conversation (the full history is otherwise sent with every turn,
  so the model remembers earlier messages).
- **Watermark** — paste text to check whether it carries this server's watermark (z-score
  against the threshold; uses `POST /v1/watermark/detect`).
- **Reply footer** — completion/prompt token counts (with how many prompt tokens came from the
  prompt cache) and why the reply ended (complete, hit max tokens, stopped).
- **Theme toggle** (top bar) — cycles dark → light → auto (follows OS preference).
- **Image upload** (paperclip icon, composer) — only rendered when the server reports vision
  is configured; attach an image, then send a message referencing it. Requires starting the
  server with both `--vision <path>` and `--proj <path>`.
- **Model info panel** — shows the loaded model file, layer count, width and context size, plus
  vision/MoE tags (from `GET /v1/config`'s `model` object, same data as `GET /v1/models`).
- **Branding** — the top-bar name comes from `--ui-title` (default `project-zero`). It is shown
  as the block wordmark from the terminal banner (`>_` + `PROJECT ZERO` in `#` glyphs, accent
  green on a transparent background) in the top bar and, large, on the empty chat screen. The
  server renders it from the title with the same font as the CLI (`GET /v1/config` →
  `ui.logo_ascii`); `--ui-logo text`, or a title with letters the font doesn't have, shows the
  plain name instead. Assets: `docs/design/brand/`.

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
| `GET /v1/models` | Model metadata (file name, context length, layers, dims, vision/MoE flags) |
| `GET /v1/config` | Server generation defaults, valid ranges, feature flags, UI title |
| `GET /v1/prompt_cache`, `POST /v1/prompt_cache/clear` | Prompt-cache snapshot stats / drop all snapshots |
| `POST /v1/watermark/detect` | `{text, key?, gamma?, z_threshold?}` → watermark z-score |
| `GET /health` | Liveness check (JSON) |
| `GET /metrics` | Prometheus text exposition (needs `--metrics`) |
| `GET /docs`, `GET /openapi.json` | Interactive API docs / OpenAPI 3.0 spec |
| `GET /`, `GET /assets/*` | The web UI (see `--web-ui` above) |

Chat completion requests accept, besides `messages`/`stream`/`max_tokens`/`temperature`/`top_p`:
`top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`,
`penalty_last_n`, `seed`, `stop` (string or array), `response_format: {"type":"json_object"}`
`watermark: {"enabled", "key", "gamma", "delta"}`, `cache_prompt` (default from
`--prompt-cache`, on) and `best_of` (default `--best-of`, capped by `--best-of-max`; the response
then includes a `best_of` object with each candidate's score). Omitted fields use the server defaults
(the CLI flags). Responses carry real `usage` token counts and `finish_reason`
(`stop` | `length` | `cancelled`); the final streaming chunk includes `usage` too.

CORS (`--cors`, `--cors-origin <origin>`, repeatable or `*`) and API-key auth (`--api-key
<key>`, checked via `Authorization: Bearer <key>`) are both **off by default** — enable them
explicitly for anything beyond loopback experimentation. Only one chat completion runs at a
time; by default a second concurrent request gets `429 Too Many Requests`; start the server with
`--queue-size N` (and optionally `--queue-timeout S`) to let up to N requests wait their turn instead
(the web UI shows the queue position). A server started with `--remote-layers` (Phase 28) shows a
`dist N nodes` tag in the header; hover it for each node's layer range. `--kv-offload <dir>` (Phase 31)
shows a `KV on disk` tag.

**Update notices (Phase 38).** The server checks this project's GitHub Releases in the background
at most once a day and the UI shows `>_ update available vX → vY · release notes · dismiss` when a
newer release exists (dismissed per version). Settings → Updates shows the status, the server's
interval/channel/repo, a **Check now** button and a banner toggle. Nothing is ever downloaded or
installed — verify the release's `SHA256SUMS`. Operators control it with `--no-update-check`,
`--update-interval <h>`, `--update-channel auto|stable|prerelease`, `--update-repo owner/name`,
`--update-api <url>` or `PZ_NO_UPDATE_CHECK=1`; it is always off when `CI` is set.
`adaptive_ai_engine --check-update` checks once from the shell (exit 3 = update available). Either way, other routes (static assets, `/metrics`, `/docs`, `/health` are never blocked by an in-flight
generation).

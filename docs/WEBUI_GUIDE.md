# Web UI & CLI Usage Guide

A practical how-to for the UI/UX surfaces added in Phase 22 (web chat UI, HTTP API hardening,
CLI/REPL polish). For the design rationale and screenshots, see the [README UI/UX
section](../README.md#ui-ux); for the design-QA writeup, see
[`docs/design/review-2026-07-15.md`](design/review-2026-07-15.md).

## Web chat UI

### Starting it

```bash
./pz --model models/smollm2.gguf --server --port 8080
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
  - *Tree search*: step-level MCTS (Phase 30). *Simulations* 0 = off; branching, max steps,
    tokens per step, exploration *c* and step temperature. Steps are scored by the server's `--prm`
    judge when one is loaded (named under the sliders), else by the model's own token probability.
    The reply arrives once the search has chosen a path.
  - *Chat template*: the model's Jinja template as the server uses it (source: `gguf` or `file`).
    Edit it and every request carries your copy as `chat_template`; *Restore model template*
    undoes it. Needed for GGUFs that ship a broken template (official BitNet b1.58 2B-4T — start
    from `docs/templates/bitnet-b1.58.jinja`, or launch with `--chat-template-file`). Read-only
    when the server runs `--no-request-template`.
  Your changes are remembered in this browser; the badge on **Settings** counts how many differ
  from the server defaults, and **Reset to server defaults** discards them.
  - *Server defaults* (Phase 40.3): which server values are not at their built-in default and
    where each comes from — the `pz config` file, a `PZ_*` environment variable or a command-line
    flag. **Save N as server defaults** writes the settings you changed here to the server's config
    file (the same file `pz config set` writes), and **Save title** sets the web UI title and
    wordmark. Saved values apply when the server restarts; a flag or environment variable given at
    startup still wins, and the message says so. The server can turn this off with
    `--no-config-write` (or `pz config set no-config-write true`).
- **New chat** — clears the conversation (the full history is otherwise sent with every turn,
  so the model remembers earlier messages).
- **Watermark** — paste text to check whether it carries this server's watermark (z-score
  against the threshold; uses `POST /v1/watermark/detect`).
- **Reply footer** — reply and prompt token counts, e.g. "7 reply tokens · 37 prompt tokens"
  (with how many prompt tokens came from the prompt cache) and why the reply ended (complete, hit
  max tokens, stopped). The prompt count covers everything the model read before replying: the
  whole conversation rendered through the chat template, which can add a system message and turn
  markers, so a short question still counts dozens of tokens.
- **Theme toggle** (top bar) — cycles dark → light → auto (follows OS preference).
- **Image upload** (paperclip icon, composer) — only rendered when the server reports vision
  is configured; attach an image, then send a message referencing it. Requires starting the
  server with both `--vision <path>` and `--proj <path>`.
- **Model info panel** — shows the loaded model file, layer count, width and context size, plus
  vision/MoE tags (from `GET /v1/config`'s `model` object, same data as `GET /v1/models`).
- **Branding** — the top-bar name comes from `--ui-title` (default `project-zero`). It is shown
  as the block wordmark from the terminal banner (`PROJECT ZERO` in `#` glyphs, accent green on a
  transparent background, no `>_` prefix) in the top bar and, large, on the empty chat screen. The
  server renders it from the title with the same font as the CLI (`GET /v1/config` →
  `ui.logo_ascii`); `--ui-logo text`, or a title with letters the font doesn't have, shows the
  plain name instead. Assets: `docs/design/brand/`.

### Enabling image upload

```bash
./pz --model models/<vision-model>.gguf \
    --vision models/<vision-tower>.gguf --proj models/<projector>.gguf \
    --server --port 8080
```

Without `--vision`/`--proj`, the attach control is hidden entirely (not just disabled) —
the web UI only ever offers controls the running server can actually serve.

## CLI / REPL

### Starting it

```bash
./pz --model models/smollm2.gguf
```

With no `--prompt`, this drops into the interactive REPL. In a terminal you'll see,
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
| `GET /v1/config` | Server generation defaults, valid ranges, feature flags, UI title; `config` = where each default comes from (`sources`: default/file/env/cli), the config file `path` and whether the UI may save (`writable`) |
| `POST /v1/config/defaults` | `{key, value}` or `{key, unset: true}` → save one default (a `/v1/config` defaults name, or `ui_title`) to the `pz config` file for the next start. Needs `Content-Type: application/json` from the server's own origin (or a CORS-allowed one); off with `--no-config-write` |
| `GET /v1/prompt_cache`, `POST /v1/prompt_cache/clear` | Prompt-cache snapshot stats / drop all snapshots |
| `POST /v1/watermark/detect` | `{text, key?, gamma?, z_threshold?}` → watermark z-score |
| `GET /health` | Liveness check (JSON) |
| `GET /metrics` | Prometheus text exposition (needs `--metrics`) |
| `GET /docs`, `GET /openapi.json` | Interactive API docs / OpenAPI 3.0 spec |
| `GET /`, `GET /assets/*` | The web UI (see `--web-ui` above) |

Chat completion requests accept, besides `messages`/`stream`/`max_tokens`/`temperature`/`top_p`:
`top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`,
`penalty_last_n`, `seed`, `stop` (string or array), `response_format: {"type":"json_object"}` (the reply is then one JSON object or array: it starts with `{` or `[` and ends at the matching bracket)
`watermark: {"enabled", "key", "gamma", "delta"}`, `cache_prompt` (default from
`--prompt-cache`, on) and `best_of` (default `--best-of`, capped by `--best-of-max`; the response
then includes a `best_of` object with each candidate's score). Omitted fields use the server defaults
(the CLI flags). Responses carry `usage` token counts and `finish_reason`
(`stop` | `length` | `cancelled`); the final streaming chunk includes `usage` too.

CORS (`--cors`, `--cors-origin <origin>`, repeatable or `*`) and API-key auth (`--api-key
<key>`, checked via `Authorization: Bearer <key>`) are both **off by default** — enable them
explicitly for anything beyond loopback experimentation. By default one chat completion runs at
a time; `--parallel N` lets up to N run together, batched into one forward pass per token
(`--parallel-batch` sets how many tokens that pass may carry, default 128). Settings → Server
defaults shows how many requests the server answers at once (and why fewer, for models that cannot
batch) and saves both values for the next start. Requests that need the whole engine (an image,
`best_of`, tree search) wait for the others and run alone. When every slot is busy a request gets
`429 Too Many Requests`; start the server with `--queue-size N` (and optionally
`--queue-timeout S`) to let up to N requests wait their turn instead (the web UI shows the queue
position). `/metrics` reports `project_zero_generation_slots`, `_slots_busy`, `queue_waiting` and
`parallel_kv_tokens`. A server started with `--remote-layers` (Phase 28) shows a
`dist N nodes` tag in the header; hover it for each node's layer range. `--kv-offload <dir>` (Phase 31)
shows a `KV on disk` tag.

**Update notices (Phase 38).** The server checks this project's GitHub Releases in the background
at most once a day and the UI shows `>_ update available vX → vY · release notes · dismiss` when a
newer release exists (dismissed per version). Settings → Updates shows the status, the server's
interval/channel/repo, a **Check now** button and a banner toggle. Nothing is ever downloaded or
installed — verify the release's `SHA256SUMS`. Operators control it with `--no-update-check`,
`--update-interval <h>`, `--update-channel auto|stable|prerelease`, `--update-repo owner/name`,
`--update-api <url>` or `PZ_NO_UPDATE_CHECK=1`; it is always off when `CI` is set.
`pz --check-update` checks once from the shell (exit 3 = update available). Either way, other routes (static assets, `/metrics`, `/docs`, `/health` are never blocked by an in-flight
generation).

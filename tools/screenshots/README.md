# Design-QA screenshot capture (Phase 22.4)

Opt-in tooling — never run as part of `make test`/`make release`/CI. Produces the reference
screenshots graded against `docs/design/ui-ux-principles.md`.

## Web UI (`webui/chat.spec.mjs`)

1. Build the engine and start a server with the web UI on and a model loaded:
   ```bash
   make release
   ./adaptive_ai_engine --model models/smollm2.gguf --server --port 8080 --threads 2 &
   ```
2. Run the capture script (uses the `playwright` package):
   ```bash
   PZ_PORT=8080 node tools/screenshots/webui/chat.spec.mjs docs/design/screenshots
   ```
3. Stop the server (`kill %1` or similar).

## CLI (`cli/capture.mjs`)

1. One-time setup: install the xterm.js dependency used to render the captured terminal
   session with pixel-accurate ANSI/cursor handling (not a hand-rolled ANSI-to-HTML
   approximation):
   ```bash
   npm --prefix tools/screenshots/cli install
   ```
2. Capture any CLI invocation — the script drives a real pty via `script(1)`, so colors,
   in-place progress updates, and the live tok/s indicator all render exactly as they would in
   a real terminal:
   ```bash
   node tools/screenshots/cli/capture.mjs \
     docs/design/screenshots/cli-repl.png "REPL: --color always" \
     -- ./adaptive_ai_engine --model models/smollm2.gguf --color always --threads 2
   ```

## After capturing

Each screenshot must be reviewed against `docs/design/ui-ux-principles.md` (principle-by-
principle, pass/fail/notes) before being accepted into `README.md` — see the Phase 22.4 entry in
`docs/ai/change-trace.md` for the review process.

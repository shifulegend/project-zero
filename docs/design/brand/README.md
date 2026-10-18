# Brand assets

| File | What it is |
|---|---|
| `project-zero-wordmark.txt` | The reference block wordmark (5 rows, `#` ink, one blank column between letters). |
| `project-zero-wordmark-green.png` | The same wordmark as a transparent (RGBA) PNG, `#00FF00` bold monospace `#`, 2576×367. For README/social use. |

The wordmark is not drawn by hand anywhere in the product: both the CLI startup banner
(`src/cli/banner.c`) and the web UI (`webui/src/lib/AsciiLogo.svelte`, fed by
`GET /v1/config` → `ui.logo_ascii`) render it from one font, `src/brand/block_font.c`.
`tests/test_block_font.c` pins that font's output to `project-zero-wordmark.txt`.

In the web UI the wordmark is plain text on a transparent background tinted with the theme
accent, so it adapts to light and dark mode. It is rendered from `--ui-title` (default
`project-zero`); titles the font cannot spell, or `--ui-logo text`, fall back to the text name.

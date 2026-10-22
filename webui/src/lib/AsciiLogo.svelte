<script>
  // The Project Zero block wordmark — the same glyphs as the terminal banner
  // (brand/block_font.c), rendered by the server from the configurable title
  // (GET /v1/config ui.logo_ascii). Plain text on a transparent background,
  // tinted with the accent, so it follows light/dark themes.
  export let rows = '';
  export let label = '';
  export let size = 'lg'; // 'sm' (top bar) | 'lg' (empty state)
</script>

<pre class="ascii-logo {size}" role="img" aria-label={label}>{rows.replace(/\n$/, '')}</pre>

<style>
  .ascii-logo {
    margin: 0;
    padding: 0;
    background: transparent;
    color: var(--accent);
    font-family: ui-monospace, 'SF Mono', Menlo, Consolas, 'DejaVu Sans Mono', monospace;
    font-weight: 700;
    line-height: 1;
    letter-spacing: 0;
    white-space: pre;
    user-select: none;
  }
  .sm {
    font-size: 5px;
  }
  .lg {
    font-size: clamp(6px, 1.45vw, 13px);
  }
</style>

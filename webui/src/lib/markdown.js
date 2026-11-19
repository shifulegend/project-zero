// Phase 22.2 — minimal, safe markdown-lite renderer for message bubbles.
// Escapes HTML first, then applies a small set of transforms (fenced code
// blocks, inline code, bold). Not a full CommonMark engine, deliberately —
// mirrors the same pragmatic scope as the CLI's cli/md_render.c.

function escapeHtml(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/** @param {string} text @returns {string} sanitized HTML */
export function renderMarkdown(text) {
  let escaped = escapeHtml(text ?? '');

  // Fenced code blocks: ```lang\ncode``` — rendered first so ** or ` inside
  // a code block isn't further transformed.
  escaped = escaped.replace(/```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g, (_m, lang, code) => {
    const langAttr = lang ? ` data-lang="${lang}"` : '';
    return `<pre class="md-code-block"${langAttr}><code>${code}</code></pre>`;
  });

  // Inline code: `code`
  escaped = escaped.replace(/`([^`\n]+)`/g, '<code class="md-inline-code">$1</code>');

  // Bold: **text**
  escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Line breaks, except inside <pre>...</pre> spans, whose newlines are kept as written. One pass
  // over the <pre> spans, so the cost grows linearly with the text (this runs on every streamed
  // token; an earlier version rescanned the whole prefix for every newline).
  const parts = [];
  let last = 0;
  for (const m of escaped.matchAll(/<pre[\s\S]*?<\/pre>/g)) {
    parts.push(escaped.slice(last, m.index).replace(/\n/g, '<br>'), m[0]);
    last = m.index + m[0].length;
  }
  parts.push(escaped.slice(last).replace(/\n/g, '<br>'));
  escaped = parts.join('');

  return escaped;
}

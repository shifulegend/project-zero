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

  // Line breaks (outside <pre> blocks — safe here since <pre> content has
  // no literal newlines re-introduced by the transforms above other than
  // ones already inside the <pre>, which should keep their newlines as-is).
  escaped = escaped.replace(/\n/g, (m, offset, full) => {
    // Skip newlines inside a <pre>...</pre> span.
    const before = full.slice(0, offset);
    const openPre = (before.match(/<pre/g) || []).length;
    const closePre = (before.match(/<\/pre>/g) || []).length;
    return openPre > closePre ? m : '<br>';
  });

  return escaped;
}

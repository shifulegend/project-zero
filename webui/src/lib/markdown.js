// Phase 22.2 — minimal, safe markdown-lite renderer for message bubbles.
// Escapes HTML first, then applies a small set of transforms (fenced code
// blocks, inline code, bold). Not a full CommonMark engine, deliberately —
// mirrors the same pragmatic scope as the CLI's cli/md_render.c.

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Inline transforms for text outside a fenced code block.
function renderInline(text) {
  return (
    text
      // Inline code: `code`
      .replace(/`([^`\n]+)`/g, '<code class="md-inline-code">$1</code>')
      // Bold: **text**
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      // Line breaks
      .replace(/\n/g, '<br>')
  );
}

/** @param {string} text @returns {string} sanitized HTML */
export function renderMarkdown(text) {
  const escaped = escapeHtml(text ?? '');

  // Fenced code blocks: ```lang\ncode```. Their content is emitted as written (escaped, with its
  // newlines): ** and ` inside a code block are not transformed. Everything between blocks gets the
  // inline transforms. One pass, so the cost grows linearly with the text (this runs on every
  // streamed token).
  const parts = [];
  let last = 0;
  for (const m of escaped.matchAll(/```([a-zA-Z0-9_+-]*)\n?([\s\S]*?)```/g)) {
    parts.push(renderInline(escaped.slice(last, m.index)));
    const langAttr = m[1] ? ` data-lang="${m[1]}"` : '';
    parts.push(`<pre class="md-code-block"${langAttr}><code>${m[2]}</code></pre>`);
    last = m.index + m[0].length;
  }
  parts.push(renderInline(escaped.slice(last)));
  return parts.join('');
}

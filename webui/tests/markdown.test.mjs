import test from 'node:test';
import assert from 'node:assert/strict';
import { renderMarkdown } from '../src/lib/markdown.js';
import { renderMarkdown as legacy } from './markdown_legacy.js';

test('plain text keeps its line breaks', () => {
  assert.equal(renderMarkdown('a\nb\n\nc'), 'a<br>b<br><br>c');
});

test('fenced code keeps newlines and gets a language hint', () => {
  assert.equal(
    renderMarkdown('x\n```js\nlet a;\nlet b;\n```\ny'),
    'x<br><pre class="md-code-block" data-lang="js"><code>let a;\nlet b;\n</code></pre><br>y'
  );
});

test('inline code and bold are rendered', () => {
  assert.equal(
    renderMarkdown('**b** `c`'),
    '<strong>b</strong> <code class="md-inline-code">c</code>'
  );
});

test('bold and inline code inside a fenced block are left as written', () => {
  const out = renderMarkdown('```\n**not bold** and `not code`\n```');
  assert.equal(out, '<pre class="md-code-block"><code>**not bold** and `not code`\n</code></pre>');
});

test('bold and inline code next to a fenced block still render', () => {
  const out = renderMarkdown('**a**\n```\nx\n```\n`b`');
  assert.equal(
    out,
    '<strong>a</strong><br><pre class="md-code-block"><code>x\n</code></pre><br><code class="md-inline-code">b</code>'
  );
});

test('HTML in the text is escaped, never rendered', () => {
  const out = renderMarkdown('<script>alert(1)</script><img src=x onerror=alert(1)>');
  assert.ok(!out.includes('<script'));
  assert.ok(!out.includes('<img'));
  assert.ok(out.includes('&lt;script&gt;'));
});

test('a fence language cannot break out of the attribute', () => {
  const out = renderMarkdown('```js" onmouseover="alert(1)\ncode\n```');
  // the opening tag is exactly the safe one; the rest is plain text inside <code>
  assert.ok(out.startsWith('<pre class="md-code-block" data-lang="js"><code>'));
  assert.equal((out.match(/<pre/g) || []).length, 1);
});

test('entities and quotes inside code blocks stay text', () => {
  const out = renderMarkdown('```\n"a" & \'b\' <i>\n```');
  assert.ok(out.includes('&amp;') && out.includes('&lt;i&gt;'));
  assert.ok(!out.includes('<i>'));
});

test('null and undefined render as empty', () => {
  assert.equal(renderMarkdown(null), '');
  assert.equal(renderMarkdown(undefined), '');
});

test('output is identical to the previous implementation on 20000 random inputs without ** or ` (those differ by design inside code blocks)', () => {
  const frag = [
    'a',
    'b c',
    '\n',
    '\n\n',
    '```',
    '```js\n',
    '<',
    '>',
    '&',
    '<pre>',
    '</pre>',
    '"',
    "'",
    'x\ny',
  ];
  let seed = 12345;
  const rnd = (n) => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed % n;
  };
  for (let i = 0; i < 20000; i++) {
    let s = '';
    for (let k = 0, n = 1 + rnd(12); k < n; k++) s += frag[rnd(frag.length)];
    assert.equal(renderMarkdown(s), legacy(s), JSON.stringify(s));
  }
});

test('a long streamed reply renders in linear time', () => {
  const text = 'line of text with **bold** and `code`\n'.repeat(20000);
  const t0 = performance.now();
  renderMarkdown(text);
  const ms = performance.now() - t0;
  assert.ok(ms < 1500, `took ${ms} ms`);
  const t1 = performance.now();
  legacy(text.slice(0, text.length / 4));
  const legacyQuarter = performance.now() - t1;
  assert.ok(
    ms < legacyQuarter * 4 || ms < 300,
    'not slower than the old version on a quarter of the input'
  );
});

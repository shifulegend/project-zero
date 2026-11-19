import test from 'node:test';
import assert from 'node:assert/strict';
import { safeHttpsUrl } from '../src/lib/safeUrl.js';

test('a normal https URL is returned', () => {
  assert.equal(safeHttpsUrl('https://github.com/shifulegend/project-zero/releases/tag/v1'), 'https://github.com/shifulegend/project-zero/releases/tag/v1');
});

test('javascript:, data:, http:, file: and relative values are rejected', () => {
  for (const bad of ['javascript:alert(1)', 'JaVaScRiPt:alert(1)', 'data:text/html,<script>1</script>', 'http://example.com/', 'file:///etc/passwd', '//example.com/x', '/relative', 'ftp://x/y']) {
    assert.equal(safeHttpsUrl(bad), null, bad);
  }
});

test('non-strings, empty strings and malformed URLs give null', () => {
  for (const bad of [null, undefined, 42, {}, '', '   ', 'https://', 'not a url']) assert.equal(safeHttpsUrl(bad), null, String(bad));
});

test('a leading-whitespace javascript URL cannot sneak through', () => {
  assert.equal(safeHttpsUrl(' javascript:alert(1)'), null);
  assert.equal(safeHttpsUrl('\tjavascript:alert(1)'), null);
});

// Phase 22.4 — captures reference screenshots of the web chat UI for design
// QA (graded against docs/design/ui-ux-principles.md).
//
// Requires: a running project-zero server (--server --web-ui on) and the
// `playwright` package (pre-installed globally in this environment at
// /opt/node22/lib/node_modules/playwright; add it as a devDependency of
// webui/ if running elsewhere).
//
// Usage: PZ_PORT=8080 node tools/screenshots/webui/chat.spec.mjs [outDir]
//
// This is a standalone script, not part of `make test`/`make release`/CI —
// screenshot capture is opt-in tooling (see `make screenshots`), matching
// the project's policy of keeping optional tooling out of the default
// build/CI path.

import { createRequire } from 'module';
import { mkdirSync } from 'fs';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  // Fall back to this environment's global install (not on the default
  // module resolution path for a script under tools/screenshots/).
  ({ chromium } = require('/opt/node22/lib/node_modules/playwright'));
}

const port = process.env.PZ_PORT || '8080';
const outDir = process.argv[2] || 'docs/design/screenshots';
mkdirSync(outDir, { recursive: true });

function stamp() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1000, height: 700 } });

  page.on('pageerror', (err) => console.error('[pageerror]', err.message));

  const ts = stamp();
  const shot = (name) => page.screenshot({ path: `${outDir}/${name}-${ts}.png` });

  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
  await shot('01-empty-state-light');

  await page.fill('textarea', 'What is the capital of France?');
  await page.click('button.send-btn');
  await page.waitForSelector('button.stop-btn', { timeout: 5000 });
  await shot('02-generating-light');

  await page.waitForSelector('button.send-btn', { timeout: 30000 });
  await shot('03-reply-light');

  await page.click('button.icon-btn'); // open sampling params
  await page.waitForSelector('input[type=range]');
  await shot('04-params-light');

  await page.click('button.theme-toggle'); // -> dark
  await page.waitForTimeout(150);
  await shot('05-dark');

  await browser.close();
  console.log(`Screenshots written to ${outDir}/ (timestamp ${ts})`);
}

main().catch((err) => { console.error(err); process.exit(1); });

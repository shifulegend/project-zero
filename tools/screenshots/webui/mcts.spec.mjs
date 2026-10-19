// Settings → Tree search (Phase 30, 2026-10-19): controls rendered from
// /v1/config (defaults.mcts, limits.mcts_*, features.mcts / features.prm).
//
// Usage: PZ_PORT=8099 node tools/screenshots/webui/mcts.spec.mjs [outDir]

import { createRequire } from 'module';
import { mkdirSync } from 'fs';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  ({ chromium } = require('/opt/node22/lib/node_modules/playwright'));
}

const port = process.env.PZ_PORT || '8080';
const outDir = process.argv[2] || 'docs/design/screenshots';
mkdirSync(outDir, { recursive: true });

async function main() {
  const browser = await chromium.launch();
  for (const scheme of ['light', 'dark']) {
    const page = await browser.newPage({ viewport: { width: 1100, height: 900 }, colorScheme: scheme });
    page.on('pageerror', (err) => console.error('[pageerror]', err.message));
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
    await page.click('button[title="Generation settings"]');
    const section = page.locator('section', { has: page.locator('h3', { hasText: 'Tree search' }) });
    await section.locator('input[type="range"]').first().evaluate((el) => {
      el.value = '16';
      el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await section.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${outDir}/mcts-settings-${scheme}.png` });
    await page.close();
  }
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });

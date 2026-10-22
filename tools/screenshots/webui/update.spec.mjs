// Phase 38 — captures the update banner and Settings → Updates section for
// design QA. Needs a server whose build is older than the latest release
// (e.g. `make release PZ_VERSION=v0.0.9`) so the banner has something to show.
//
// Usage: PZ_PORT=8099 node tools/screenshots/webui/update.spec.mjs [outDir]

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
    const page = await browser.newPage({ viewport: { width: 1000, height: 700 }, colorScheme: scheme });
    page.on('pageerror', (err) => console.error('[pageerror]', err.message));
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
    await page.waitForSelector('.update', { timeout: 15000 });
    await page.screenshot({ path: `${outDir}/update-banner-${scheme}.png` });
    await page.click('button[title="Generation settings"]');
    await page.waitForSelector('.upd .state');
    const section = page.locator('section', { has: page.locator('h3', { hasText: 'Updates' }) });
    await section.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${outDir}/update-settings-${scheme}.png` });
    await page.close();
  }
  // Dismiss persists per version.
  const page = await browser.newPage({ viewport: { width: 1000, height: 400 } });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
  await page.waitForSelector('.update');
  await page.click('.update .dismiss');
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);
  console.log('banner after dismiss+reload:', await page.locator('.update').count());
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });

// Phase 40.3 — captures Settings → Server defaults (value sources, save to
// the `pz config` file) for design QA, and exercises one save end to end.
// Start the server with a throwaway config so the save is harmless, e.g.
//   PZ_CONFIG=/tmp/pz-shot.toml PZ_TOP_K=20 ./pz --model <m> --server --port 8099 --top-p 0.8
//
// Usage: PZ_PORT=8099 node tools/screenshots/webui/server_defaults.spec.mjs [outDir]

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
    const page = await browser.newPage({ viewport: { width: 1000, height: 760 }, colorScheme: scheme });
    page.on('pageerror', (err) => console.error('[pageerror]', err.message));
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle' });
    await page.click('button[title="Generation settings"]');
    const section = page.locator('section', { has: page.locator('h3', { hasText: 'Server defaults' }) });
    await section.waitFor({ timeout: 15000 });
    if (scheme === 'light') {
      // Change one knob, then save it as a server default.
      const temp = page.locator('label.field', { hasText: 'Temperature' }).locator('input[type=range]');
      await temp.fill('0.9');
      await section.locator('button', { hasText: 'as server defaults' }).click();
      await section.locator('.ok, .err').first().waitFor({ timeout: 10000 });
      console.log('save result:', await section.locator('.ok, .err').first().innerText());
    }
    await section.scrollIntoViewIfNeeded();
    await section.screenshot({ path: `${outDir}/server-defaults-${scheme}.png` });
    await page.close();
  }
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });

// Settings → Chat template (2026-10-19): the model's template as published by
// /v1/config, and the "edited · sent per request" state after editing it.
//
// Usage: PZ_PORT=8099 node tools/screenshots/webui/chat_template.spec.mjs [outDir]

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
    const area = page.locator('textarea.mono');
    await area.scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${outDir}/chat-template-model-${scheme}.png` });
    await area.fill((await area.inputValue()).replace('Human: ', 'User: '));
    await page.screenshot({ path: `${outDir}/chat-template-edited-${scheme}.png` });
    await page.close();
  }
  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });

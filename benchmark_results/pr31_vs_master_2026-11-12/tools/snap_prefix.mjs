#!/usr/bin/env node
// Renders the FIRST part of a recorded pty stream (from capture.mjs's PZ_CAPTURE_RAW_OUT) with the same
// capture.html/xterm.js pipeline as tools/screenshots/cli/capture.mjs. capture.mjs only screenshots after the
// command exits, so this is how a mid-run (PROCESSING) frame is produced from the real recording:
// the stream is cut right after the line that starts the Nth config (marker text), i.e. while that run is in progress.
// Usage: snap_prefix.mjs <raw> <out.png> <marker-regex> [caption]
import { readFileSync } from 'fs';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';
const require = createRequire(import.meta.url);
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const [raw, out, marker, caption] = process.argv.slice(2);
const buf = readFileSync(raw);
const text = buf.toString('latin1');
const m = new RegExp(marker).exec(text);
if (!m) { console.error('marker not found'); process.exit(2); }
const end = text.indexOf('\n', m.index) + 1;
const prefix = buf.subarray(0, end);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1200 } });
const u = pathToFileURL(process.env.CAPTURE_HTML); u.searchParams.set('rows', process.env.PZ_CAPTURE_ROWS || '60');
await page.goto(u.href);
await page.waitForFunction(() => window.pzTerm !== undefined);
await page.evaluate((b64) => { window.pzTerm.write(Uint8Array.from(atob(b64), c => c.charCodeAt(0))); }, prefix.toString('base64'));
await page.waitForTimeout(300);
await page.evaluate(() => { const t = window.pzTerm, b = t.buffer.active; let last = 0;
  for (let y = 0; y < t.rows; y++) { const l = b.getLine(y); if (l && l.translateToString(true).length > 0) last = y; }
  const r = Math.min(t.rows, last + 2); if (r < t.rows) t.resize(t.cols, r); });
if (caption) await page.evaluate((c) => { const e = document.createElement('div'); e.textContent = c;
  e.style.cssText = 'color:#8b93a3;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;padding:0 16px 12px;'; document.body.appendChild(e); }, caption);
await page.locator('body').screenshot({ path: out });
await browser.close(); console.log('Wrote ' + out);

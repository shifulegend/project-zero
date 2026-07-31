#!/usr/bin/env node
// Phase 22.4 — captures a real terminal session (via `script`, preserving
// ANSI/cursor control bytes) and renders it through xterm.js in a headless
// Playwright page for a pixel-accurate screenshot — avoids hand-rolling an
// ANSI-to-HTML converter that would have to reimplement cursor movement
// (\r, \x1b[K) to match what a real terminal does with the CLI's in-place
// progress/live-stats updates.
//
// Usage:
//   node tools/screenshots/cli/capture.mjs <output.png> [caption] -- <command...>
//
// For interactive input (e.g. the REPL), set PZ_CAPTURE_STDIN to the lines
// to feed in (newline-separated) — without it, stdin is closed immediately
// (fine for one-shot --prompt runs, but an interactive REPL would just see
// EOF and exit before generating anything).
//
// The captured command has a 60s default wall-clock budget; set
// PZ_CAPTURE_TIMEOUT_MS to raise it for slow real-model runs (e.g. a large
// model at a few hundred ms/token easily needs several minutes).
//
// The terminal defaults to 70 rows; output longer than that scrolls earlier
// content (including the startup banner) out of view before the screenshot
// is taken. Set PZ_CAPTURE_ROWS to a larger value for runs with a lot of
// startup output (calibration + hardware profile + long model config), so
// nothing scrolls off before the final screenshot.
//
// Example (one-shot):
//   node tools/screenshots/cli/capture.mjs docs/design/screenshots/cli-repl.png \
//     -- ./adaptive_ai_engine --model models/smollm2.gguf --color always --threads 2
//
// Example (REPL):
//   PZ_CAPTURE_STDIN=$'Explain **bold** briefly\n/quit' \
//     node tools/screenshots/cli/capture.mjs out.png -- ./adaptive_ai_engine --model models/smollm2.gguf
//
// Requires: `script` (util-linux, present on this Linux environment),
// the `playwright` package (global at /opt/node22/lib/node_modules/playwright
// in this environment), and `npm install` run once in this directory for
// @xterm/xterm (see package.json) — opt-in dev tooling, not part of
// make test/release/CI.

import { execFileSync } from 'child_process';
import { readFileSync, mkdtempSync, mkdirSync } from 'fs';
import { tmpdir } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  // Fall back to this environment's global install.
  ({ chromium } = require('/opt/node22/lib/node_modules/playwright'));
}

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const sepIndex = argv.indexOf('--');
  if (sepIndex === -1) {
    console.error('Usage: capture.mjs <output.png> [caption] -- <command...>');
    process.exit(1);
  }
  const before = argv.slice(0, sepIndex);
  const command = argv.slice(sepIndex + 1);
  const outputPath = before[0];
  const caption = before[1] || '';
  return { outputPath, caption, command };
}

async function main() {
  const { outputPath, caption, command } = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(outputPath), { recursive: true });

  const workDir = mkdtempSync(join(tmpdir(), 'pz-cli-capture-'));
  const rawLogPath = join(workDir, 'raw.log');

  const commandStr = command.map((s) => `'${s.replace(/'/g, `'\\''`)}'`).join(' ');
  const stdinText = process.env.PZ_CAPTURE_STDIN;
  execFileSync('script', ['-qec', commandStr, rawLogPath], {
    input: stdinText !== undefined ? stdinText + '\n' : undefined,
    stdio: stdinText !== undefined ? ['pipe', 'ignore', 'ignore'] : 'ignore',
    timeout: process.env.PZ_CAPTURE_TIMEOUT_MS ? Number(process.env.PZ_CAPTURE_TIMEOUT_MS) : 60000,
  });

  const raw = readFileSync(rawLogPath);

  // Optional: persist the raw pty byte stream next to the screenshot so
  // benchmark sweeps can extract numbers ([gen] tok/s, hardware profile)
  // from the exact bytes the screenshot renders, instead of re-running the
  // command outside a TTY (which would suppress banner/color and change
  // what the screenshot shows). Set PZ_CAPTURE_RAW_OUT to a file path.
  // (2026-07-17, added for the 3-axis comparison sweep.)
  if (process.env.PZ_CAPTURE_RAW_OUT) {
    const { copyFileSync } = await import('fs');
    copyFileSync(rawLogPath, process.env.PZ_CAPTURE_RAW_OUT);
  }

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 900, height: 1200 } });
  const rowsOverride = process.env.PZ_CAPTURE_ROWS;
  const captureUrl = pathToFileURL(join(__dirname, 'capture.html'));
  if (rowsOverride) captureUrl.searchParams.set('rows', rowsOverride);
  await page.goto(captureUrl.href);
  await page.waitForFunction(() => window.pzTerm !== undefined);

  // Feed raw bytes as a base64 string across the Node/browser boundary to
  // avoid encoding issues with binary/control bytes.
  await page.evaluate((b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    window.pzTerm.write(bytes);
  }, raw.toString('base64'));

  await page.waitForTimeout(300); // let xterm.js finish its render pass

  // Trim the terminal down to however many rows the captured session
  // actually used. `rows` (default 70, override via PZ_CAPTURE_ROWS) is
  // sized generously so long startup output doesn't scroll the banner out
  // of view, but most runs use far fewer rows than that — without this,
  // fullPage:true screenshots a mostly-empty terminal padded with a wall of
  // unused blank rows below the real content.
  await page.evaluate(() => {
    const term = window.pzTerm;
    const buf = term.buffer.active;
    let lastUsedRow = 0;
    for (let y = 0; y < term.rows; y++) {
      const line = buf.getLine(y);
      if (line && line.translateToString(true).length > 0) lastUsedRow = y;
    }
    const trimmedRows = Math.min(term.rows, lastUsedRow + 2);
    if (trimmedRows < term.rows) term.resize(term.cols, trimmedRows);
  });

  if (caption) {
    await page.evaluate((text) => {
      const capEl = document.createElement('div');
      capEl.textContent = text;
      capEl.style.cssText =
        'color:#8b93a3;font-family:ui-monospace,Menlo,Consolas,monospace;' +
        'font-size:12px;padding:0 16px 12px;';
      document.body.appendChild(capEl);
    }, caption);
  }

  // Screenshot the body element directly rather than page.screenshot({
  // fullPage: true }): when trimmed content is shorter than the 1200px
  // viewport, document.documentElement.scrollHeight (what fullPage relies
  // on) is clamped to the viewport height by the browser, not the actual
  // content height — the body element's bounding box isn't.
  await page.locator('body').screenshot({ path: outputPath });
  await browser.close();
  console.log(`Wrote ${outputPath}`);
}

main().catch((err) => { console.error(err); process.exit(1); });

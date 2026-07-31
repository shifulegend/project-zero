#!/usr/bin/env node
// Phase 22.5 — captures a real terminal session's *timed* output (via
// `script --log-timing`, not just the final bytes like capture.mjs) and
// replays it frame-by-frame through the same xterm.js/Playwright harness,
// screenshotting after each timed write, then encodes the frames into an
// animated GIF. Built specifically for the CLI startup banner's reveal +
// shimmer animation, which is otherwise invisible in a single static
// screenshot.
//
// Usage:
//   node tools/screenshots/cli/capture-gif.mjs <output.gif> [marker] -- <command...>
//
// `marker` (default "Project Zero Engine") is the literal text that marks
// the end of the animation to capture — the script replays real-timed
// writes up through the chunk containing that marker, then stops (any
// bytes after the marker within that same chunk are trimmed so the GIF
// doesn't spill into whatever prints next). No stdin is fed to the child
// (matches capture.mjs's non-interactive default) — fine for capturing
// startup animation, which happens before the REPL ever reads a line.
//
// Requires everything capture.mjs requires, plus the `gifenc` and `pngjs`
// dev dependencies (see package.json) — opt-in dev tooling, not part of
// make test/release/CI.

import { execFileSync } from 'child_process';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join, dirname } from 'path';
import { fileURLToPath, pathToFileURL } from 'url';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require('playwright'));
} catch {
  ({ chromium } = require('/opt/node22/lib/node_modules/playwright'));
}
const { GIFEncoder, quantize, applyPalette } = require('gifenc');
const { PNG } = require('pngjs');

const __dirname = dirname(fileURLToPath(import.meta.url));

function parseArgs(argv) {
  const sepIndex = argv.indexOf('--');
  if (sepIndex === -1) {
    console.error('Usage: capture-gif.mjs <output.gif> [marker] -- <command...>');
    process.exit(1);
  }
  const before = argv.slice(0, sepIndex);
  const command = argv.slice(sepIndex + 1);
  const outputPath = before[0];
  const marker = before[1] || 'Project Zero Engine';
  return { outputPath, marker, command };
}

// Parses `script --log-timing` classic format: one "<delay_seconds>
// <byte_count>" line per write.
function parseTiming(text) {
  return text
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => {
      const [delay, bytes] = line.trim().split(/\s+/);
      return { delaySec: parseFloat(delay), nBytes: parseInt(bytes, 10) };
    });
}

async function main() {
  const { outputPath, marker, command } = parseArgs(process.argv.slice(2));
  mkdirSync(dirname(outputPath), { recursive: true });

  const workDir = mkdtempSync(join(tmpdir(), 'pz-cli-gif-'));
  const rawLogPath = join(workDir, 'raw.log');
  const timingPath = join(workDir, 'timing.log');

  const commandStr = command.map((s) => `'${s.replace(/'/g, `'\\''`)}'`).join(' ');
  execFileSync(
    'script',
    ['-q', '-e', '-c', commandStr, '-T', timingPath, '-O', rawLogPath],
    { stdio: 'ignore', timeout: 60000 },
  );

  const raw = readFileSync(rawLogPath);
  const chunks = parseTiming(readFileSync(timingPath, 'utf8'));

  const markerOffset = raw.indexOf(marker);
  if (markerOffset === -1) {
    console.error(`Marker "${marker}" not found in captured output — nothing to trim to.`);
    process.exit(1);
  }

  const browser = await chromium.launch();
  const ROWS = 70, COLS = 100;

  // script(1)'s own "Script started on ... [COMMAND=...]" header line wraps
  // across a variable number of terminal rows depending on the command's
  // string length (long absolute paths push it past 100 cols easily) — so
  // the banner's actual starting row can't be hardcoded. Measure it first:
  // do a single non-timed full write of the whole animation slice into a
  // throwaway page, then find the first row containing a '#' glyph column
  // — that's the banner's own block-font content. (Can't key off "first
  // blank row" instead: by the time the full slice has been replayed, the
  // reveal/shimmer frames have long since overwritten banner.c's originally
  // -blank reserved rows with real content, so the only truly blank row
  // left is *after* the banner, not before it.)
  const measurePage = await browser.newPage({ viewport: { width: 900, height: 1200 } });
  await measurePage.goto(pathToFileURL(join(__dirname, 'capture.html')).href);
  await measurePage.waitForFunction(() => window.pzTerm !== undefined);
  await measurePage.evaluate((b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    window.pzTerm.write(bytes);
  }, raw.subarray(0, markerOffset).toString('base64'));
  await measurePage.waitForTimeout(200);
  const bannerStartRow = await measurePage.evaluate(() => {
    const buf = window.pzTerm.buffer.active;
    for (let i = 0; i < buf.length; i++) {
      const line = buf.getLine(i);
      if (line && line.translateToString(true).includes('#')) return i;
    }
    return 0;
  });
  await measurePage.close();

  const page = await browser.newPage({ viewport: { width: 900, height: 1200 } });
  await page.goto(pathToFileURL(join(__dirname, 'capture.html')).href);
  await page.waitForFunction(() => window.pzTerm !== undefined);

  // xterm.js fills `.xterm-screen` with exactly rows/cols cells (unlike
  // `#term`, which also includes capture.html's own 16px CSS padding on
  // all sides — dividing the padded box by row count overestimates each
  // row's height and throws the crop off by a visible row). Dividing the
  // unpadded screen box by row/col count gives the true per-cell pixel
  // size without hardcoding font metrics.
  const screenBox = await page.locator('.xterm-screen').boundingBox();
  const rowH = screenBox.height / ROWS;
  const colW = screenBox.width / COLS;
  const clip = {
    x: screenBox.x,
    y: screenBox.y + rowH * bannerStartRow,
    width: colW * 80,          // banner text is ~71 cols; 80 leaves a little margin
    height: rowH * 6,          // 5 reserved banner rows + 1 trailing blank line
  };

  const frames = []; // { png: Buffer, delayMs: number }
  let offset = 0;
  for (const { delaySec, nBytes } of chunks) {
    const chunkEnd = offset + nBytes;
    if (offset >= markerOffset) break; // already past the animation we want

    const writeEnd = Math.min(chunkEnd, markerOffset);
    const slice = raw.subarray(offset, writeEnd);
    await page.evaluate((b64) => {
      const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
      window.pzTerm.write(bytes);
    }, slice.toString('base64'));

    const delayMs = Math.max(20, Math.round(delaySec * 1000));
    frames.push({ png: await page.screenshot({ clip }), delayMs });

    offset = chunkEnd;
    if (writeEnd >= markerOffset) break;
  }

  // Hold on the final settled frame for a beat before the GIF loops.
  for (let i = 0; i < 2; i++) {
    frames.push({ png: await page.screenshot({ clip }), delayMs: 700 });
  }

  await browser.close();

  const gif = GIFEncoder();
  for (const { png, delayMs } of frames) {
    const decoded = PNG.sync.read(png);
    const palette = quantize(decoded.data, 256);
    const index = applyPalette(decoded.data, palette);
    gif.writeFrame(index, decoded.width, decoded.height, {
      palette,
      delay: delayMs,
      transparent: false,
    });
  }
  gif.finish();
  writeFileSync(outputPath, gif.bytes());
  console.log(`Wrote ${outputPath} (${frames.length} frames)`);
}

main().catch((err) => { console.error(err); process.exit(1); });

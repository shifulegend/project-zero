// Phase 38: shared update-check state for the banner and the Settings section.
// The server owns the check (interval, channel, opt-out); the browser only
// remembers whether to show the banner and which version was dismissed —
// per-browser conveniences, so localStorage is fine (wrapped: it can throw).
import { writable } from 'svelte/store';
import { fetchUpdate, checkUpdateNow } from './api.js';

const BANNER_KEY = 'pz.update.banner';
const DISMISSED_KEY = 'pz.update.dismissed';

function read(key, fallback) {
  try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch { return fallback; }
}
function write(key, value) {
  try { if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value); } catch { /* ignore */ }
}

export const updateInfo = writable(null);
export const updateError = writable(null);
export const showBanner = writable(read(BANNER_KEY, '1') !== '0');
export const dismissedVersion = writable(read(DISMISSED_KEY, ''));

showBanner.subscribe((v) => write(BANNER_KEY, v ? null : '0'));
dismissedVersion.subscribe((v) => write(DISMISSED_KEY, v || null));

async function pollWhileChecking(info) {
  // The server answers immediately; a started check lands a few seconds
  // later and moves checked_at.
  if (!info?.checking) return info;
  const before = info.checked_at;
  for (let i = 0; i < 8; i++) {
    await new Promise((r) => setTimeout(r, 1500));
    info = await fetchUpdate();
    updateInfo.set(info);
    if (info.checked_at !== before) break;
  }
  return info;
}

export async function refreshUpdate() {
  try {
    const info = await fetchUpdate();
    updateInfo.set(info);
    updateError.set(null);
    await pollWhileChecking(info);
  } catch (e) {
    updateError.set(e.message ?? String(e));
  }
}

export async function checkNow() {
  try {
    const info = await checkUpdateNow();
    updateInfo.set(info);
    updateError.set(null);
    await pollWhileChecking(info);
  } catch (e) {
    updateError.set(e.message ?? String(e));
  }
}

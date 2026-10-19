// Generation settings for the web UI.
//
// Nothing here is a hardcoded default or limit: every value starts from the
// server's GET /v1/config response (which reflects the engine's CLI flags),
// and the viewer's own overrides are layered on top and remembered in
// localStorage (a per-browser convenience — the server stays authoritative).

const STORAGE_KEY = 'pz.settings.v1';

/** Settings object derived purely from /v1/config. */
export function defaultsFromConfig(cfg) {
  const d = cfg?.defaults ?? {};
  return {
    temperature: d.temperature,
    top_p: d.top_p,
    top_k: d.top_k,
    min_p: d.min_p,
    max_tokens: d.max_tokens,
    repetition_penalty: d.repetition_penalty,
    presence_penalty: d.presence_penalty,
    frequency_penalty: d.frequency_penalty,
    penalty_last_n: d.penalty_last_n,
    seed: d.seed ?? -1,
    json_mode: !!d.json_mode,
    ignore_eos: !!d.ignore_eos,
    cache_prompt: d.cache_prompt !== false,
    best_of: d.best_of ?? 1,
    mcts_simulations: d.mcts?.simulations ?? 0,
    mcts_expand: d.mcts?.expand,
    mcts_max_depth: d.mcts?.max_depth,
    mcts_step_tokens: d.mcts?.step_tokens,
    mcts_c: d.mcts?.c,
    mcts_temperature: d.mcts?.temperature,
    system_prompt: d.system_prompt ?? '',
    // The model's active Jinja chat template; editing it makes every request
    // carry the edited copy as "chat_template" (when the server allows it).
    chat_template: cfg?.chat_template?.text ?? '',
    stop: (d.stop ?? []).join('\n'),
    watermark_enabled: !!d.watermark?.enabled,
    watermark_key: '',
    watermark_gamma: d.watermark?.gamma,
    watermark_delta: d.watermark?.delta,
  };
}

function readStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

/** Server defaults overlaid with this viewer's saved overrides (known keys only). */
export function loadSettings(cfg) {
  const base = defaultsFromConfig(cfg);
  const stored = readStored();
  for (const k of Object.keys(base)) {
    if (k in stored && stored[k] !== null && stored[k] !== undefined) base[k] = stored[k];
  }
  return base;
}

/** Persists only the values that differ from the server defaults. */
export function saveSettings(settings, cfg) {
  const base = defaultsFromConfig(cfg);
  const diff = {};
  for (const k of Object.keys(base)) if (settings[k] !== base[k]) diff[k] = settings[k];
  try {
    if (Object.keys(diff).length) localStorage.setItem(STORAGE_KEY, JSON.stringify(diff));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage unavailable (private mode) — settings just won't persist */
  }
}

export function clearSavedSettings() {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
}

/** Number of settings that differ from the server defaults (badge on the button). */
export function countOverrides(settings, cfg) {
  const base = defaultsFromConfig(cfg);
  return Object.keys(base).filter((k) => settings[k] !== base[k]).length;
}

/** Request-body fields for POST /v1/chat/completions. */
export function requestParams(settings, cfg) {
  const maxStop = cfg?.limits?.max_stop ?? 0;
  const stop = String(settings.stop ?? '')
    .split('\n')
    .map((s) => s.replace(/\\n/g, '\n'))
    .filter((s) => s.length > 0)
    .slice(0, maxStop);
  const body = {
    temperature: settings.temperature,
    top_p: settings.top_p,
    top_k: settings.top_k,
    min_p: settings.min_p,
    max_tokens: settings.max_tokens,
    repetition_penalty: settings.repetition_penalty,
    presence_penalty: settings.presence_penalty,
    frequency_penalty: settings.frequency_penalty,
    penalty_last_n: settings.penalty_last_n,
    stop,
  };
  if (Number.isInteger(settings.seed) && settings.seed >= 0) body.seed = settings.seed;
  body.response_format = { type: settings.json_mode ? 'json_object' : 'text' };
  if (cfg?.features?.prompt_cache) body.cache_prompt = !!settings.cache_prompt;
  body.ignore_eos = !!settings.ignore_eos;
  if (cfg?.features?.best_of) body.best_of = settings.best_of;
  if (cfg?.features?.mcts) {
    body.mcts = {
      simulations: settings.mcts_simulations,
      expand: settings.mcts_expand,
      max_depth: settings.mcts_max_depth,
      step_tokens: settings.mcts_step_tokens,
      c: settings.mcts_c,
      temperature: settings.mcts_temperature,
    };
  }
  const wm = {
    enabled: !!settings.watermark_enabled,
    gamma: settings.watermark_gamma,
    delta: settings.watermark_delta,
  };
  if (settings.watermark_key) wm.key = settings.watermark_key;
  body.watermark = wm;
  const tmpl = settings.chat_template ?? '';
  if (cfg?.features?.request_chat_template && tmpl.trim() &&
      tmpl !== (cfg?.chat_template?.text ?? '')) body.chat_template = tmpl;
  return body;
}

/**
 * System message to prepend, or null. An empty prompt that overrides a
 * non-empty server default is sent as an empty system turn so the server's
 * default is suppressed rather than silently re-applied.
 */
export function systemMessage(settings, cfg) {
  const sp = settings.system_prompt ?? '';
  if (!cfg?.features?.system_prompt) return null;
  if (sp.trim()) return { role: 'system', content: sp };
  if ((cfg?.defaults?.system_prompt ?? '') !== '') return { role: 'system', content: '' };
  return null;
}

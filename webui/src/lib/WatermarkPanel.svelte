<script>
  // Phase 33: paste any text and check it for this server's watermark.
  import { detectWatermark } from './api.js';

  export let config;
  export let defaultKey = '';

  let text = '';
  let key = defaultKey;
  let result = null;
  let error = null;
  let busy = false;

  async function run() {
    if (!text.trim() || busy) return;
    busy = true;
    error = null;
    result = null;
    try {
      result = await detectWatermark({ text, key: key || undefined });
    } catch (e) {
      error = e.message ?? String(e);
    }
    busy = false;
  }
</script>

<div class="panel">
  <div class="row">
    <label class="grow">
      <span>Text to check</span>
      <textarea rows="4" bind:value={text} placeholder="Paste text generated with watermarking on"></textarea>
    </label>
  </div>
  <div class="row bottom">
    <label>
      <span>Key {config?.defaults?.watermark?.server_key ? '(blank = server key)' : ''}</span>
      <input type="password" bind:value={key} autocomplete="off" />
    </label>
    <button class="primary" on:click={run} disabled={busy || !text.trim()}>{busy ? 'Checking…' : 'Detect'}</button>
    {#if result}
      <div class="result" class:hit={result.watermarked}>
        <strong>{result.watermarked ? 'Watermark detected' : 'No watermark detected'}</strong>
        <span class="mono">z = {result.z_score.toFixed(2)} (threshold {result.z_threshold}) · {result.green_tokens}/{result.tokens_scored} green</span>
      </div>
    {/if}
    {#if error}<div class="error" role="alert">Error: {error}</div>{/if}
  </div>
</div>

<style>
  .panel {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  .row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
  }
  .row.bottom {
    align-items: flex-end;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-dim);
  }
  .grow {
    flex: 1;
  }
  textarea, input {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-family: inherit;
    font-size: 0.85rem;
    box-sizing: border-box;
    width: 100%;
  }
  textarea:focus, input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .primary {
    background: var(--accent);
    color: var(--accent-contrast);
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    cursor: pointer;
    min-height: 2.1rem;
  }
  .primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .result {
    display: flex;
    flex-direction: column;
    font-size: 0.85rem;
    padding: 0.35rem 0.75rem;
    border-left: 3px solid var(--border);
  }
  .result.hit {
    border-left-color: var(--accent);
  }
  .mono {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
  }
  .error {
    color: #e5484d;
    font-size: 0.85rem;
  }
</style>

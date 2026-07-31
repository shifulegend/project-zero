<script>
  export let temperature;
  export let top_p;
  export let max_tokens;
</script>

<div class="panel">
  <label>
    <span>Temperature <span class="val">{temperature.toFixed(2)}</span></span>
    <input type="range" min="0" max="2" step="0.05" bind:value={temperature} />
  </label>
  <label>
    <span>Top-p <span class="val">{top_p.toFixed(2)}</span></span>
    <input type="range" min="0" max="1" step="0.05" bind:value={top_p} />
  </label>
  <label>
    <span>Max tokens <span class="val">{max_tokens}</span></span>
    <input type="range" min="16" max="2048" step="16" bind:value={max_tokens} />
  </label>
</div>

<style>
  .panel {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    font-size: 0.8rem;
    color: var(--text-dim);
    min-width: 12rem;
  }
  .val {
    color: var(--text);
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  input[type='range'] {
    accent-color: var(--accent);
  }
</style>

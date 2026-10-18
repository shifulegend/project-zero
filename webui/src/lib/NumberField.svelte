<script>
  // Integer input bounded by a server-provided range — used where a slider
  // would be unusable (top-k spans the whole vocabulary, seeds are arbitrary).
  export let label;
  export let value;
  export let range = { min: 0, max: 100, step: 1 };
  export let hint = '';
  export let placeholder = '';

  function onChange(e) {
    const raw = e.target.value.trim();
    let v = raw === '' ? range.min : Math.round(Number(raw));
    if (!Number.isFinite(v)) v = range.min;
    v = Math.min(range.max, Math.max(range.min, v));
    value = v;
    e.target.value = String(v);
  }
</script>

<label class="field" title={hint}>
  <span>{label}</span>
  <input type="number" min={range.min} max={range.max} step={range.step} {value} {placeholder} on:change={onChange} />
  {#if hint}<span class="hint">{hint}</span>{/if}
</label>

<style>
  .field {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-dim);
    min-width: 0;
  }
  input {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.35rem 0.5rem;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.85rem;
    width: 100%;
    box-sizing: border-box;
  }
  input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .hint {
    font-size: 0.7rem;
  }
</style>

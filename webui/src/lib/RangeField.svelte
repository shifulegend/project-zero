<script>
  // One labelled slider + numeric readout, bounded by a server-provided range
  // ({min, max, step} from GET /v1/config "limits").
  export let label;
  export let value;
  export let range = { min: 0, max: 1, step: 0.01 };
  export let hint = '';
  export let offValue = null; // value that means "disabled" (shown as "off")

  $: decimals = String(range.step).includes('.') ? String(range.step).split('.')[1].length : 0;
  $: display = offValue !== null && value === offValue ? 'off' : Number(value).toFixed(decimals);

  function onInput(e) {
    const v = Number(e.target.value);
    value = decimals === 0 ? Math.round(v) : v;
  }
</script>

<label class="field" title={hint}>
  <span class="head">
    <span>{label}</span>
    <span class="val" class:off={display === 'off'}>{display}</span>
  </span>
  <input type="range" min={range.min} max={range.max} step={range.step} {value} on:input={onInput} />
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
  .head {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
  }
  .val {
    color: var(--text);
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .val.off {
    color: var(--text-dim);
  }
  .hint {
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  input[type='range'] {
    accent-color: var(--accent);
    width: 100%;
  }
</style>

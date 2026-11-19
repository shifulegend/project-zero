<script>
  // Activation numerics, the quality bar for fast settings, and the engine's own accuracy
  // probe (POST /v1/selfcheck). Values come from GET /v1/config runtime; changes are saved
  // as server defaults and apply at the next start, like the other startup settings.
  import { runSelfCheck } from './api.js';

  export let runtime;
  export let writable = false;
  export let saving = false;
  export let onSave = () => {};

  let actQuant = null;
  let tolerance = null;
  $: if (actQuant === null && runtime?.act_quant) {
    actQuant = runtime.act_quant.requested;
    tolerance = runtime.quality_tolerance;
  }
  $: changes = runtime?.act_quant
    ? [
        ...(actQuant !== runtime.act_quant.requested ? [{ key: 'act_quant', value: actQuant }] : []),
        ...(tolerance !== runtime.quality_tolerance ? [{ key: 'quality_tolerance', value: tolerance }] : []),
      ]
    : [];

  // Decimal values (0.5, 2.5): NumberField rounds to integers, which would turn 2.5 into 3.
  function onTolerance(e) {
    let v = Number(e.target.value.trim());
    if (!Number.isFinite(v)) v = runtime.quality_tolerance;
    v = Math.min(100, Math.max(0, v));
    tolerance = v;
    e.target.value = String(v);
  }

  let busy = false;
  let result = null;
  let error = '';
  async function run() {
    busy = true; error = ''; result = null;
    try { result = await runSelfCheck(); }
    catch (e) { error = e.message ?? String(e); }
    busy = false;
  }
</script>

{#if runtime?.act_quant}
  <div class="check">
    <span class="dim">
      Ternary activations: <b>{runtime.act_quant.effective}</b>{runtime.act_quant.requested !== 'auto' ? ` (asked for ${runtime.act_quant.requested})` : ''}
      · LM head: <b>{runtime.classifier}</b>
    </span>
    {#if writable}
      <div class="row gap">
        <label class="text">
          <span>Activation numerics</span>
          <select bind:value={actQuant}>
            <option value="auto">auto (int8 where the CPU has an int8 dot product)</option>
            <option value="int8">int8</option>
            <option value="float">float</option>
          </select>
        </label>
        <label class="text" title="Largest perplexity increase a fast setting may cost; a quantised LM head over it falls back to BF16. 0 = no bar.">
          <span>Quality tolerance (%)</span>
          <input type="number" min="0" max="100" step="0.1" value={tolerance} on:change={onTolerance} />
        </label>
        <button class="link-btn" disabled={saving || !changes.length} on:click={() => onSave(changes)}
                title="Applies when the server restarts">Save for next start</button>
      </div>
    {/if}
    <div class="row gap">
      <button class="link-btn" disabled={busy} on:click={run}
              title="Scores a short passage twice (about {runtime.self_check_tokens} tokens each); waits for any running reply">
        {busy ? 'Checking…' : 'Run accuracy self-check'}
      </button>
      {#if result}
        <span class="mono" class:bad={result.verdict === 'FAIL'}>
          {result.verdict}{result.verdict === 'SKIPPED' ? `: ${result.note}` :
            `: ${result.delta_pct >= 0 ? '+' : ''}${result.delta_pct.toFixed(2)}% perplexity (${result.perplexity.configured.toFixed(2)} vs ${result.perplexity.reference.toFixed(2)}), tolerance ${result.tolerance_pct}%`}
        </span>
      {:else if error}
        <span class="bad">Error: {error}</span>
      {/if}
    </div>
  </div>
{/if}

<style>
  .check { display: flex; flex-direction: column; gap: 0.5rem; }
  .row { display: flex; align-items: center; }
  .row.gap { gap: 0.75rem; flex-wrap: wrap; }
  .dim { color: var(--text-dim); font-size: 0.75rem; }
  .text { display: flex; flex-direction: column; gap: 0.25rem; font-size: 0.8rem; }
  .text select, .text input {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.5rem;
    font: inherit;
  }
  .mono { font-family: var(--mono, ui-monospace, monospace); font-size: 0.8rem; }
  .bad { color: var(--danger, #e5484d); }
  .link-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font: inherit;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .link-btn:hover { border-color: var(--accent); }
  .link-btn:disabled { opacity: 0.5; cursor: default; }
</style>

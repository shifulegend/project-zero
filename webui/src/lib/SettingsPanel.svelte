<script>
  // Every generation knob the engine exposes, rendered from GET /v1/config:
  // defaults, ranges and feature flags all come from the server.
  import RangeField from './RangeField.svelte';
  import NumberField from './NumberField.svelte';
  import PromptCacheStatus from './PromptCacheStatus.svelte';
  import UpdateStatus from './UpdateStatus.svelte';

  export let settings;
  export let config;
  export let onReset = () => {};

  $: L = config?.limits ?? {};
  $: features = config?.features ?? {};
  $: randomSeed = !(Number.isInteger(settings.seed) && settings.seed >= 0);

  function toggleRandomSeed(e) {
    settings.seed = e.target.checked ? -1 : 0;
  }
</script>

<div class="panel">
  <section>
    <h3>Sampling</h3>
    <RangeField label="Temperature" bind:value={settings.temperature} range={L.temperature} hint="0 = greedy (deterministic)" />
    <RangeField label="Top-p" bind:value={settings.top_p} range={L.top_p} />
    <NumberField label="Top-k" bind:value={settings.top_k} range={L.top_k} hint="0 = off" />
    <RangeField label="Min-p" bind:value={settings.min_p} range={L.min_p} offValue={0} />
    <RangeField label="Max tokens" bind:value={settings.max_tokens} range={L.max_tokens} />
    {#if features.best_of}
      <RangeField label="Best of N" bind:value={settings.best_of} range={L.best_of} offValue={1}
        hint="Generate N replies, keep the one the model finds most likely (N× slower; streams once chosen)" />
    {/if}
  </section>

  <section>
    <h3>Repetition</h3>
    <RangeField label="Repetition penalty" bind:value={settings.repetition_penalty} range={L.repetition_penalty} offValue={1} />
    <RangeField label="Presence penalty" bind:value={settings.presence_penalty} range={L.presence_penalty} offValue={0} />
    <RangeField label="Frequency penalty" bind:value={settings.frequency_penalty} range={L.frequency_penalty} offValue={0} />
    <NumberField label="Penalty window (tokens)" bind:value={settings.penalty_last_n} range={L.penalty_last_n} hint="-1 = all generated, 0 = off" />
  </section>

  <section>
    <h3>Output</h3>
    <label class="check">
      <input type="checkbox" checked={randomSeed} on:change={toggleRandomSeed} />
      <span>Random seed</span>
    </label>
    {#if !randomSeed}
      <NumberField label="Seed" bind:value={settings.seed} range={{ min: 0, max: 2147483647, step: 1 }} hint="Same seed + settings = same reply" />
    {/if}
    {#if features.json_mode}
      <label class="check">
        <input type="checkbox" bind:checked={settings.json_mode} />
        <span>JSON mode <span class="dim">(grammar-constrained)</span></span>
      </label>
    {/if}
    <label class="check" title="Never stop at the end-of-sequence token; always generate Max tokens (useful for speed tests).">
      <input type="checkbox" bind:checked={settings.ignore_eos} />
      <span>Ignore end-of-sequence <span class="dim">(run to max tokens)</span></span>
    </label>
    {#if features.prompt_cache}
      <label class="check" title="Skip re-reading the part of the conversation the model has already processed. Output is identical either way.">
        <input type="checkbox" bind:checked={settings.cache_prompt} />
        <span>Reuse prompt cache <span class="dim">(faster multi-turn)</span></span>
      </label>
      {#if features.prompt_store}<PromptCacheStatus />{/if}
    {/if}
    <label class="text">
      <span>Stop sequences <span class="dim">(one per line, max {L.max_stop ?? 0}; \n for newline)</span></span>
      <textarea rows="2" bind:value={settings.stop} spellcheck="false"></textarea>
    </label>
  </section>

  {#if features.watermark}
    <section>
      <h3>Watermark</h3>
      <label class="check">
        <input type="checkbox" bind:checked={settings.watermark_enabled} />
        <span>Watermark replies</span>
      </label>
      <label class="text">
        <span>Key <span class="dim">{config?.defaults?.watermark?.server_key ? '(blank = server key)' : '(required)'}</span></span>
        <input type="password" bind:value={settings.watermark_key} autocomplete="off" />
      </label>
      <RangeField label="Green-list γ" bind:value={settings.watermark_gamma} range={L.watermark_gamma} />
      <RangeField label="Bias δ" bind:value={settings.watermark_delta} range={L.watermark_delta} />
    </section>
  {/if}

  {#if features.system_prompt}
    <section class="wide">
      <h3>System prompt</h3>
      <textarea rows="3" bind:value={settings.system_prompt} placeholder="Instructions applied to every conversation"></textarea>
    </section>
  {/if}

  <section>
    <h3>Updates</h3>
    <UpdateStatus />
  </section>

  <div class="actions">
    <button class="link-btn" on:click={onReset} title="Discard your overrides and use the server's defaults">Reset to server defaults</button>
  </div>
</div>

<style>
  .panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
    gap: 1rem 1.5rem;
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
    max-height: 55vh;
    overflow-y: auto;
  }
  section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    min-width: 0;
  }
  section.wide {
    grid-column: 1 / -1;
  }
  h3 {
    margin: 0;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .check {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    cursor: pointer;
  }
  .check input {
    accent-color: var(--accent);
  }
  .text {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-dim);
  }
  .dim {
    color: var(--text-dim);
    font-size: 0.75rem;
  }
  textarea, .text input {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-family: inherit;
    font-size: 0.85rem;
    resize: vertical;
    box-sizing: border-box;
    width: 100%;
  }
  textarea:focus, .text input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .actions {
    grid-column: 1 / -1;
    display: flex;
    justify-content: flex-end;
  }
  .link-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .link-btn:hover {
    border-color: var(--accent);
  }
</style>

<script lang="ts">
  import NumberField from './NumberField.svelte';
  import PromptCacheStatus from './PromptCacheStatus.svelte';
  import type { Features, Ranges, Settings } from './types';

  export let settings: Settings;
  export let L: Ranges;
  export let features: Features;

  $: randomSeed = !(Number.isInteger(settings.seed) && settings.seed >= 0);

  function toggleRandomSeed(e: Event): void {
    settings.seed = (e.target as HTMLInputElement).checked ? -1 : 0;
  }

</script>

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

<style>
  section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    min-width: 0;
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
  textarea {
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
  textarea:focus {
    outline: none;
    border-color: var(--accent);
  }
</style>

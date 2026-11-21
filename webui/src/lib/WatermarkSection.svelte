<script lang="ts">
  import RangeField from './RangeField.svelte';
  import type { Ranges, ServerConfig, Settings } from './types';

  export let settings: Settings;
  export let L: Ranges;
  export let config: ServerConfig;

</script>

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
  .text input {
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
  .text input:focus {
    outline: none;
    border-color: var(--accent);
  }
</style>

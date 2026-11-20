<script lang="ts">
  // Every generation knob the engine exposes, rendered from GET /v1/config:
  // defaults, ranges and feature flags all come from the server. Each group of controls is its own
  // section component; this panel lays them out.
  import SamplingSection from './SamplingSection.svelte';
  import TreeSearchSection from './TreeSearchSection.svelte';
  import RepetitionSection from './RepetitionSection.svelte';
  import OutputSection from './OutputSection.svelte';
  import WatermarkSection from './WatermarkSection.svelte';
  import TemplateSection from './TemplateSection.svelte';
  import ServerDefaultsSection from './ServerDefaultsSection.svelte';
  import UpdateStatus from './UpdateStatus.svelte';
  import type { Ranges, ServerConfig, Settings } from './types';

  export let settings: Settings;
  export let config: ServerConfig;
  export let onReset: () => void = () => {};

  $: L = (config?.limits ?? {}) as Ranges;
  $: features = config?.features ?? {};
</script>

<div class="panel">
  <SamplingSection bind:settings {L} {features} />
  {#if features.mcts}<TreeSearchSection bind:settings {L} {features} />{/if}
  <RepetitionSection bind:settings {L} />
  <OutputSection bind:settings {L} {features} />
  {#if features.watermark}<WatermarkSection bind:settings {L} {config} />{/if}

  {#if features.system_prompt}
    <section class="wide">
      <h3>System prompt</h3>
      <textarea rows="3" bind:value={settings.system_prompt} placeholder="Instructions applied to every conversation"></textarea>
    </section>
  {/if}

  {#if config?.chat_template}<TemplateSection bind:settings {config} {features} />{/if}
  {#if config?.config}<ServerDefaultsSection bind:settings {config} />{/if}

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
  .link-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  h3 {
    margin: 0;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    font-family: ui-monospace, Menlo, Consolas, monospace;
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

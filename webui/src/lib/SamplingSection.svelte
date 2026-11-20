<script lang="ts">
  import RangeField from './RangeField.svelte';
  import NumberField from './NumberField.svelte';
  import type { Features, Ranges, Settings } from './types';

  export let settings: Settings;
  export let L: Ranges;
  export let features: Features;

</script>

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
</style>

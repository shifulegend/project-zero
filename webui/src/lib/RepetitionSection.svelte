<script lang="ts">
  import RangeField from './RangeField.svelte';
  import NumberField from './NumberField.svelte';
  import type { Ranges, Settings } from './types';

  export let settings: Settings;
  export let L: Ranges;

</script>

  <section>
    <h3>Repetition</h3>
    <RangeField label="Repetition penalty" bind:value={settings.repetition_penalty} range={L.repetition_penalty} offValue={1} />
    <RangeField label="Presence penalty" bind:value={settings.presence_penalty} range={L.presence_penalty} offValue={0} />
    <RangeField label="Frequency penalty" bind:value={settings.frequency_penalty} range={L.frequency_penalty} offValue={0} />
    <NumberField label="Penalty window (tokens)" bind:value={settings.penalty_last_n} range={L.penalty_last_n} hint="-1 = all generated, 0 = off" />
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

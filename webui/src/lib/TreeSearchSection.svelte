<script lang="ts">
  import RangeField from './RangeField.svelte';
  import type { Features, Ranges, Settings } from './types';

  export let settings: Settings;
  export let L: Ranges;
  export let features: Features;

</script>

<section>
  <h3>Tree search</h3>
  <RangeField label="Simulations" bind:value={settings.mcts_simulations} range={L.mcts_simulations} offValue={0}
    hint="Step-level MCTS: explore alternative reasoning steps, answer along the best path (slower; streams once chosen)" />
  {#if settings.mcts_simulations > 0}
    <RangeField label="Branching" bind:value={settings.mcts_expand} range={L.mcts_expand} />
    <RangeField label="Max steps" bind:value={settings.mcts_max_depth} range={L.mcts_max_depth} />
    <RangeField label="Tokens per step" bind:value={settings.mcts_step_tokens} range={L.mcts_step_tokens} />
    <RangeField label="Exploration (c)" bind:value={settings.mcts_c} range={L.mcts_c} />
    <RangeField label="Step temperature" bind:value={settings.mcts_temperature} range={L.mcts_temperature} />
  {/if}
  <span class="dim">Steps scored by {features.prm ? `reward model ${features.prm}` : "the model's own token probability (start the server with --prm for a judge)"}</span>
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
  .dim {
    color: var(--text-dim);
    font-size: 0.75rem;
  }
</style>

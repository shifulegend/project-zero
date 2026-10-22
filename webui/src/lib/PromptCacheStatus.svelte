<script>
  // Phase 26: how much of the server's prompt-snapshot budget is in use, with
  // a manual clear. Refreshed when shown and after clearing.
  import { onMount } from 'svelte';
  import { fetchPromptCache, clearPromptCache } from './api.js';

  let stats = null;
  let error = null;
  let busy = false;

  const mb = (b) => (b / (1024 * 1024)).toFixed(1);

  async function refresh() {
    try { stats = await fetchPromptCache(); error = null; }
    catch (e) { error = e.message ?? String(e); }
  }

  async function clear() {
    busy = true;
    try { stats = await clearPromptCache(); error = null; }
    catch (e) { error = e.message ?? String(e); }
    busy = false;
  }

  onMount(refresh);
</script>

<div class="status">
  {#if stats}
    <span class="mono">{stats.entries} cached · {mb(stats.bytes_used)} / {mb(stats.budget_bytes)} MB · {stats.hits} hits</span>
    <button class="small" on:click={clear} disabled={busy} title="Drop every stored conversation prefix">Clear</button>
  {:else if error}
    <span class="err">Error: {error}</span>
  {/if}
</div>

<style>
  .status {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: var(--text-dim);
  }
  .mono {
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .small {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    font-size: 0.75rem;
    cursor: pointer;
  }
  .small:hover:not(:disabled) {
    border-color: var(--accent);
  }
  .err {
    color: #e5484d;
  }
</style>

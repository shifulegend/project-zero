<script>
  export let modelInfo = null;
</script>

<div class="model-info">
  <span class="dot" class:online={modelInfo}></span>
  {#if modelInfo}
    <span>{modelInfo.id}</span>
    <span class="sep">·</span>
    <span class="dim">served by {modelInfo.owned_by}</span>
  {:else}
    <span class="dim">connecting to server…</span>
  {/if}
</div>

<style>
  .model-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.25rem;
    font-size: 0.8rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .dot {
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: var(--text-dim);
  }
  .dot.online {
    background: var(--accent);
  }
  .sep {
    color: var(--text-dim);
  }
  .dim {
    color: var(--text-dim);
  }
</style>

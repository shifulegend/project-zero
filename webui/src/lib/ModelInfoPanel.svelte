<script>
  export let modelInfo = null;
</script>

<div class="model-info">
  <span class="dot" class:online={modelInfo}></span>
  {#if modelInfo}
    <span>{modelInfo.id}</span>
    <span class="sep">·</span>
    {#if modelInfo.context_length}
      <span class="dim mono">{modelInfo.n_layers}L · d{modelInfo.dim} · ctx {modelInfo.context_length}</span>
      <span class="sep">·</span>
    {/if}
    <span class="dim">served by {modelInfo.owned_by}</span>
    {#if modelInfo.vision}<span class="tag">vision</span>{/if}
    {#if modelInfo.moe}<span class="tag">MoE</span>{/if}
  {:else}
    <span class="dim">connecting to server…</span>
  {/if}
</div>

<style>
  .model-info {
    display: flex;
    flex-wrap: wrap;
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
  .mono {
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .tag {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0 0.3rem;
    font-size: 0.7rem;
    color: var(--text-dim);
  }
</style>

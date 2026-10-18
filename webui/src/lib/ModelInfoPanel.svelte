<script>
  export let modelInfo = null;
  export let runtime = null; // GET /v1/config "runtime"
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
    {#if runtime}
      <span class="sep">·</span>
      <span class="dim mono" title="SIMD backend · threads · attention kernel">{runtime.simd} · {runtime.threads}T · attn {runtime.attention_kernel}</span>
    {/if}
    {#if runtime?.distributed?.length}
      <span class="tag mono" title={runtime.distributed.map((n) => `layers ${n.layers} → ${n.host}:${n.port}`).join('\n')}>
        dist {runtime.distributed.length + 1} nodes
      </span>
    {/if}
    {#if modelInfo.vision}<span class="tag">vision</span>{/if}
    {#if modelInfo.moe}<span class="tag">MoE</span>{/if}
    {#if modelInfo.recurrent}<span class="tag" title="Recurrent state: prompt caching is not used">SSM / recurrent</span>{/if}
    {#if modelInfo.chat_format && modelInfo.chat_format !== 'embedded'}<span class="tag" title="Prompt format (no embedded chat template)">{modelInfo.chat_format} format</span>{/if}
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

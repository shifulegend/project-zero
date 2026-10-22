<script>
  // Phase 38: one quiet line when the server knows of a newer release.
  // Dismissal is remembered per version, so the next release shows again.
  import { onMount } from 'svelte';
  import { updateInfo, showBanner, dismissedVersion, refreshUpdate } from './update.js';

  onMount(refreshUpdate);

  $: info = $updateInfo;
  $: visible = info?.status === 'available' && $showBanner && $dismissedVersion !== info.latest;
</script>

{#if visible}
  <div class="update" role="status">
    <span class="mark">&gt;_</span>
    <span>update available</span>
    <span class="mono">{info.current} → {info.latest}</span>
    {#if info.prerelease}<span class="tag">pre-release</span>{/if}
    {#if info.url}
      <a class="link" href={info.url} target="_blank" rel="noopener noreferrer">release notes</a>
    {/if}
    <span class="dim hint">verify with SHA256SUMS before installing</span>
    <button class="dismiss" on:click={() => dismissedVersion.set(info.latest)} title="Hide until the next release">dismiss</button>
  </div>
{/if}

<style>
  .update {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 1.25rem;
    font-size: 0.8rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }
  .mark {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    color: var(--accent);
  }
  .mono {
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .dim {
    color: var(--text-dim);
  }
  .tag {
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0 0.3rem;
    font-size: 0.7rem;
    color: var(--text-dim);
  }
  .link {
    color: var(--text);
    text-decoration: underline;
    text-underline-offset: 2px;
  }
  .dismiss {
    margin-left: auto;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.75rem;
    cursor: pointer;
  }
  .dismiss:hover {
    border-color: var(--accent);
  }
  @media (max-width: 640px) {
    .hint { display: none; }
  }
</style>

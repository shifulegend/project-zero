<script>
  // Phase 38: Settings → Updates. Shows what the server checks and lets the
  // user re-check or hide the banner. Interval / channel / repo / opt-out are
  // server flags (--update-*), shown read-only so nothing is hidden.
  import { updateInfo, updateError, showBanner, checkNow, refreshUpdate } from './update.js';
  import { onMount } from 'svelte';

  let busy = false;
  onMount(() => { if (!$updateInfo) refreshUpdate(); });

  async function check() {
    busy = true;
    await checkNow();
    busy = false;
  }

  const labels = {
    available: 'update available',
    up_to_date: 'up to date',
    unknown: 'not checked yet',
    disabled: 'checks off',
    error: 'check failed',
  };

  $: info = $updateInfo;
  $: s = info?.settings;
  $: when = info?.checked_at ? new Date(info.checked_at * 1000).toLocaleString() : 'never';
</script>

<div class="upd">
  {#if info}
    <div class="row">
      <span class="state" class:on={info.status === 'available'}>{labels[info.status] ?? info.status}</span>
      <span class="mono dim">{info.current}{info.latest ? ` · latest ${info.latest}` : ''}</span>
    </div>
    <div class="row dim mono small-text">
      {#if s?.enabled}
        every {s.interval_hours} h · {s.channel} channel · {s.repo} · last {when}
      {:else}
        disabled on the server (--no-update-check, --update-interval 0, PZ_NO_UPDATE_CHECK or CI)
      {/if}
    </div>
    {#if info.error}<div class="row err small-text">{info.error}</div>{/if}
    <div class="row">
      <label class="check">
        <input type="checkbox" bind:checked={$showBanner} />
        <span>Show update banner</span>
      </label>
      <button class="small" on:click={check} disabled={busy || !s?.enabled}
              title="Ask the server to check GitHub Releases now (at most once a minute)">
        {busy ? 'Checking…' : 'Check now'}
      </button>
    </div>
  {:else if $updateError}
    <span class="err">Error: {$updateError}</span>
  {/if}
</div>

<style>
  .upd {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    font-size: 0.8rem;
  }
  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    flex-wrap: wrap;
  }
  .state {
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .state.on {
    color: var(--accent);
  }
  .mono {
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .dim {
    color: var(--text-dim);
  }
  .small-text {
    font-size: 0.72rem;
  }
  .check {
    display: flex;
    align-items: center;
    gap: 0.4rem;
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

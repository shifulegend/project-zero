<script>
  import { onMount } from 'svelte';
  import ChatWindow from './lib/ChatWindow.svelte';
  import SamplingControls from './lib/SamplingControls.svelte';
  import ThemeToggle from './lib/ThemeToggle.svelte';
  import ModelInfoPanel from './lib/ModelInfoPanel.svelte';

  let temperature = 0.7;
  let top_p = 0.9;
  let max_tokens = 512;
  let modelInfo = null;
  let controlsOpen = false;

  onMount(async () => {
    try {
      const res = await fetch('/v1/models');
      if (res.ok) {
        const json = await res.json();
        modelInfo = json.data?.[0] ?? null;
      }
    } catch {
      /* server not reachable yet / no models — panel just shows "unknown" */
    }
  });
</script>

<div class="shell">
  <header class="topbar">
    <div class="brand">
      <span class="brand-mark">&gt;_</span>
      <span class="brand-name">project-zero</span>
    </div>
    <div class="topbar-actions">
      <button class="icon-btn" on:click={() => (controlsOpen = !controlsOpen)} aria-expanded={controlsOpen} title="Sampling parameters">
        Params
      </button>
      <ThemeToggle />
    </div>
  </header>

  {#if controlsOpen}
    <SamplingControls bind:temperature bind:top_p bind:max_tokens />
  {/if}

  <ModelInfoPanel {modelInfo} />

  <main class="main">
    <ChatWindow {temperature} {top_p} {max_tokens} visionCapable={modelInfo?.vision === true} />
  </main>
</div>

<style>
  :global(:root) {
    --bg: #f7f8fa;
    --panel: #ffffff;
    --border: #d8dce3;
    --text: #1a1d23;
    --text-dim: #5b6270;
    --accent: #1f8a5f;
    --accent-contrast: #ffffff;
    --user-bubble: #eef1f5;
    --code-bg: #eef0f3;
  }

  :global(:root[data-theme='dark']) {
    --bg: #0b0d12;
    --panel: #12151c;
    --border: #262b36;
    --text: #e6e8ee;
    --text-dim: #8b93a3;
    --accent: #7ee2a8;
    --accent-contrast: #0b0d12;
    --user-bubble: #1a1e27;
    --code-bg: #0d1b14;
  }

  @media (prefers-color-scheme: dark) {
    :global(:root:not([data-theme='light'])) {
      --bg: #0b0d12;
      --panel: #12151c;
      --border: #262b36;
      --text: #e6e8ee;
      --text-dim: #8b93a3;
      --accent: #7ee2a8;
      --accent-contrast: #0b0d12;
      --user-bubble: #1a1e27;
      --code-bg: #0d1b14;
    }
  }

  :global(html, body) {
    margin: 0;
    height: 100%;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  }

  :global(#app) {
    height: 100%;
  }

  .shell {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }

  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.25rem;
    border-bottom: 1px solid var(--border);
    background: var(--panel);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    font-size: 1.05rem;
  }

  .brand-mark {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    color: var(--accent);
  }

  .topbar-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .icon-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.85rem;
    cursor: pointer;
  }

  .icon-btn:hover {
    border-color: var(--accent);
  }

  .main {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
</style>

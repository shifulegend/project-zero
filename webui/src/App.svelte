<script>
  import { onMount } from 'svelte';
  import ChatWindow from './lib/ChatWindow.svelte';
  import SettingsPanel from './lib/SettingsPanel.svelte';
  import WatermarkPanel from './lib/WatermarkPanel.svelte';
  import ThemeToggle from './lib/ThemeToggle.svelte';
  import ModelInfoPanel from './lib/ModelInfoPanel.svelte';
  import UpdateBanner from './lib/UpdateBanner.svelte';
  import AsciiLogo from './lib/AsciiLogo.svelte';
  import { fetchConfig } from './lib/api.js';
  import { loadSettings, saveSettings, clearSavedSettings, defaultsFromConfig, countOverrides } from './lib/settings.js';

  // Everything below is populated from GET /v1/config — the UI carries no
  // default sampling values, limits or branding of its own.
  let config = null;
  let settings = null;
  let configError = null;
  let panel = null; // null | 'settings' | 'watermark'
  let chat;

  onMount(async () => {
    try {
      config = await fetchConfig();
      settings = loadSettings(config);
      if (config?.ui?.title) document.title = config.ui.title;
    } catch (e) {
      configError = e.message ?? String(e);
    }
  });

  $: if (config && settings) saveSettings(settings, config);
  $: brand = config?.ui?.title ?? '';
  $: logoAscii = config?.ui?.logo === 'ascii' ? config?.ui?.logo_ascii ?? '' : '';
  $: modelInfo = config?.model ?? null;
  $: overrides = config && settings ? countOverrides(settings, config) : 0;

  function toggle(name) {
    panel = panel === name ? null : name;
  }

  function resetSettings() {
    clearSavedSettings();
    settings = defaultsFromConfig(config);
  }
</script>

<div class="shell">
  <header class="topbar">
    <div class="brand">
      {#if logoAscii}
        <AsciiLogo rows={logoAscii} label={brand} size="sm" />
      {:else}
        <span class="brand-name">{brand}</span>
      {/if}
    </div>
    <div class="topbar-actions">
      <button class="icon-btn" on:click={() => chat?.clearConversation()} title="Start a new conversation" disabled={!settings}>New chat</button>
      <button class="icon-btn" class:active={panel === 'settings'} on:click={() => toggle('settings')} aria-expanded={panel === 'settings'} title="Generation settings" disabled={!settings}>
        Settings{#if overrides}<span class="badge" title="{overrides} setting(s) differ from the server defaults">{overrides}</span>{/if}
      </button>
      {#if config?.features?.watermark}
        <button class="icon-btn" class:active={panel === 'watermark'} on:click={() => toggle('watermark')} aria-expanded={panel === 'watermark'} title="Check text for this server's watermark">Watermark</button>
      {/if}
      <ThemeToggle />
    </div>
  </header>

  {#if panel === 'settings' && settings}
    <SettingsPanel bind:settings {config} onReset={resetSettings} />
  {:else if panel === 'watermark' && config}
    <WatermarkPanel {config} defaultKey={settings?.watermark_key ?? ''} />
  {/if}

  <ModelInfoPanel {modelInfo} runtime={config?.runtime ?? null} />
  {#if config}<UpdateBanner />{/if}

  <main class="main">
    {#if configError}
      <div class="boot-error" role="alert"><strong>Error:</strong> could not load server configuration ({configError}).</div>
    {:else if settings}
      <ChatWindow bind:this={chat} {settings} {config} {brand} {logoAscii} visionCapable={modelInfo?.vision === true} />
    {/if}
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
    flex-wrap: wrap;
    gap: 0.5rem;
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

  .topbar-actions {
    display: flex;
    flex-wrap: wrap;
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

  .icon-btn:hover:not(:disabled),
  .icon-btn.active {
    border-color: var(--accent);
  }

  .icon-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .badge {
    display: inline-block;
    margin-left: 0.35rem;
    min-width: 1.1rem;
    padding: 0 0.25rem;
    border-radius: 4px;
    background: var(--accent);
    color: var(--accent-contrast);
    font-size: 0.7rem;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    text-align: center;
  }

  .boot-error {
    margin: auto;
    color: #e5484d;
    font-size: 0.9rem;
  }

  .main {
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
</style>

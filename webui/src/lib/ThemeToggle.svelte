<script lang="ts">
  import { onMount } from 'svelte';

  type Theme = 'auto' | 'light' | 'dark';
  let theme: Theme = 'auto';

  function apply(t: Theme): void {
    const root = document.documentElement;
    if (t === 'auto') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', t);
  }

  function toggle(): void {
    theme = theme === 'dark' ? 'light' : theme === 'light' ? 'auto' : 'dark';
    apply(theme);
    try { localStorage.setItem('pz-theme', theme); } catch { /* private mode */ }
  }

  onMount(() => {
    try {
      const saved = localStorage.getItem('pz-theme');
      if (saved === 'auto' || saved === 'light' || saved === 'dark') { theme = saved; apply(theme); }
    } catch { /* private mode */ }
  });
</script>

<button class="theme-toggle" on:click={toggle} title="Cycle theme (dark / light / auto)">
  {#if theme === 'dark'}Dark{:else if theme === 'light'}Light{:else}Auto{/if}
</button>

<style>
  .theme-toggle {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.85rem;
    cursor: pointer;
    min-width: 4.5rem;
  }
  .theme-toggle:hover {
    border-color: var(--accent);
  }
</style>

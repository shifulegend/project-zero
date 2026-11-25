<script lang="ts">
  import NumberField from './NumberField.svelte';
  import AccuracyCheck from './AccuracyCheck.svelte';
  import { serverDefaultChanges, sourcesSummary } from './settings';
  import { saveServerDefault } from './api';
  import { errorText } from './errors';
  import type { ConfigInfo, ServerConfig, Settings } from './types';

  export let settings: Settings;
  export let config: ServerConfig;

  // Phase 40.3: save this browser's changes as the server's own defaults
  // (written to the `pz config` file; the server applies them at its next start).
  const SOURCE_LABEL: Record<string, string> = { file: 'config file', env: 'environment', cli: 'command line' };
  // The parent only renders this section when the server reports a `config` block.
  $: features = config?.features ?? {};
  $: cfgInfo = config.config as ConfigInfo;
  $: changes = config && settings ? serverDefaultChanges(settings, config) : [];
  $: sources = sourcesSummary(config);
  let title: string | null = null;
  $: if (title === null && config?.ui?.title) title = config.ui.title;
  // Phase 23 stage 2b: how many chat requests the server answers at once
  // (--parallel) and its batch size; saved for the next start like the title.
  $: par = features.parallel ?? null;
  let parallel: number | undefined;
  let parallelBatch: number | undefined;
  $: if (parallel === undefined && par) { parallel = par.requested; parallelBatch = par.batch_tokens; }
  $: parallelChanges = par
    ? [
        ...(parallel !== par.requested ? [{ key: 'parallel', value: parallel }] : []),
        ...(parallelBatch !== par.batch_tokens ? [{ key: 'parallel_batch', value: parallelBatch }] : []),
      ]
    : [];
  let saving = false;
  let saveMsg = '';
  let saveErr = '';

  async function saveAll(items: { key: string; value: unknown }[]): Promise<void> {
    saving = true;
    saveMsg = '';
    saveErr = '';
    const overridden: string[] = [];
    try {
      for (const { key, value } of items) {
        const r = await saveServerDefault(key, value);
        if (r.overridden_by) overridden.push(`${key} (${SOURCE_LABEL[r.overridden_by] ?? r.overridden_by})`);
      }
      saveMsg = `Saved ${items.length} setting${items.length === 1 ? '' : 's'} to ${cfgInfo.path}. ` +
        'They apply when the server restarts.' +
        (cfgInfo.file_used ? '' : ' Note: this server was started with --no-config, so it ignores the file.') +
        (overridden.length ? ` Still overridden at startup by: ${overridden.join(', ')}.` : '');
    } catch (e) {
      saveErr = errorText(e);
    } finally {
      saving = false;
    }
  }
</script>

  <section class="wide">
    <h3>Server defaults</h3>
    {#if Object.keys(sources).length}
      <span class="dim">
        Server values not at their built-in default:
        {#each Object.entries(sources) as [src, keys], i}{i ? ' · ' : ''}<b>{SOURCE_LABEL[src] ?? src}</b>: {keys.join(', ')}{/each}
      </span>
    {:else}
      <span class="dim">Every server value is the built-in default.</span>
    {/if}
    {#if par}
      <span class="dim">
        Answers {par.slots} chat request{par.slots === 1 ? '' : 's'} at once{par.slots > 1 ? ', batched into one pass per token' : ''}{par.note ? ` (asked for ${par.requested}: ${par.note})` : ''}.
      </span>
    {/if}
    {#if cfgInfo.writable && par}
      <div class="row gap">
        <NumberField label="Requests at once" bind:value={parallel} range={{ min: 1, max: par.max, step: 1 }}
                     hint="More users served together; each one's reply gets a little slower. 1 = one at a time." />
        <NumberField label="Tokens per batched pass" bind:value={parallelBatch}
                     range={{ min: 1, max: par.batch_tokens_max, step: 1 }}
                     hint="How much of waiting prompts is read per step while others are replying." />
        <button class="link-btn" disabled={saving || !parallelChanges.length}
                on:click={() => saveAll(parallelChanges)}
                title="Write to {cfgInfo.path}; applies when the server restarts">Save for next start</button>
      </div>
    {/if}
    <AccuracyCheck runtime={config?.runtime} writable={cfgInfo.writable} {saving} onSave={saveAll} />
    {#if cfgInfo.writable}
      <label class="text">
        <span>Web UI title <span class="dim">(also drawn as the wordmark)</span></span>
        <div class="row gap">
          <input type="text" bind:value={title} maxlength="64" />
          <button class="link-btn" disabled={saving || !title?.trim() || title === config.ui?.title}
                  on:click={() => saveAll([{ key: 'ui_title', value: title?.trim() }])}>Save title</button>
        </div>
      </label>
      <div class="row gap">
        <span class="dim">{changes.length ? `Changed here: ${changes.map((c) => c.key).join(', ')}` : 'No changes from the server defaults.'}</span>
        <button class="link-btn" disabled={saving || !changes.length} on:click={() => saveAll(changes)}
                title="Write these values to {cfgInfo.path}; the server uses them from its next start">
          {saving ? 'Saving…' : `Save ${changes.length || ''} as server defaults`}
        </button>
      </div>
    {:else}
      <span class="dim">{cfgInfo.path ? 'Saving is disabled on this server (--no-config-write).' : 'This server has no config file location.'}
        Use <code>pz config set</code> on the server instead.</span>
    {/if}
    {#if saveMsg}<span class="ok">{saveMsg}</span>{/if}
    {#if saveErr}<span class="err" role="alert">Error: {saveErr}</span>{/if}
  </section>

<style>
  section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    min-width: 0;
  }
  section.wide {
    grid-column: 1 / -1;
  }
  .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .row.gap {
    gap: 0.75rem;
  }
  .row.gap input {
    flex: 1;
  }
  .ok {
    color: var(--accent);
    font-size: 0.8rem;
  }
  .err {
    color: #e5484d;
    font-size: 0.8rem;
  }
  .link-btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  h3 {
    margin: 0;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .text {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-dim);
  }
  .dim {
    color: var(--text-dim);
    font-size: 0.75rem;
  }
  .text input {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-family: inherit;
    font-size: 0.85rem;
    resize: vertical;
    box-sizing: border-box;
    width: 100%;
  }
  .text input:focus {
    outline: none;
    border-color: var(--accent);
  }
  .link-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 0.4rem 0.7rem;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .link-btn:hover {
    border-color: var(--accent);
  }
</style>

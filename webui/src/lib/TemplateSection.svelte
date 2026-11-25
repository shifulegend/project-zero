<script lang="ts">
  import type { ChatTemplateInfo, Features, ServerConfig, Settings } from './types';

  export let settings: Settings;
  export let config: ServerConfig;
  export let features: Features;

  // The parent only renders this section when the server reports a chat template.
  $: tmpl = config.chat_template as ChatTemplateInfo;
  $: templateEdited = (settings.chat_template ?? '') !== (config?.chat_template?.text ?? '');

</script>

<section class="wide">
  <h3>Chat template <span class="tag">{templateEdited ? 'edited · sent per request' : `model · ${tmpl.source}`}</span></h3>
  {#if features.request_chat_template}
    <textarea class="mono" rows="6" spellcheck="false" bind:value={settings.chat_template}
              maxlength={tmpl.max_bytes}
              placeholder="Jinja template — messages, bos_token, eos_token, add_generation_prompt"></textarea>
    <div class="row">
      <span class="dim">{(settings.chat_template ?? '').length} / {tmpl.max_bytes} chars</span>
      {#if templateEdited}
        <button class="link-btn" on:click={() => (settings.chat_template = config.chat_template?.text ?? '')}>Restore model template</button>
      {/if}
    </div>
  {:else}
    <pre class="mono readonly">{tmpl.text || '(none — legacy chat format)'}</pre>
    <span class="dim">The server ignores per-request templates (--no-request-template).</span>
  {/if}
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
  .mono {
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.75rem;
  }
  pre.readonly {
    margin: 0;
    padding: 0.5rem;
    max-height: 10rem;
    overflow: auto;
    white-space: pre-wrap;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .tag {
    text-transform: none;
    letter-spacing: 0;
    margin-left: 0.4rem;
    color: var(--accent, var(--text-dim));
  }
  .row {
    display: flex;
    justify-content: space-between;
    align-items: center;
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
  .dim {
    color: var(--text-dim);
    font-size: 0.75rem;
  }
  textarea {
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
  textarea:focus {
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

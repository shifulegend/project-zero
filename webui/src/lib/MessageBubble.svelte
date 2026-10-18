<script>
  import { renderMarkdown } from './markdown.js';

  export let role; // 'user' | 'assistant'
  export let content;
  export let imageDataUrl = null;
  export let streaming = false;
  export let meta = null; // {finish_reason, usage} once the reply completes
  export let brand = '';

  const FINISH_LABEL = { stop: 'complete', length: 'hit max tokens', cancelled: 'stopped' };
</script>

<div class="bubble {role}">
  <div class="role-label">{role === 'user' ? 'You' : brand}</div>
  {#if imageDataUrl}
    <img class="attached-image" src={imageDataUrl} alt="attached" />
  {/if}
  <div class="content">
    {@html renderMarkdown(content)}{#if streaming}<span class="cursor">▌</span>{/if}
  </div>
  {#if role === 'assistant' && meta && !streaming}
    <div class="meta">
      {#if meta.usage}{meta.usage.completion_tokens} tokens · {meta.usage.prompt_tokens} prompt{/if}
      {#if meta.finish_reason}<span class:warn={meta.finish_reason === 'length'}> · {FINISH_LABEL[meta.finish_reason] ?? meta.finish_reason}</span>{/if}
    </div>
  {/if}
</div>

<style>
  .bubble {
    max-width: 42rem;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    line-height: 1.5;
  }
  .bubble.user {
    align-self: flex-end;
    background: var(--user-bubble);
  }
  .bubble.assistant {
    align-self: flex-start;
    background: transparent;
    border: 1px solid var(--border);
  }
  .role-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
    margin-bottom: 0.25rem;
  }
  .content {
    word-wrap: break-word;
    overflow-wrap: break-word;
  }
  .content :global(pre.md-code-block) {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    overflow-x: auto;
    font-size: 0.85rem;
  }
  .content :global(code.md-inline-code) {
    background: var(--code-bg);
    border-radius: 4px;
    padding: 0.1rem 0.3rem;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.9em;
  }
  .attached-image {
    max-width: 12rem;
    max-height: 12rem;
    border-radius: 8px;
    display: block;
    margin-bottom: 0.5rem;
  }
  .meta {
    margin-top: 0.4rem;
    font-size: 0.7rem;
    color: var(--text-dim);
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }
  .meta .warn {
    color: var(--text);
  }
  .cursor {
    color: var(--accent);
    animation: blink 1s step-start infinite;
  }
  @keyframes blink {
    50% { opacity: 0; }
  }
</style>

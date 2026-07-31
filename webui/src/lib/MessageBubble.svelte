<script>
  import { renderMarkdown } from './markdown.js';

  export let role; // 'user' | 'assistant'
  export let content;
  export let imageDataUrl = null;
  export let streaming = false;
</script>

<div class="bubble {role}">
  <div class="role-label">{role === 'user' ? 'You' : 'project-zero'}</div>
  {#if imageDataUrl}
    <img class="attached-image" src={imageDataUrl} alt="attached" />
  {/if}
  <div class="content">
    {@html renderMarkdown(content)}{#if streaming}<span class="cursor">▌</span>{/if}
  </div>
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
  .cursor {
    color: var(--accent);
    animation: blink 1s step-start infinite;
  }
  @keyframes blink {
    50% { opacity: 0; }
  }
</style>

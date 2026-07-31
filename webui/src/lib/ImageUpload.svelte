<script>
  export let imageDataUrl = null;

  function onFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { imageDataUrl = reader.result; };
    reader.readAsDataURL(file);
  }

  function clear() {
    imageDataUrl = null;
  }
</script>

<div class="image-upload">
  {#if imageDataUrl}
    <div class="preview">
      <img src={imageDataUrl} alt="attached preview" />
      <button type="button" class="remove" on:click={clear} aria-label="Remove image">×</button>
    </div>
  {:else}
    <label class="attach-btn" title="Attach an image" aria-label="Attach an image">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
      </svg>
      <input type="file" accept="image/*" on:change={onFileChange} hidden />
    </label>
  {/if}
</div>

<style>
  .image-upload {
    display: flex;
    align-items: center;
  }
  .attach-btn {
    cursor: pointer;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    font-size: 1rem;
    display: inline-flex;
  }
  .attach-btn:hover {
    border-color: var(--accent);
  }
  .preview {
    position: relative;
  }
  .preview img {
    width: 2.5rem;
    height: 2.5rem;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
  }
  .remove {
    position: absolute;
    top: -0.4rem;
    right: -0.4rem;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 50%;
    width: 1.2rem;
    height: 1.2rem;
    line-height: 1;
    cursor: pointer;
    font-size: 0.8rem;
  }
</style>

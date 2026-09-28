<script>
  import { createBlock, BLOCK_TYPES } from './lib/blocks.js';
  import { serializeEmail } from './lib/serialize.js';
  import BlockEditor from './lib/components/BlockEditor.svelte';
  import Preview from './lib/components/Preview.svelte';

  const initialBlocks = [createBlock('heading'), createBlock('paragraph'), createBlock('button')];
  let blocks = $state(initialBlocks);
  let selectedId = $state(initialBlocks[0]?.id ?? null);
  let previewWidth = $state(600);
  let copyStatus = $state('');

  const html = $derived(serializeEmail(blocks, { width: previewWidth }));

  function addBlock(type) {
    const block = createBlock(type);
    blocks = [...blocks, block];
    selectedId = block.id;
  }

  function removeBlock(id) {
    blocks = blocks.filter((b) => b.id !== id);
    if (selectedId === id) selectedId = blocks[0]?.id ?? null;
  }

  function moveBlock(id, direction) {
    const index = blocks.findIndex((b) => b.id === id);
    const target = index + direction;
    if (index === -1 || target < 0 || target >= blocks.length) return;
    const next = blocks.slice();
    [next[index], next[target]] = [next[target], next[index]];
    blocks = next;
  }

  function updateProps(id, props) {
    blocks = blocks.map((b) => (b.id === id ? { ...b, props: { ...b.props, ...props } } : b));
  }

  async function copyHtml() {
    try {
      await navigator.clipboard.writeText(html);
      copyStatus = 'Copied!';
    } catch {
      copyStatus = 'Copy failed';
    }
    setTimeout(() => (copyStatus = ''), 2000);
  }
</script>

<div class="app">
  <header class="toolbar">
    <h1>Email Template Builder</h1>
    <div class="width-toggle" role="group" aria-label="Preview width">
      <button type="button" class:active={previewWidth === 600} onclick={() => (previewWidth = 600)}>
        Desktop (600px)
      </button>
      <button type="button" class:active={previewWidth === 375} onclick={() => (previewWidth = 375)}>
        Mobile (375px)
      </button>
    </div>
    <button type="button" class="copy-btn" onclick={copyHtml}>{copyStatus || 'Copy HTML'}</button>
  </header>

  <div class="layout">
    <aside class="palette">
      <h2>Add block</h2>
      {#each BLOCK_TYPES as bt (bt.type)}
        <button type="button" onclick={() => addBlock(bt.type)}>+ {bt.label}</button>
      {/each}
    </aside>

    <section class="blocks">
      <h2>Blocks</h2>
      {#each blocks as block, i (block.id)}
        <BlockEditor
          {block}
          selected={selectedId === block.id}
          isFirst={i === 0}
          isLast={i === blocks.length - 1}
          onSelect={() => (selectedId = block.id)}
          onRemove={() => removeBlock(block.id)}
          onMoveUp={() => moveBlock(block.id, -1)}
          onMoveDown={() => moveBlock(block.id, 1)}
          onChange={(props) => updateProps(block.id, props)}
        />
      {/each}
      {#if blocks.length === 0}
        <p class="empty">No blocks yet — add one from the palette.</p>
      {/if}
    </section>

    <section class="preview">
      <h2>Live preview</h2>
      <Preview {html} width={previewWidth} />
    </section>
  </div>
</div>

<style>
  /* App chrome only — never leaks into the generated email HTML, which is
     rendered in an isolated sandboxed iframe (see Preview.svelte). */
  :global(body) {
    margin: 0;
    font-family:
      system-ui,
      -apple-system,
      sans-serif;
    background: #f9fafb;
  }
  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
    box-sizing: border-box;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 20px;
    background: #111827;
    color: white;
    flex-shrink: 0;
  }
  .toolbar h1 {
    font-size: 16px;
    margin: 0;
    flex: 1;
  }
  .width-toggle button {
    background: transparent;
    border: 1px solid #4b5563;
    color: white;
    padding: 6px 12px;
    cursor: pointer;
    font-size: 13px;
  }
  .width-toggle button.active {
    background: #2563eb;
    border-color: #2563eb;
  }
  .copy-btn {
    background: #2563eb;
    border: none;
    color: white;
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    min-width: 90px;
  }
  .layout {
    display: grid;
    grid-template-columns: 180px 320px 1fr;
    gap: 16px;
    flex: 1;
    overflow: hidden;
    padding: 16px;
    box-sizing: border-box;
  }
  .palette,
  .blocks {
    overflow-y: auto;
  }
  h2 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280;
    margin: 0 0 8px 0;
  }
  .palette button {
    display: block;
    width: 100%;
    text-align: left;
    padding: 8px 10px;
    margin-bottom: 6px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
  }
  .palette button:hover {
    border-color: #2563eb;
  }
  .preview {
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .empty {
    color: #6b7280;
    font-size: 13px;
  }
</style>

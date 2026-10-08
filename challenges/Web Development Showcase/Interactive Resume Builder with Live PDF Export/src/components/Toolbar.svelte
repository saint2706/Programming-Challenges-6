<script>
  /** App bar: title, page count, view switch on small screens, reset menu, import/export, print. */
  import ImportExport from './ImportExport.svelte';

  let { doc, pageCount, oversize = 0, view, onview, onprint, onimport, onerror, onreplace } = $props();

  let menu = $state();

  function replace(kind) {
    if (menu) menu.open = false;
    onreplace(kind);
  }
</script>

<header class="toolbar no-print">
  <h1 class="app-title">Resume Builder</h1>

  <div class="view-switch" role="group" aria-label="View">
    <button type="button" class="button" aria-pressed={view === 'edit'} onclick={() => onview('edit')}>Edit</button>
    <button type="button" class="button" aria-pressed={view === 'preview'} onclick={() => onview('preview')}>Preview</button>
  </div>

  <p class="page-count" role="status">
    {pageCount} {pageCount === 1 ? 'page' : 'pages'}
    {#if oversize > 0}<span class="warn">· {oversize} block{oversize === 1 ? '' : 's'} taller than a page</span>{/if}
  </p>

  <div class="toolbar-actions">
    <details class="menu" bind:this={menu}>
      <summary class="button">Start over</summary>
      <div class="menu-list">
        <button type="button" onclick={() => replace('seed')}>My resume</button>
        <button type="button" onclick={() => replace('long')}>Long example (3+ pages)</button>
        <button type="button" onclick={() => replace('blank')}>Blank</button>
      </div>
    </details>
    <ImportExport {doc} {onimport} {onerror} />
    <button type="button" class="button primary" onclick={onprint}>Export PDF</button>
  </div>
</header>

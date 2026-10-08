<script>
  /** The side panel: design controls, basics and the section list with its entry forms. */
  import BasicsForm from './BasicsForm.svelte';
  import SectionList from './SectionList.svelte';
  import StyleControls from './StyleControls.svelte';

  let { doc, apply, onfit, fitStatus, pageCount } = $props();
</script>

<div class="editor-panel">
  <details class="group" open>
    <summary>Design</summary>
    <StyleControls layout={doc['x-layout']} {apply} {onfit} {fitStatus} {pageCount} />
  </details>

  <details class="group" open>
    <summary>Basics</summary>
    <BasicsForm basics={doc.basics} {apply} />
  </details>

  <details class="group" open>
    <summary>Sections</summary>
    <p class="hint">Reorder with the handle (drag, or Alt plus Up or Down), hide what you do not need, and open a section to edit it. You can also type directly on the page.</p>
    <SectionList {doc} {apply} />
  </details>
</div>

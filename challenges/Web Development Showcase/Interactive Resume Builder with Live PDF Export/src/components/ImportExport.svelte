<script>
  /** Download the resume as JSON, or import one. Nothing is applied unless the whole file is valid. */
  import { MAX_IMPORT_BYTES, exportFileName, exportJson, importJson } from '../lib/io.js';

  let { doc, onimport, onerror } = $props();

  let fileInput = $state();

  function download() {
    const blob = new Blob([exportJson(doc)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = exportFileName(doc);
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function chosen(event) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    if (file.size > MAX_IMPORT_BYTES) {
      onerror(`That file is too large to import (the limit is ${MAX_IMPORT_BYTES / 1000} KB).`);
      return;
    }
    const result = importJson(await file.text());
    if (result.ok) onimport(result.doc, `Imported ${file.name}.`);
    else onerror(result.message);
  }
</script>

<button type="button" class="button" onclick={() => fileInput.click()}>Import JSON</button>
<button type="button" class="button" onclick={download}>Download JSON</button>
<input bind:this={fileInput} type="file" accept="application/json,.json" class="sr-only" tabindex="-1" aria-hidden="true" onchange={chosen} />

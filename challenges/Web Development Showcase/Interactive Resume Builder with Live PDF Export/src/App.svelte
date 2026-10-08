<script>
  /**
   * The app: owns the resume document, wires the editor panel, the live preview, autosave and
   * printing together. The document lives in `$state.raw` and is replaced wholesale by the pure
   * operations in lib/model.js, so every edit is a new value and nothing is mutated in place.
   */
  import { onMount } from 'svelte';
  import Announcer from './components/Announcer.svelte';
  import EditorPanel from './components/EditorPanel.svelte';
  import Preview from './components/Preview.svelte';
  import Toolbar from './components/Toolbar.svelte';
  import { announce } from './lib/announcer.svelte.js';
  import { fitToPages } from './lib/fit.js';
  import { MeasureUnavailable } from './lib/measure.js';
  import { setLayout, setPath } from './lib/model.js';
  import { createPrinter } from './lib/print.js';
  import { longResume } from './lib/sample-long.js';
  import { blankResume, seedResume } from './lib/seed.js';
  import { CORRUPT_KEY, createAutosaver, createStorage } from './lib/storage.js';

  const storage = createStorage();
  const loaded = storage.load();

  let doc = $state.raw(loaded.status === 'ok' ? loaded.doc : seedResume());
  let notice = $state(
    loaded.status === 'corrupt'
      ? {
          kind: 'error',
          text: `Your saved resume could not be read (${loaded.reason}), so the starting resume was loaded. The unreadable data was kept in this browser under "${CORRUPT_KEY}".`,
        }
      : storage.available
        ? null
        : {
            kind: 'info',
            text: 'Browser storage is unavailable, so your changes cannot be saved automatically. Use Download JSON to keep your work.',
          },
  );
  let view = $state('edit');
  let pageInfo = $state({ pageCount: 1, oversize: 0 });
  let fitStatus = $state('');
  let preview = $state();
  let undoDoc = $state.raw(null);

  /** Apply a pure change from lib/model.js. A refused change is announced, not thrown. */
  function apply(change) {
    try {
      doc = change(doc);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      announce(error.message);
    }
  }

  /** In-place typing on the page. A path that no longer exists (entry just removed) is ignored. */
  function edit(path, value) {
    try {
      doc = setPath(doc, path, value);
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
    }
  }

  /** Replace the whole resume, keeping a one-step undo. `keepLayout` keeps the current design. */
  function replaceWith(next, message, keepLayout) {
    undoDoc = doc;
    doc = keepLayout ? { ...next, 'x-layout': doc['x-layout'] } : next;
    notice = { kind: 'info', text: message, undo: true };
    fitStatus = '';
    announce(message);
  }

  function onreplace(kind) {
    const builders = {
      seed: [seedResume, 'Loaded the starting resume.'],
      long: [longResume, 'Loaded the long example resume.'],
      blank: [blankResume, 'Started a blank resume.'],
    };
    const [build, message] = builders[kind];
    replaceWith(build(), message, true);
  }

  function undo() {
    if (!undoDoc) return;
    doc = undoDoc;
    undoDoc = null;
    notice = null;
    announce('Restored your previous resume.');
  }

  function fit(pages) {
    let message;
    try {
      const result = fitToPages(pages, (scale) => preview.pageCountAt(scale));
      apply((d) => setLayout(d, { scale: result.scale }));
      const percent = Math.round(result.scale * 100);
      const word = pages === 1 ? 'page' : 'pages';
      message = result.fits
        ? `Fits in ${pages} ${word} at ${percent}% text size.`
        : `Cannot fit in ${pages} ${word} even at ${percent}%: it needs ${result.pageCount}.`;
    } catch (error) {
      if (error instanceof MeasureUnavailable) message = 'The preview cannot be measured right now. Try again in a moment.';
      else if (error instanceof RangeError) message = 'Enter a whole number of pages, 1 or more.';
      else throw error;
    }
    fitStatus = message;
    return message;
  }

  const saver = createAutosaver((d) => storage.save(d));
  $effect(() => {
    saver.schedule(doc);
  });

  const printer = createPrinter({
    flush: () => preview?.flush(),
    getName: () => doc.basics.name,
  });

  onMount(() => {
    const flushSave = () => saver.flush();
    window.addEventListener('pagehide', flushSave);
    const detach = printer.attach();
    return () => {
      window.removeEventListener('pagehide', flushSave);
      detach();
      saver.flush();
    };
  });
</script>

<div class="app" data-view={view}>
  <a class="skip-link no-print" href="#preview">Skip to the preview</a>

  <Toolbar
    {doc}
    pageCount={pageInfo.pageCount}
    oversize={pageInfo.oversize}
    {view}
    onview={(next) => (view = next)}
    onprint={() => printer.print()}
    onimport={(next, message) => replaceWith(next, message, false)}
    onerror={(text) => {
      notice = { kind: 'error', text };
      announce(text);
    }}
    {onreplace}
  />

  {#if notice}
    <div class="notice no-print {notice.kind}" role={notice.kind === 'error' ? 'alert' : 'status'}>
      <p>{notice.text}</p>
      {#if notice.undo && undoDoc}<button type="button" class="button small" onclick={undo}>Undo</button>{/if}
      <button type="button" class="icon-button" aria-label="Dismiss message" onclick={() => (notice = null)}>×</button>
    </div>
  {/if}

  <main class="workspace">
    <aside class="panel no-print" aria-label="Editor">
      <EditorPanel {doc} {apply} onfit={fit} {fitStatus} pageCount={pageInfo.pageCount} />
    </aside>
    <div class="stage" id="preview">
      <Preview
        bind:this={preview}
        {doc}
        onedit={edit}
        onpaginate={(result) => (pageInfo = { pageCount: result.pageCount, oversize: result.oversize.length })}
      />
    </div>
  </main>

  <Announcer />
</div>

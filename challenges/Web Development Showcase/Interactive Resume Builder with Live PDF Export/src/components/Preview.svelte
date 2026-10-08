<script>
  /**
   * The live, paginated preview. Pages are real DOM boxes (the same ones that get printed), so what
   * you see is what the PDF contains.
   *
   * The loop: document edit -> `buildRegions` -> measure only the blocks whose content changed (in a
   * hidden copy of the page, same width and fonts) -> `paginate` -> render pages. Measuring and
   * rendering are synchronous (`flushSync`), so `flush()` can make the preview current before a
   * print. Repagination is coalesced to one run per animation frame, and the caret is saved and
   * restored because a block that changes page is recreated by Svelte.
   */
  import { flushSync, onMount, untrack } from 'svelte';
  import { blockIndex, buildRegions } from '../lib/blocks.js';
  import { restoreCaret, saveCaret } from '../lib/caret.js';
  import { ensureFonts, fontVars, onFontsLoaded } from '../lib/fonts.js';
  import { MeasureUnavailable, createMeasurer, heightKey } from '../lib/measure.js';
  import { pageCss, pageMetrics } from '../lib/page.js';
  import { paginate } from '../lib/paginate.js';
  import { createFrameScheduler } from '../lib/scheduler.js';
  import '../styles/preview.css';
  import '../styles/templates.css';
  import Block from './Block.svelte';
  import Page from './Page.svelte';

  let { doc, onedit, onpaginate, editable = true } = $props();

  const PAGE_GAP_PX = 24;
  const VIEW_PADDING_PX = 32;

  const layout = $derived(doc['x-layout']);
  const regions = $derived(buildRegions(doc));
  const blocks = $derived(blockIndex(regions));
  const metrics = $derived(pageMetrics(layout.pageSize));
  const vars = $derived(fontVars(layout.fontPair));

  let fontEpoch = $state(0);
  let ready = $state(false);
  let painted = $state(false);
  let pagination = $state.raw({ pages: [], pageCount: 1, oversize: [] });

  // Blocks currently rendered in the hidden measuring tree, and the scale it is laid out at.
  let toMeasure = $state.raw([]);
  let measureScale = $state(null);
  let measureRoot = $state();
  let scaleInUse = null;

  const keyFor = (scale) =>
    `${layout.template}|${layout.pageSize}|${layout.fontPair}|${scale}|${fontEpoch}`;

  // True between `beforeprint` and `afterprint`. The browser re-resolves fonts when it switches to
  // print media, which would otherwise trigger a re-measure under print CSS mid-print.
  let printing = false;

  function measureInDom(needed) {
    // Heights read from a tree that is not laid out are all zero; never cache those.
    if (!measureRoot || getComputedStyle(measureRoot).display === 'none') throw new MeasureUnavailable();
    const regionOf = new Map(regions.flatMap((r) => r.blocks.map((b) => [b.id, r.id])));
    measureScale = scaleInUse ?? layout.scale;
    toMeasure = needed.map((block) => ({ region: regionOf.get(block.id), block }));
    flushSync();
    const results = needed.map((block) => {
      const el = measureRoot.querySelector(`[data-block="${block.id}"]`);
      if (!el) throw new Error(`block "${block.id}" was not rendered for measuring`);
      const height = el.getBoundingClientRect().height;
      if (!block.rows) return { height };
      const rowEls = [...el.querySelectorAll('[data-row]')];
      return { height, rows: rowEls.map((row) => row.getBoundingClientRect().height) };
    });
    toMeasure = [];
    measureScale = null;
    return results;
  }

  const measurer = createMeasurer({ measure: measureInDom });

  /** Measure (what is not cached) and paginate at `scale`, without touching what is displayed. */
  function paginateAt(scale) {
    scaleInUse = scale;
    try {
      return paginate(measurer.resolve(regions, keyFor(scale)), metrics.pagination);
    } finally {
      scaleInUse = null;
    }
  }

  const scheduler = createFrameScheduler(() => repaginate());

  /** `force` is for an explicit `flush()`, which must work even while printing. */
  function repaginate(force = false) {
    if (!ready || (printing && !force)) return;
    const caret = saveCaret();
    let result;
    try {
      result = paginateAt(layout.scale);
    } catch (error) {
      if (error instanceof MeasureUnavailable) return;
      throw error;
    }
    pagination = result;
    painted = true;
    flushSync();
    if (caret && document.activeElement?.getAttribute('data-path') !== caret.path) restoreCaret(caret);
    onpaginate?.(result);
  }

  /** Make the preview current right now (used before printing). */
  export function flush() {
    scheduler.cancel();
    repaginate(true);
  }

  /** How many pages the resume needs at `scale`; does not change the preview. */
  export function pageCountAt(scale) {
    return paginateAt(scale).pageCount;
  }

  // Edits, template/page-size/font/scale changes and finished font loads all schedule one pass.
  $effect(() => {
    regions;
    layout;
    ready;
    fontEpoch;
    untrack(() => scheduler.request());
  });

  // Wait for the fonts the resume uses before the first measurement; again if the pair changes.
  $effect(() => {
    const pair = layout.fontPair;
    const sample = untrack(() => JSON.stringify(doc));
    let cancelled = false;
    ensureFonts(pair, sample).then(() => {
      if (cancelled) return;
      ready = true;
      fontEpoch++;
    });
    return () => {
      cancelled = true;
    };
  });

  onMount(() => {
    const stop = onFontsLoaded(() => {
      fontEpoch++;
    });
    const startPrint = () => {
      printing = true;
    };
    const endPrint = () => {
      printing = false;
      scheduler.request();
    };
    window.addEventListener('beforeprint', startPrint);
    window.addEventListener('afterprint', endPrint);
    return () => {
      stop();
      window.removeEventListener('beforeprint', startPrint);
      window.removeEventListener('afterprint', endPrint);
      scheduler.cancel();
    };
  });

  // `?e2e` in the URL exposes a small read-only hook for the browser tests and the benchmark.
  onMount(() => {
    if (!new URLSearchParams(location.search).has('e2e')) return undefined;
    window.__resume = {
      /** What the paginator believed a block's height was (null if not measured at this layout). */
      measured: (id) => {
        const block = blocks.get(id);
        const entry = block && measurer.cache.get(heightKey(keyFor(layout.scale), block));
        return entry ? { height: entry.height, rows: entry.rows ?? null } : null;
      },
      /** Run one full repagination now; `cold` forgets every cached height first. Returns ms. */
      repaginate: ({ cold = false } = {}) => {
        if (cold) measurer.cache.clear();
        const started = performance.now();
        repaginate(true);
        return performance.now() - started;
      },
      stats: () => ({ ...measurer.stats, cached: measurer.cache.size }),
    };
    return () => {
      delete window.__resume;
    };
  });

  // The @page rule has to match the page size or the printed sheet would not.
  $effect(() => {
    let style = document.head.querySelector('style[data-page-rule]');
    if (!style) {
      style = document.createElement('style');
      style.setAttribute('data-page-rule', '');
      document.head.append(style);
    }
    style.textContent = pageCss(layout.pageSize);
  });
  onMount(() => () => document.head.querySelector('style[data-page-rule]')?.remove());

  // Fit the page stack to the width of the window on screen (printing ignores this).
  let viewWidth = $state(0);
  let scroller = $state();
  onMount(() => {
    if (typeof ResizeObserver !== 'function' || !scroller) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      viewWidth = entry.contentRect.width;
    });
    observer.observe(scroller);
    return () => observer.disconnect();
  });
  const viewScale = $derived(
    viewWidth > 0 ? Math.min(1.25, Math.max(0.3, (viewWidth - VIEW_PADDING_PX) / metrics.widthPx)) : 1,
  );
  const stackHeight = $derived(
    pagination.pageCount * metrics.heightPx + (pagination.pageCount - 1) * PAGE_GAP_PX,
  );

  const styleVars = $derived(
    `--accent:${layout.accent};--scale:${layout.scale};--font-heading:${vars['--font-heading']};--font-body:${vars['--font-body']};--page-w:${metrics.widthMm}mm;--page-h:${metrics.heightMm}mm;--page-gap:${PAGE_GAP_PX}px`,
  );
</script>

<div class="preview" style={styleVars}>
  <div class="measure" aria-hidden="true" bind:this={measureRoot}>
    <div
      class="page page-measure t-{layout.template} size-{layout.pageSize}"
      style:--scale={measureScale ?? layout.scale}
    >
      <div class="page-body">
        {#each regions as region (region.id)}
          <div class="region region-{region.id}">
            {#each toMeasure.filter((m) => m.region === region.id) as m (m.block.id)}
              <Block block={m.block} editable={false} />
            {/each}
          </div>
        {/each}
      </div>
    </div>
  </div>

  <!-- The scroll area must be focusable so keyboard users can scroll a tall preview. -->
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div
    class="preview-scroll"
    bind:this={scroller}
    role="region"
    aria-label="Resume preview"
    tabindex="0"
  >
    {#if painted}
      <div
        class="pages-frame"
        style:width="{metrics.widthPx * viewScale}px"
        style:height="{stackHeight * viewScale}px"
        data-ready="true"
        data-page-count={pagination.pageCount}
        data-oversize={pagination.oversize.length}
      >
        <div class="pages-scaler" style:transform="scale({viewScale})">
          {#each pagination.pages as page, i (i)}
            <Page
              {page}
              number={i + 1}
              total={pagination.pageCount}
              {blocks}
              {layout}
              {editable}
              {onedit}
            />
          {/each}
        </div>
      </div>
    {:else}
      <p class="preview-status" role="status">Preparing preview…</p>
    {/if}
  </div>
</div>

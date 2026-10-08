<script>
  /** Template, page size, font pair, accent colour, type size and "Fit to N pages". */
  import { announce } from '../lib/announcer.svelte.js';
  import { contrastRatio, meetsAA } from '../lib/contrast.js';
  import { FONT_PAIRS } from '../lib/fonts.js';
  import { setLayout } from '../lib/model.js';
  import { PAGE_SIZES, SCALE_MAX, SCALE_MIN } from '../lib/schema.js';
  import { TEMPLATES } from '../templates/index.js';

  let { layout, apply, onfit, fitStatus = '', pageCount = 1 } = $props();

  const uid = $props.id();
  const set = (patch) => apply((doc) => setLayout(doc, patch));
  const PAGE_LABEL = { a4: 'A4 (210 × 297 mm)', letter: 'US Letter (8.5 × 11 in)' };

  const ratio = $derived(contrastRatio(layout.accent, '#ffffff'));
  const lowContrast = $derived(!meetsAA(layout.accent));

  let target = $state(1);
  const sizePercent = $derived(Math.round(layout.scale * 100));

  function fit() {
    const message = onfit(Number(target));
    if (message) announce(message);
  }
</script>

<div class="style-controls">
  <fieldset class="choice-group">
    <legend>Template</legend>
    {#each Object.values(TEMPLATES) as template (template.id)}
      <label class="choice">
        <input type="radio" name="{uid}-template" value={template.id} checked={layout.template === template.id} onchange={() => set({ template: template.id })} />
        <span class="choice-name">{template.name}</span>
        <span class="choice-hint">{template.description}</span>
      </label>
    {/each}
  </fieldset>

  <fieldset class="choice-group">
    <legend>Page size</legend>
    {#each PAGE_SIZES as size (size)}
      <label class="choice inline">
        <input type="radio" name="{uid}-size" value={size} checked={layout.pageSize === size} onchange={() => set({ pageSize: size })} />
        <span class="choice-name">{PAGE_LABEL[size]}</span>
      </label>
    {/each}
  </fieldset>

  <div class="field">
    <label for="{uid}-font">Fonts</label>
    <select id="{uid}-font" value={layout.fontPair} onchange={(e) => set({ fontPair: e.currentTarget.value })}>
      {#each Object.values(FONT_PAIRS) as pair (pair.id)}
        <option value={pair.id}>{pair.name}</option>
      {/each}
    </select>
  </div>

  <div class="field">
    <label for="{uid}-accent">Accent colour</label>
    <div class="accent-row">
      <input id="{uid}-accent" type="color" value={layout.accent} oninput={(e) => set({ accent: e.currentTarget.value })} aria-describedby="{uid}-contrast" />
      <code>{layout.accent}</code>
      <span id="{uid}-contrast" class="contrast" class:low={lowContrast}>
        {ratio.toFixed(1)}:1 on white{lowContrast ? ' — too pale for small text' : ''}
      </span>
    </div>
  </div>

  <div class="field">
    <label for="{uid}-scale">Text size</label>
    <div class="range-row">
      <input id="{uid}-scale" type="range" min={SCALE_MIN} max={SCALE_MAX} step="0.01" value={layout.scale} oninput={(e) => set({ scale: Number(e.currentTarget.value) })} />
      <output for="{uid}-scale">{sizePercent}%</output>
    </div>
  </div>

  <div class="field fit">
    <label for="{uid}-fit">Fit to pages <span class="muted">(now {pageCount})</span></label>
    <div class="fit-row">
      <input id="{uid}-fit" type="number" min="1" max="20" step="1" bind:value={target} />
      <button type="button" class="button" onclick={fit}>Fit</button>
    </div>
    {#if fitStatus}<p class="hint" role="status">{fitStatus}</p>{/if}
  </div>
</div>

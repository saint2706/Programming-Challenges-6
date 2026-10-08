<script>
  /**
   * The ordered list of resume sections: reorder (drag, Alt+Up/Down on the handle, or the move
   * buttons), show or hide, and open a section to edit its entries. Every change is announced.
   */
  import { onMount, tick, untrack } from 'svelte';
  import { announce } from '../lib/announcer.svelte.js';
  import { enableSectionDnd, moveRelative } from '../lib/dnd.js';
  import { addEntry, moveSection, setLayout, setSectionHidden } from '../lib/model.js';
  import { LIMITS } from '../lib/schema.js';
  import { SECTION_META } from '../lib/sections.js';
  import EntryForm from './EntryForm.svelte';

  let { doc, apply } = $props();

  const order = $derived(doc['x-layout'].sectionOrder);
  const hidden = $derived(doc['x-layout'].hidden);
  let open = $state(null);
  let listEl = $state();

  const title = (id) => SECTION_META[id].title;

  async function shift(id, delta) {
    const from = order.indexOf(id);
    const to = from + delta;
    if (to < 0 || to >= order.length) return;
    apply((d) => moveSection(d, id, delta));
    announce(`${title(id)} moved ${delta < 0 ? 'up' : 'down'}. Position ${to + 1} of ${order.length}.`);
    await tick();
    listEl?.querySelector(`li[data-section="${id}"] .handle`)?.focus();
  }

  function onhandlekey(event, id) {
    if (!event.altKey || (event.key !== 'ArrowUp' && event.key !== 'ArrowDown')) return;
    event.preventDefault();
    shift(id, event.key === 'ArrowUp' ? -1 : 1);
  }

  function toggleHidden(id, show) {
    apply((d) => setSectionHidden(d, id, !show));
    announce(`${title(id)} ${show ? 'shown on' : 'hidden from'} the resume.`);
  }

  function dropped(id, targetId, edge) {
    const next = moveRelative(order, id, targetId, edge);
    if (next === order) return;
    apply((d) => setLayout(d, { sectionOrder: next }));
    announce(`${title(id)} moved. Position ${next.indexOf(id) + 1} of ${next.length}.`);
  }

  // Re-register pointer drag whenever the order changes (the list items are keyed and persist).
  $effect(() => {
    order;
    if (!listEl) return undefined;
    return untrack(() => enableSectionDnd(listEl, dropped));
  });

  function add(id) {
    apply((d) => addEntry(d, id));
    announce(`New ${SECTION_META[id].singular} added to ${title(id)}.`);
  }
</script>

<ol class="sections" bind:this={listEl} aria-label="Resume sections">
  {#each order as id, i (id)}
    <li data-section={id} class:is-hidden={hidden.includes(id)}>
      <div class="section-head">
        <button
          type="button"
          class="handle"
          aria-label="Reorder {title(id)}: drag, or press Alt plus Up or Down arrow"
          onkeydown={(event) => onhandlekey(event, id)}
        >
          <span aria-hidden="true">⠿</span>
        </button>
        <button
          type="button"
          class="section-title"
          aria-expanded={open === id}
          aria-controls="section-body-{id}"
          onclick={() => (open = open === id ? null : id)}
        >
          {title(id)}
          {#if id !== 'summary'}<span class="count">{doc[id].length}</span>{/if}
        </button>
        <label class="show-toggle">
          <input type="checkbox" checked={!hidden.includes(id)} onchange={(e) => toggleHidden(id, e.currentTarget.checked)} />
          <span>Show</span><span class="sr-only"> {title(id)} on the resume</span>
        </label>
        <button type="button" class="icon-button" disabled={i === 0} aria-label="Move {title(id)} up" onclick={() => shift(id, -1)}>↑</button>
        <button type="button" class="icon-button" disabled={i === order.length - 1} aria-label="Move {title(id)} down" onclick={() => shift(id, 1)}>↓</button>
      </div>

      {#if open === id}
        <div class="section-body" id="section-body-{id}">
          {#if id === 'summary'}
            <p class="hint">The summary is edited under Basics, or directly on the page.</p>
          {:else}
            {#each doc[id] as entry, index (index)}
              <EntryForm section={id} {index} {entry} count={doc[id].length} {apply} />
            {/each}
            <button type="button" class="button" disabled={doc[id].length >= LIMITS.entries} onclick={() => add(id)}>
              Add {SECTION_META[id].singular}
            </button>
          {/if}
        </div>
      {/if}
    </li>
  {/each}
</ol>

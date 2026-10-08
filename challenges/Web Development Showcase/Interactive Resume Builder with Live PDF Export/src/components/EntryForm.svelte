<script>
  /** The form for one entry (a job, a school, a skill group...), built from the section's field list. */
  import { announce } from '../lib/announcer.svelte.js';
  import {
    addBullet,
    moveBullet,
    moveEntry,
    removeBullet,
    removeEntry,
    setPath,
  } from '../lib/model.js';
  import { LIMITS } from '../lib/schema.js';
  import { SECTION_META, entryLabel } from '../lib/sections.js';
  import Field from './Field.svelte';
  import ListInput from './ListInput.svelte';

  let { section, index, entry, count, apply } = $props();

  const meta = $derived(SECTION_META[section]);
  const label = $derived(entryLabel(section, entry, index));

  function move(to, direction) {
    apply((doc) => moveEntry(doc, section, index, to));
    announce(`${label} moved ${direction}. Entry ${to + 1} of ${count}.`);
  }

  function remove() {
    apply((doc) => removeEntry(doc, section, index));
    announce(`${label} removed.`);
  }

  const set = (key, value) => apply((doc) => setPath(doc, [section, index, key], value));
</script>

<fieldset class="entry">
  <legend>{label}</legend>

  <div class="entry-actions">
    <button type="button" class="icon-button" disabled={index === 0} aria-label="Move {label} up" onclick={() => move(index - 1, 'up')}>↑</button>
    <button type="button" class="icon-button" disabled={index === count - 1} aria-label="Move {label} down" onclick={() => move(index + 1, 'down')}>↓</button>
    <button type="button" class="icon-button danger" aria-label="Remove {label}" onclick={remove}>Remove</button>
  </div>

  {#each meta.fields as field (field.key)}
    {#if field.type === 'bullets'}
      <div class="bullets-editor">
        <p class="field-group-label">{field.label}</p>
        {#each entry[field.key] as bullet, j (j)}
          <div class="bullet-row">
            <Field
              label="Bullet {j + 1}"
              type="textarea"
              rows={2}
              maxlength={LIMITS.long}
              value={bullet}
              oncommit={(text) => apply((doc) => setPath(doc, [section, index, field.key, j], text))}
            />
            <div class="bullet-actions">
              <button type="button" class="icon-button" disabled={j === 0} aria-label="Move bullet {j + 1} of {label} up" onclick={() => apply((doc) => moveBullet(doc, section, index, field.key, j, j - 1))}>↑</button>
              <button type="button" class="icon-button" disabled={j === entry[field.key].length - 1} aria-label="Move bullet {j + 1} of {label} down" onclick={() => apply((doc) => moveBullet(doc, section, index, field.key, j, j + 1))}>↓</button>
              <button type="button" class="icon-button danger" aria-label="Remove bullet {j + 1} of {label}" onclick={() => apply((doc) => removeBullet(doc, section, index, field.key, j))}>×</button>
            </div>
          </div>
        {/each}
        <button
          type="button"
          class="button small"
          disabled={entry[field.key].length >= LIMITS.bullets}
          onclick={() => apply((doc) => addBullet(doc, section, index, field.key))}
        >
          Add bullet
        </button>
      </div>
    {:else if field.type === 'csv'}
      <ListInput label={field.label} value={entry[field.key]} onchange={(list) => set(field.key, list)} />
    {:else}
      <Field
        label={field.label}
        type={field.type}
        rows={field.type === 'textarea' ? 3 : undefined}
        maxlength={field.type === 'textarea' ? LIMITS.long : LIMITS.short}
        value={entry[field.key]}
        oncommit={(text) => set(field.key, text)}
      />
    {/if}
  {/each}
</fieldset>

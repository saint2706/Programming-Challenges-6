<script>
  /**
   * Renders one block. With `editable` every text leaf is an `Editable`; without it the same
   * elements and classes are rendered as plain text, which is what the hidden measuring tree
   * uses, so measured heights match the visible page exactly. A test pins that the two modes
   * produce the same element structure.
   */
  import { fieldLabel } from '../lib/labels.js';
  import Editable from './Editable.svelte';

  let { block, editable = false, onedit, rowRange = null } = $props();

  const d = $derived(block.data);
  const rows = $derived(
    block.rows ? (rowRange ? block.rows.slice(rowRange[0], rowRange[1]) : block.rows) : [],
  );
</script>

{#snippet text(field, placeholder, options = {})}
  {#if editable}
    <Editable
      path={field.path}
      value={field.value}
      label={fieldLabel(field.path)}
      {placeholder}
      multiline={options.multiline}
      csv={options.csv}
      {onedit}
    />
  {:else}
    <span class="editable" class:multiline={options.multiline} data-placeholder={placeholder}
      >{options.csv ? field.value.join(', ') : field.value}</span
    >
  {/if}
{/snippet}

{#snippet link(item)}
  {#if item.href}
    <a href={item.href} target="_blank" rel="noopener noreferrer">{item.label}</a>
  {:else}
    <span>{item.label}</span>
  {/if}
{/snippet}

{#snippet contactList(contact)}
  <ul class="contact">
    {#if contact.location}<li>{contact.location}</li>{/if}
    {#if contact.email}<li>{@render link(contact.email)}</li>{/if}
    {#if contact.phone}<li>{@render link(contact.phone)}</li>{/if}
    {#if contact.website}<li>{@render link(contact.website)}</li>{/if}
    {#each contact.profiles as profile, i (i)}
      <li>{@render link(profile)}</li>
    {/each}
  </ul>
{/snippet}

{#snippet entryLink(item)}
  {#if item}
    <a class="link" href={item.href} target="_blank" rel="noopener noreferrer">{item.label}</a>
  {/if}
{/snippet}

{#if block.kind === 'header'}
  <header class="b b-header" data-block={block.id}>
    <h1 class="name">{@render text(d.name, 'Your name')}</h1>
    <p class="headline">{@render text(d.label, 'Headline')}</p>
    {#if d.contact}{@render contactList(d.contact)}{/if}
  </header>
{:else if block.kind === 'contact'}
  <div class="b b-contact" data-block={block.id}>{@render contactList(d)}</div>
{:else if block.kind === 'heading'}
  <h2 class="b b-heading" data-block={block.id}>{d.title}</h2>
{:else if block.kind === 'text'}
  <p class="b b-text" data-block={block.id}>{@render text(d, 'Write something', { multiline: true })}</p>
{:else if block.kind === 'work-header'}
  <div class="b b-entry b-work-header" data-block={block.id}>
    <div class="row">
      <span class="primary">{@render text(d.position, 'Position')}</span>
      <span class="dates">{d.dates}</span>
    </div>
    <div class="row secondary">
      <span class="org">{@render text(d.name, 'Company')}</span>
      {@render entryLink(d.link)}
    </div>
  </div>
{:else if block.kind === 'education-header'}
  <div class="b b-entry b-education-header" data-block={block.id}>
    <div class="row">
      <span class="primary">{@render text(d.institution, 'Institution')}</span>
      <span class="dates">{d.dates}</span>
    </div>
    <div class="row secondary">
      <span class="degree">
        {@render text(d.studyType, 'Degree')}{#if d.studyType.value && d.area.value}<span class="sep">,</span>{/if}
        {@render text(d.area, 'Field of study')}
      </span>
      {#if d.score.value}<span class="score">{@render text(d.score, 'Score')}</span>{/if}
      {@render entryLink(d.link)}
    </div>
  </div>
{:else if block.kind === 'project-header'}
  <div class="b b-entry b-project-header" data-block={block.id}>
    <div class="row">
      <span class="primary">{@render text(d.name, 'Project name')}</span>
      <span class="dates">{d.dates}</span>
    </div>
    {#if d.link}
      <div class="row secondary">{@render entryLink(d.link)}</div>
    {/if}
  </div>
{:else if block.kind === 'bullets'}
  <ul class="b b-bullets" data-block={block.id}>
    {#each rows as row (row.id)}
      <li data-row={row.id}>{@render text(row, 'Bullet')}</li>
    {/each}
  </ul>
{:else if block.kind === 'skill'}
  <div class="b b-skill" data-block={block.id}>
    <span class="skill-name">{@render text(d.name, 'Group')}</span>
    <span class="skill-keywords">{@render text(d.keywords, 'Skills, comma separated', { csv: true })}</span>
  </div>
{:else if block.kind === 'certificate'}
  <div class="b b-cert" data-block={block.id}>
    <span class="cert-name">{@render text(d.name, 'Certification')}</span>
    <span class="cert-meta">
      <span class="issuer">{@render text(d.issuer, 'Issuer')}</span>
      <span class="dates">{d.date}</span>
    </span>
  </div>
{:else if block.kind === 'language'}
  <div class="b b-lang" data-block={block.id}>
    <span class="lang-name">{@render text(d.language, 'Language')}</span>
    <span class="fluency">{@render text(d.fluency, 'Fluency')}</span>
  </div>
{:else if block.kind === 'award'}
  <div class="b b-entry b-award" data-block={block.id}>
    <div class="row">
      <span class="primary">{@render text(d.title, 'Award')}</span>
      <span class="dates">{d.date}</span>
    </div>
    <div class="row secondary">
      <span class="org">{@render text(d.awarder, 'Awarded by')}</span>
    </div>
  </div>
{/if}

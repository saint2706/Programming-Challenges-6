<script>
  /** One physical page: a fixed-size box holding the fragments the paginator assigned to it. */
  import Block from './Block.svelte';

  let { page, number, total, blocks, layout, editable = true, onedit } = $props();
</script>

<article
  class="page t-{layout.template} size-{layout.pageSize}"
  aria-label="Page {number} of {total}"
  data-page={number - 1}
>
  <div class="page-body">
    {#each Object.keys(page.regions) as regionId (regionId)}
      <div class="region region-{regionId}">
        {#each page.regions[regionId] as fragment (fragment.blockId)}
          {@const block = blocks.get(fragment.blockId)}
          {#if block}
            <Block
              {block}
              {editable}
              {onedit}
              rowRange={fragment.rowStart === null ? null : [fragment.rowStart, fragment.rowEnd]}
            />
          {/if}
        {/each}
      </div>
    {/each}
  </div>
</article>

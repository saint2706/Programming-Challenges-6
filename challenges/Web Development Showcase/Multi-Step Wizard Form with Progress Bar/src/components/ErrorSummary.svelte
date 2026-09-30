<script>
  let { errors, labels, el = $bindable() } = $props();

  const entries = $derived(Object.entries(errors));

  // Buttons, not href="#...": an anchor would rewrite location.hash and fight the step router.
  function focusField(name) {
    const target = document.getElementById(`f-${name}`) ?? document.querySelector(`[name="${name}"]`);
    target?.focus();
  }
</script>

{#if entries.length}
  <div class="summary" role="alert" tabindex="-1" bind:this={el}>
    <h3>{entries.length === 1 ? 'There is 1 problem' : `There are ${entries.length} problems`} with this step</h3>
    <ul>
      {#each entries as [name, message] (name)}
        <li><button type="button" class="link" onclick={() => focusField(name)}>{labels[name] ?? name}: {message}</button></li>
      {/each}
    </ul>
  </div>
{/if}

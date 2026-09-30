<script>
  // Label + control + hint + error wiring. The control is supplied as a snippet that receives
  // the ids/aria attributes it must spread, so every field gets identical a11y plumbing.
  let { name, label, error = '', hint = '', optional = false, children } = $props();

  const id = $derived(`f-${name}`);
  const hintId = $derived(`${id}-hint`);
  const errId = $derived(`${id}-err`);
  const describedby = $derived([hint ? hintId : '', error ? errId : ''].filter(Boolean).join(' ') || undefined);
</script>

<div class="field" class:invalid={!!error}>
  <label for={id}>{label}{#if optional}<span class="opt"> (optional)</span>{/if}</label>
  {@render children({ id, describedby, invalid: !!error })}
  {#if hint}<p class="hint" id={hintId}>{hint}</p>{/if}
  {#if error}<p class="error" id={errId}>{error}</p>{/if}
</div>

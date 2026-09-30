<script>
  import Field from './Field.svelte';

  let { w, name, label, type = 'text', hint = '', optional = false, ...rest } = $props();
</script>

<Field {name} {label} {hint} {optional} error={w.errors[name]}>
  {#snippet children({ id, describedby, invalid })}
    <input
      {id}
      {name}
      {type}
      value={w.data[name]}
      aria-describedby={describedby}
      aria-invalid={invalid ? 'true' : undefined}
      aria-required={optional ? undefined : 'true'}
      oninput={(e) => {
        w.data[name] = e.currentTarget.value;
        w.edit(name);
      }}
      onblur={() => w.touch(name)}
      {...rest}
    />
  {/snippet}
</Field>

<script>
  /**
   * A comma separated list in one text box (skills). It keeps the text you typed, so "JS," does
   * not lose its comma on the way through the model, and only resyncs when the list changes
   * from somewhere else (for example an in-place edit on the page).
   */
  import { untrack } from 'svelte';

  let { label, value, onchange } = $props();

  const uid = $props.id();
  const toList = (text) =>
    text
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  const same = (a, b) => a.length === b.length && a.every((item, i) => item === b[i]);

  let draft = $state(untrack(() => value.join(', ')));

  $effect(() => {
    const external = value;
    untrack(() => {
      if (!same(toList(draft), external)) draft = external.join(', ');
    });
  });

  function oninput(event) {
    draft = event.currentTarget.value;
    onchange(toList(draft));
  }
</script>

<div class="field">
  <label for="{uid}-input">{label}</label>
  <input id="{uid}-input" type="text" autocomplete="off" value={draft} {oninput} />
</div>

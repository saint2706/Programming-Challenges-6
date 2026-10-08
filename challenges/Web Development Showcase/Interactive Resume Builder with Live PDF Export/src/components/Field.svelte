<script>
  /**
   * A labelled form field for one string in the resume. Dates and links are validated as you type:
   * a valid value is committed immediately; an invalid one stays in the box (so you can fix it)
   * with an error message, and the resume keeps its last good value.
   */
  import { isSafeHref } from '../lib/links.js';

  let { label, value, type = 'text', maxlength = 500, rows = 3, placeholder = '', autocomplete = 'off', oncommit } = $props();

  const uid = $props.id();
  const DATE = /^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$/;

  function check(kind, text) {
    if (kind === 'date' && text !== '' && !DATE.test(text)) return 'Use YYYY, YYYY-MM or YYYY-MM-DD.';
    if (kind === 'url' && text !== '' && !isSafeHref(text)) return 'Use an http(s), mailto or tel link.';
    if (kind === 'email' && text !== '' && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text)) return 'Use a valid email address.';
    if (kind === 'tel' && text !== '' && !/^[0-9\s+\-().]*$/.test(text)) return 'Use digits, spaces, +, -, (), or .';
    return '';
  }

  // The text being typed while it is not yet valid; null when the box mirrors the resume.
  let draft = $state(null);
  const problem = $derived(draft === null ? '' : check(type, draft));
  const shown = $derived(draft ?? value);

  function oninput(event) {
    const text = event.currentTarget.value;
    if (check(type, text)) {
      draft = text;
      return;
    }
    draft = null;
    oncommit(text);
  }
</script>

<div class="field">
  <label for="{uid}-input">{label}</label>
  {#if type === 'textarea'}
    <textarea
      id="{uid}-input"
      {rows}
      {maxlength}
      {placeholder}
      value={shown}
      {oninput}
      aria-invalid={problem ? 'true' : undefined}
      aria-describedby={problem ? `${uid}-error` : undefined}
    ></textarea>
  {:else}
    <input
      id="{uid}-input"
      type={type === 'date' || type === 'url' ? 'text' : type}
      inputmode={type === 'date' ? 'numeric' : undefined}
      {autocomplete}
      {maxlength}
      placeholder={type === 'date' ? 'YYYY-MM' : placeholder}
      value={shown}
      {oninput}
      aria-invalid={problem ? 'true' : undefined}
      aria-describedby={problem ? `${uid}-error` : undefined}
    />
  {/if}
  {#if problem}<p class="field-error" id="{uid}-error" role="alert">{problem}</p>{/if}
</div>

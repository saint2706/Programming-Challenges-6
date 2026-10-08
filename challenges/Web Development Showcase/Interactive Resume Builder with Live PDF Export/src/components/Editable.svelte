<script>
  /**
   * One piece of text you can type into on the page. It writes straight to the resume (through
   * `onedit(path, value)`) and never re-renders its own text while you type: the DOM is only
   * touched when the model's text differs from what is already shown, because replacing a text
   * node resets the caret.
   */

  let { path, value, label, multiline = false, placeholder = '', csv = false, onedit } = $props();

  // `plaintext-only` stops the browser inserting markup; fall back to plain `true` where it is
  // not supported (paste is intercepted either way, so no markup can sneak in).
  const plainTextOnly = (() => {
    try {
      const probe = document.createElement('div');
      probe.contentEditable = 'plaintext-only';
      return probe.contentEditable === 'plaintext-only';
    } catch {
      return false;
    }
  })();

  let el = $state();
  const text = $derived(csv ? value.join(', ') : value);

  const toList = (raw) =>
    raw
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

  $effect(() => {
    if (!el || el.textContent === text) return;
    // While you type "JS, " the model holds ["JS"]; rewriting the field to "JS" would eat the
    // comma and move the caret. Leave a focused list field alone if it already means the same list.
    if (csv && document.activeElement === el) {
      const typed = toList(el.textContent ?? '');
      if (typed.length === value.length && typed.every((item, i) => item === value[i])) return;
    }
    el.textContent = text;
  });

  function commit() {
    const raw = el.textContent ?? '';
    onedit?.(path, csv ? toList(raw) : raw);
  }

  function onkeydown(event) {
    if (event.key === 'Enter' && !multiline) event.preventDefault();
  }

  function insert(text) {
    // execCommand keeps the browser's undo stack and fires `input` itself.
    if (typeof document.execCommand === 'function' && document.execCommand('insertText', false, text)) return;
    const selection = getSelection();
    if (!selection || selection.rangeCount === 0) return;
    const range = selection.getRangeAt(0);
    range.deleteContents();
    const node = document.createTextNode(text);
    range.insertNode(node);
    range.setStartAfter(node);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
    el.normalize();
    commit();
  }

  function onpaste(event) {
    event.preventDefault();
    const pasted = (event.clipboardData?.getData('text/plain') ?? '').replace(/\r\n?/g, '\n');
    insert(multiline ? pasted : pasted.replace(/\s*\n\s*/g, ' '));
  }
</script>

<!-- A contenteditable element is natively focusable, so it needs no tabindex. -->
<!-- svelte-ignore a11y_interactive_supports_focus -->
<span
  bind:this={el}
  class="editable"
  class:multiline
  contenteditable={plainTextOnly ? 'plaintext-only' : 'true'}
  role="textbox"
  aria-label={label}
  aria-multiline={multiline}
  spellcheck="true"
  data-path={path}
  data-placeholder={placeholder}
  oninput={commit}
  {onkeydown}
  {onpaste}
></span>

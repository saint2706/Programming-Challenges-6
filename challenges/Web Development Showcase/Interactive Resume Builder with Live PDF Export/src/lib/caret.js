/**
 * Keeps the caret in the field you are typing in when repagination moves that field to another
 * page. Svelte destroys and recreates a block's DOM when it changes page, which drops focus, so
 * the app saves `{ path, offset }` before the update and restores it after. Fields are found by
 * their `data-path`, never by DOM identity.
 */

const SELECTOR = '[data-path][contenteditable]';

const editableOf = (node) => (node?.nodeType === 1 ? node : node?.parentElement)?.closest(SELECTOR) ?? null;

/** `{ path, offset }` for a collapsed caret inside an editable field, otherwise null. */
export function saveCaret() {
  const selection = globalThis.getSelection?.();
  if (!selection || selection.rangeCount === 0 || !selection.isCollapsed) return null;
  const field = editableOf(selection.anchorNode);
  if (!field) return null;
  const range = document.createRange();
  range.selectNodeContents(field);
  range.setEnd(selection.anchorNode, selection.anchorOffset);
  return { path: field.dataset.path, offset: range.toString().length };
}

function placeCaret(field, offset) {
  const range = document.createRange();
  const walker = document.createTreeWalker(field, NodeFilter.SHOW_TEXT);
  let remaining = offset;
  let last = null;
  let placed = false;
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    last = node;
    if (remaining <= node.length) {
      range.setStart(node, remaining);
      placed = true;
      break;
    }
    remaining -= node.length;
  }
  if (!placed) {
    if (last) range.setStart(last, last.length);
    else range.setStart(field, 0);
  }
  range.collapse(true);
  const selection = globalThis.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

/** Focus the field with `saved.path` and put the caret at `saved.offset`. False if it is gone. */
export function restoreCaret(saved) {
  if (!saved) return false;
  const field = [...document.querySelectorAll(SELECTOR)].find((el) => el.dataset.path === saved.path);
  if (!field) return false;
  field.focus({ preventScroll: true });
  placeCaret(field, saved.offset);
  return true;
}

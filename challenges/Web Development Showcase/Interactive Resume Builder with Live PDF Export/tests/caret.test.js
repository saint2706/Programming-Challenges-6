import { afterEach, describe, expect, it } from 'vitest';
import { restoreCaret, saveCaret } from '../src/lib/caret.js';

/** Replace the page with markup from the (static, test-owned) string. */
function mount(html) {
  const parsed = new DOMParser().parseFromString(html, 'text/html');
  document.body.replaceChildren(...parsed.body.childNodes);
}

function placeCaret(node, offset) {
  const range = document.createRange();
  range.setStart(node, offset);
  range.collapse(true);
  const selection = getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}

afterEach(() => {
  document.body.replaceChildren();
  getSelection().removeAllRanges();
});

describe('saveCaret', () => {
  it('records the field path and the character offset', () => {
    mount('<div><span contenteditable="true" data-path="work.0.position">Staff Engineer</span></div>');
    placeCaret(document.querySelector('span').firstChild, 5);
    expect(saveCaret()).toEqual({ path: 'work.0.position', offset: 5 });
  });

  it('counts characters across several text nodes', () => {
    mount('<span contenteditable="true" data-path="a">Hello <b>big</b> world</span>');
    placeCaret(document.querySelector('span').lastChild, 3);
    expect(saveCaret()).toEqual({ path: 'a', offset: 'Hello big wo'.length });
  });

  it('returns null with no selection, outside a field, or for a range selection', () => {
    mount('<p>plain</p><span contenteditable="true" data-path="a">text</span>');
    expect(saveCaret()).toBeNull();
    placeCaret(document.querySelector('p').firstChild, 2);
    expect(saveCaret()).toBeNull();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('span'));
    getSelection().removeAllRanges();
    getSelection().addRange(range);
    expect(saveCaret()).toBeNull();
  });
});

describe('restoreCaret', () => {
  it('puts the caret back in a re-created field at the same offset', () => {
    mount('<span contenteditable="true" data-path="work.0.position">Staff Engineer</span>');
    placeCaret(document.querySelector('span').firstChild, 5);
    const saved = saveCaret();

    mount('<div><span contenteditable="true" tabindex="0" data-path="work.0.position">Staff Engineer</span></div>');
    expect(restoreCaret(saved)).toBe(true);
    const selection = getSelection();
    expect(selection.isCollapsed).toBe(true);
    expect(selection.anchorNode.parentElement.dataset.path).toBe('work.0.position');
    expect(selection.anchorOffset).toBe(5);
    expect(document.activeElement.dataset.path).toBe('work.0.position');
  });

  it('clamps to the end when the text got shorter', () => {
    mount('<span contenteditable="true" tabindex="0" data-path="a">abc</span>');
    expect(restoreCaret({ path: 'a', offset: 99 })).toBe(true);
    expect(getSelection().anchorOffset).toBe(3);
  });

  it('handles an empty field', () => {
    mount('<span contenteditable="true" tabindex="0" data-path="a"></span>');
    expect(restoreCaret({ path: 'a', offset: 4 })).toBe(true);
    expect(getSelection().anchorOffset).toBe(0);
  });

  it('returns false when the field is gone or nothing was saved', () => {
    mount('<span contenteditable="true" data-path="a">x</span>');
    expect(restoreCaret({ path: 'missing', offset: 0 })).toBe(false);
    expect(restoreCaret(null)).toBe(false);
  });

  it('does not match a path that merely contains the saved one', () => {
    mount('<span contenteditable="true" tabindex="0" data-path="work.10.name">x</span>');
    expect(restoreCaret({ path: 'work.1.name', offset: 0 })).toBe(false);
  });
});

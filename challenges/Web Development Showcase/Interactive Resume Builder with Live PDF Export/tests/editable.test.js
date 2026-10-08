import { fireEvent, render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Editable from '../src/components/Editable.svelte';

function setup(props = {}) {
  const onedit = vi.fn();
  const utils = render(Editable, {
    props: {
      path: 'work.0.position',
      value: 'Engineer',
      label: 'Position, Experience entry 1',
      onedit,
      ...props,
    },
  });
  return { onedit, box: screen.getByRole('textbox'), ...utils };
}

function caretAtEnd(el) {
  const range = document.createRange();
  range.selectNodeContents(el);
  range.collapse(false);
  getSelection().removeAllRanges();
  getSelection().addRange(range);
}

describe('Editable', () => {
  it('is a labelled textbox carrying its document path and text', () => {
    const { box } = setup();
    expect(box).toHaveAccessibleName('Position, Experience entry 1');
    expect(box).toHaveAttribute('data-path', 'work.0.position');
    expect(box).toHaveAttribute('contenteditable');
    expect(box).toHaveTextContent('Engineer');
    expect(box).toHaveAttribute('aria-multiline', 'false');
  });

  it('reports typed text with its path', async () => {
    const { box, onedit } = setup();
    box.textContent = 'Staff Engineer';
    await fireEvent.input(box);
    expect(onedit).toHaveBeenCalledWith('work.0.position', 'Staff Engineer');
  });

  it('does not allow Enter in a single-line field, but does in a multi-line one', async () => {
    const single = setup();
    expect(await fireEvent.keyDown(single.box, { key: 'Enter' })).toBe(false);
    single.unmount();
    const multi = setup({ multiline: true, path: 'basics.summary' });
    expect(multi.box).toHaveAttribute('aria-multiline', 'true');
    expect(await fireEvent.keyDown(multi.box, { key: 'Enter' })).toBe(true);
  });

  it('pastes plain text only, flattening line breaks in a single-line field', async () => {
    const { box, onedit } = setup();
    caretAtEnd(box);
    await fireEvent.paste(box, { clipboardData: { getData: () => ' and\n  more' } });
    expect(box.textContent).toBe('Engineer and more');
    expect(onedit).toHaveBeenLastCalledWith('work.0.position', 'Engineer and more');
    expect(box.querySelector('*')).toBeNull();
  });

  it('keeps line breaks when pasting into a multi-line field', async () => {
    const { box, onedit } = setup({ multiline: true, value: 'One', path: 'basics.summary' });
    caretAtEnd(box);
    await fireEvent.paste(box, { clipboardData: { getData: () => '\nTwo' } });
    expect(onedit).toHaveBeenLastCalledWith('basics.summary', 'One\nTwo');
  });

  it('edits a list as comma separated text', async () => {
    const { box, onedit } = setup({ csv: true, value: ['JS', 'TS'], path: 'skills.0.keywords' });
    expect(box).toHaveTextContent('JS, TS');
    box.textContent = 'JS, TS ,, Go ,';
    await fireEvent.input(box);
    expect(onedit).toHaveBeenCalledWith('skills.0.keywords', ['JS', 'TS', 'Go']);
  });

  it('does not rewrite a list field you are typing in just to normalise its spacing', async () => {
    const { box, rerender } = setup({ csv: true, value: ['JS'], path: 'skills.0.keywords' });
    box.focus();
    // typed "JS,G" with no space: the model now holds ["JS", "G"], whose text would be "JS, G"
    box.textContent = 'JS,G';
    await rerender({ value: ['JS', 'G'] });
    expect(box.textContent).toBe('JS,G');
    // but a genuinely different list pushed from outside still wins
    await rerender({ value: ['JS', 'Go'] });
    expect(box.textContent).toBe('JS, Go');
  });

  it('shows a new value pushed from outside', async () => {
    const { box, rerender } = setup();
    await rerender({ value: 'Architect' });
    expect(box).toHaveTextContent('Architect');
  });

  it('leaves the DOM alone when the text already matches, so the caret does not move', async () => {
    const { box, rerender } = setup();
    const node = box.firstChild;
    box.textContent = 'Engineer!';
    const typed = box.firstChild;
    await rerender({ value: 'Engineer!' });
    expect(box.firstChild).toBe(typed);
    expect(node).not.toBe(typed);
  });

  it('carries its placeholder for the empty state', () => {
    const { box } = setup({ value: '', placeholder: 'Job title' });
    expect(box).toHaveAttribute('data-placeholder', 'Job title');
    expect(box).toHaveTextContent('');
  });

  it('treats markup in the text as text', async () => {
    const { box } = setup({ value: '<img src=x onerror=alert(1)>' });
    expect(box.querySelector('img')).toBeNull();
    expect(box.textContent).toBe('<img src=x onerror=alert(1)>');
  });
});

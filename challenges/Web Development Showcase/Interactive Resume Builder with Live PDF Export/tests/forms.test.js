import { fireEvent, render, screen } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import EntryForm from '../src/components/EntryForm.svelte';
import Field from '../src/components/Field.svelte';
import ListInput from '../src/components/ListInput.svelte';
import { announcer } from '../src/lib/announcer.svelte.js';
import { LIMITS } from '../src/lib/schema.js';
import { longResume } from '../src/lib/sample-long.js';

describe('Field', () => {
  it('commits valid text immediately', async () => {
    const oncommit = vi.fn();
    render(Field, { props: { label: 'Company', value: 'Acme', oncommit } });
    await userEvent.setup().type(screen.getByLabelText('Company'), '!');
    expect(oncommit).toHaveBeenLastCalledWith('Acme!');
  });

  it('keeps an invalid date in the box with an error, and commits nothing', async () => {
    const oncommit = vi.fn();
    render(Field, { props: { label: 'Start', type: 'date', value: '2020-01', oncommit } });
    const input = screen.getByLabelText('Start');
    await fireEvent.input(input, { target: { value: '2020-1' } });
    expect(oncommit).not.toHaveBeenCalled();
    expect(input).toHaveValue('2020-1');
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent('Use YYYY, YYYY-MM or YYYY-MM-DD.');
    expect(input).toHaveAccessibleDescription('Use YYYY, YYYY-MM or YYYY-MM-DD.');

    await fireEvent.input(input, { target: { value: '2020-10' } });
    expect(oncommit).toHaveBeenCalledWith('2020-10');
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('refuses an unsafe link but accepts an empty one', async () => {
    const oncommit = vi.fn();
    render(Field, { props: { label: 'Link', type: 'url', value: '', oncommit } });
    const input = screen.getByLabelText('Link');
    await fireEvent.input(input, { target: { value: 'javascript:alert(1)' } });
    expect(oncommit).not.toHaveBeenCalled();
    expect(screen.getByRole('alert')).toBeInTheDocument();
    await fireEvent.input(input, { target: { value: '' } });
    expect(oncommit).toHaveBeenCalledWith('');
  });
});

describe('ListInput', () => {
  it('keeps the comma you just typed and reports the parsed list', async () => {
    const onchange = vi.fn();
    render(ListInput, { props: { label: 'Skills', value: ['JS'], onchange } });
    const input = screen.getByLabelText('Skills');
    await fireEvent.input(input, { target: { value: 'JS,' } });
    expect(onchange).toHaveBeenLastCalledWith(['JS']);
    expect(input).toHaveValue('JS,');
    await fireEvent.input(input, { target: { value: 'JS, Go ,, Rust' } });
    expect(onchange).toHaveBeenLastCalledWith(['JS', 'Go', 'Rust']);
  });

  it('follows a list that changed somewhere else', async () => {
    const { rerender } = render(ListInput, { props: { label: 'Skills', value: ['JS'], onchange: () => {} } });
    await rerender({ value: ['JS', 'Go'] });
    expect(screen.getByLabelText('Skills')).toHaveValue('JS, Go');
  });
});

describe('EntryForm', () => {
  function setup(overrides = {}) {
    const doc = longResume();
    const apply = vi.fn((change) => change(doc));
    const props = { section: 'work', index: 1, entry: doc.work[1], count: doc.work.length, apply, ...overrides };
    render(EntryForm, { props });
    return { doc, apply, props };
  }

  it('is a group named after the entry, with a field per schema field', () => {
    setup();
    expect(screen.getByRole('group', { name: 'Senior Software Engineer, Harbor & Pine Software' })).toBeInTheDocument();
    for (const label of ['Position', 'Company', 'Link', 'Start', 'End (blank means present)', 'Summary']) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }
    expect(screen.getAllByLabelText(/^Bullet \d$/)).toHaveLength(5);
  });

  it('edits a field through apply with the pure model operation', async () => {
    const { apply, doc } = setup();
    await userEvent.setup().type(screen.getByLabelText('Position'), '!');
    const next = apply.mock.results.at(-1).value;
    expect(next.work[1].position).toBe('Senior Software Engineer!');
    expect(doc.work[1].position).toBe('Senior Software Engineer');
  });

  it('moves, removes and announces', async () => {
    const { apply } = setup();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Move Senior Software Engineer, Harbor & Pine Software up' }));
    expect(apply.mock.results.at(-1).value.work[0].position).toBe('Senior Software Engineer');
    await Promise.resolve();
    expect(announcer.message).toMatch(/moved up\. Entry 1 of 6\./);

    await user.click(screen.getByRole('button', { name: 'Remove Senior Software Engineer, Harbor & Pine Software' }));
    expect(apply.mock.results.at(-1).value.work).toHaveLength(5);
  });

  it('disables the move buttons at the ends', () => {
    setup({ index: 0, entry: longResume().work[0] });
    expect(screen.getByRole('button', { name: 'Move Staff Software Engineer, Brightwave Systems up' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Move Staff Software Engineer, Brightwave Systems down' })).toBeEnabled();
  });

  it('adds, moves and removes bullets, and stops at the limit', async () => {
    const { apply } = setup();
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Add bullet' }));
    expect(apply.mock.results.at(-1).value.work[1].highlights).toHaveLength(6);
    await user.click(screen.getByRole('button', { name: /Move bullet 2 of .* up/ }));
    expect(apply.mock.results.at(-1).value.work[1].highlights[0]).toBe(longResume().work[1].highlights[1]);
    await user.click(screen.getByRole('button', { name: /Remove bullet 1 of/ }));
    expect(apply.mock.results.at(-1).value.work[1].highlights).toHaveLength(4);
  });

  it('disables Add bullet at the bullet limit', () => {
    const doc = longResume();
    const entry = { ...doc.work[0], highlights: Array(LIMITS.bullets).fill('x') };
    render(EntryForm, { props: { section: 'work', index: 0, entry, count: 1, apply: () => {} } });
    expect(screen.getByRole('button', { name: 'Add bullet' })).toBeDisabled();
  });

  it('edits skills as a comma separated list', async () => {
    const doc = longResume();
    const apply = vi.fn((change) => change(doc));
    render(EntryForm, { props: { section: 'skills', index: 0, entry: doc.skills[0], count: 5, apply } });
    await fireEvent.input(screen.getByLabelText('Skills (comma separated)'), { target: { value: 'Go, Rust' } });
    expect(apply.mock.results.at(-1).value.skills[0].keywords).toEqual(['Go', 'Rust']);
  });
});

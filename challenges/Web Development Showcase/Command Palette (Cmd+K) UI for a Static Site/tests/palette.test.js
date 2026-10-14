import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createPalette, urlPath } from '../src/palette.js';
import { createRecents } from '../src/recent.js';
import { safeStorage } from '../src/storage.js';

const ITEMS = [
    { id: 'p:start', type: 'page', title: 'Getting started', hint: 'Set it up', url: 'start.html' },
    { id: 'p:keys', type: 'page', title: 'Keyboard shortcuts', url: 'keys.html' },
    { id: 'h:keys#moving', type: 'heading', title: 'Moving', hint: 'Keyboard shortcuts', url: 'keys.html#moving' },
    { id: 'h:keys#closing', type: 'heading', title: 'Closing', hint: 'Keyboard shortcuts', url: 'keys.html#closing' },
];

let run;
let setup;

function make(options = {}) {
    run = vi.fn();
    const navigate = vi.fn();
    const openTab = vi.fn();
    const store = new Map();
    const recents = createRecents(
        safeStorage(() => ({
            getItem: (k) => store.get(k) ?? null,
            setItem: (k, v) => store.set(k, v),
            removeItem: (k) => store.delete(k),
        })),
    );
    const palette = createPalette({
        loadItems: () => Promise.resolve(ITEMS),
        actions: [{ id: 'a:top', type: 'action', title: 'Scroll to top', run }],
        context: { tag: 'ctx' },
        recents,
        navigate,
        openTab,
        currentPage: 'keys.html',
        ...options,
    });
    setup = { palette, navigate, openTab, recents };
    return setup;
}

const input = () => document.querySelector('.cp-input');
const options = () => [...document.querySelectorAll('[role="option"]')];
const titles = () => options().map((o) => o.querySelector('.cp-title').textContent);
const active = () => document.getElementById(input().getAttribute('aria-activedescendant'));

function type(text) {
    input().value = text;
    input().dispatchEvent(new Event('input', { bubbles: true }));
}

function press(key, init = {}) {
    const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
    input().dispatchEvent(event);
    return event;
}

/** Open and wait for the index to load. */
async function openLoaded() {
    setup.palette.open();
    await vi.waitFor(() => expect(options().length).toBeGreaterThan(2));
}

beforeEach(() => {
    vi.useRealTimers();
    document.body.replaceChildren();
    make();
});
afterEach(() => document.body.replaceChildren());

describe('structure and ARIA', () => {
    it('is a dialog holding a combobox that controls a listbox', async () => {
        await openLoaded();
        const field = input();
        expect(document.querySelector('dialog.cp').open).toBe(true);
        expect(field.getAttribute('role')).toBe('combobox');
        expect(field.getAttribute('aria-controls')).toBe('cp-list');
        expect(field.getAttribute('aria-autocomplete')).toBe('list');
        expect(field.getAttribute('aria-expanded')).toBe('true');
        expect(document.getElementById('cp-list').getAttribute('role')).toBe('listbox');
        expect(document.activeElement).toBe(field);
    });

    it('groups the options and labels each group', async () => {
        await openLoaded();
        for (const group of document.querySelectorAll('[role="group"]')) {
            const label = document.getElementById(group.getAttribute('aria-labelledby'));
            expect(label.textContent).toMatch(/Actions|Pages/u);
        }
        // empty query: actions and pages, never headings
        expect(titles()).toEqual(['Scroll to top', 'Getting started', 'Keyboard shortcuts']);
    });

    it('keeps aria-activedescendant on the one selected option', async () => {
        await openLoaded();
        expect(options().filter((o) => o.getAttribute('aria-selected') === 'true')).toHaveLength(1);
        expect(active()).toBe(options()[0]);
        press('ArrowDown');
        expect(active()).toBe(options()[1]);
        expect(options()[0].getAttribute('aria-selected')).toBe('false');
    });

    it('marks the page you are on', async () => {
        await openLoaded();
        const here = options().find((o) => o.dataset.id === 'p:keys');
        expect(here.textContent).toContain('You are here');
    });
});

describe('searching', () => {
    it('re-renders on input and highlights matches with spans, not markup', async () => {
        await openLoaded();
        type('gs');
        expect(titles()).toEqual(['Getting started']);
        const marks = [...options()[0].querySelectorAll('.cp-m')].map((m) => m.textContent);
        expect(marks).toEqual(['G', 's']);
    });

    it('puts headings in their own group with the page as the hint', async () => {
        await openLoaded();
        type('moving');
        const option = options()[0];
        expect(option.querySelector('.cp-hint').textContent).toBe('Keyboard shortcuts');
        expect(document.getElementById('cp-group-heading').textContent).toBe('Headings');
    });

    it('shows an empty state, drops aria-activedescendant and says so', async () => {
        await openLoaded();
        type('zzzq');
        expect(options()).toHaveLength(0);
        expect(document.querySelector('.cp-empty').hidden).toBe(false);
        expect(document.querySelector('.cp-empty').textContent).toContain('No results for “zzzq”');
        expect(input().hasAttribute('aria-activedescendant')).toBe(false);
        expect(input().getAttribute('aria-expanded')).toBe('false');
        await vi.waitFor(() => expect(document.querySelector('.cp-status').textContent).toBe('No results'));
    });

    it('announces the result count after a short pause', async () => {
        await openLoaded();
        type('k');
        await vi.waitFor(() => expect(document.querySelector('.cp-status').textContent).toMatch(/^\d+ results?$/u));
    });

    it('renders hostile titles as text', async () => {
        const items = [{ id: 'x', type: 'page', title: '<img src=x onerror=alert(1)> & "quotes"', url: 'x.html' }];
        document.body.replaceChildren();
        make({ loadItems: () => Promise.resolve(items) });
        setup.palette.open();
        await vi.waitFor(() => expect(options().length).toBeGreaterThan(1));
        type('img');
        expect(document.querySelector('.cp-list img')).toBeNull();
        expect(titles()).toEqual(['<img src=x onerror=alert(1)> & "quotes"']);
    });

    it('still works, with actions only, when the index fails to load', async () => {
        document.body.replaceChildren();
        make({ loadItems: () => Promise.reject(new Error('offline')) });
        setup.palette.open();
        await vi.waitFor(() => expect(options()).toHaveLength(1));
        type('zzzq');
        await vi.waitFor(() => expect(document.querySelector('.cp-empty').textContent).toContain('index could not be loaded'));
    });
});

describe('keyboard', () => {
    it('wraps with the arrows, and moves with Ctrl+N/P and Ctrl+J/K', async () => {
        await openLoaded();
        press('ArrowUp');
        expect(active()).toBe(options().at(-1));
        press('ArrowDown');
        expect(active()).toBe(options()[0]);
        press('n', { ctrlKey: true });
        expect(active()).toBe(options()[1]);
        press('p', { ctrlKey: true });
        expect(active()).toBe(options()[0]);
        press('j', { ctrlKey: true });
        press('k', { ctrlKey: true });
        expect(active()).toBe(options()[0]);
    });

    it('moves a page at a time and to the ends', async () => {
        await openLoaded();
        press('PageDown');
        expect(active()).toBe(options().at(-1));
        press('PageUp');
        expect(active()).toBe(options()[0]);
        press('End');
        expect(active()).toBe(options().at(-1));
        press('Home');
        expect(active()).toBe(options()[0]);
    });

    it('leaves Home and End to the caret while there is text', async () => {
        await openLoaded();
        type('k');
        expect(press('End').defaultPrevented).toBe(false);
        expect(press('End', { ctrlKey: true }).defaultPrevented).toBe(true);
    });

    it('does not take the letters n, j, p, k without Ctrl', async () => {
        await openLoaded();
        for (const key of ['n', 'j', 'p', 'k']) expect(press(key).defaultPrevented).toBe(false);
    });

    it('Enter opens the active page, closes the palette and remembers it', async () => {
        await openLoaded();
        type('gs');
        press('Enter');
        expect(setup.navigate).toHaveBeenCalledWith('start.html');
        expect(document.querySelector('dialog.cp').open).toBe(false);
        expect(setup.recents.get()).toEqual(['p:start']);
    });

    it('Ctrl/Cmd/Shift+Enter opens a new tab and leaves the palette open', async () => {
        await openLoaded();
        type('gs');
        for (const mod of [{ ctrlKey: true }, { metaKey: true }, { shiftKey: true }]) press('Enter', mod);
        expect(setup.openTab).toHaveBeenCalledTimes(3);
        expect(setup.openTab).toHaveBeenCalledWith('start.html');
        expect(setup.navigate).not.toHaveBeenCalled();
        expect(document.querySelector('dialog.cp').open).toBe(true);
    });

    it('runs an action with the context, after closing', async () => {
        await openLoaded();
        press('Enter'); // the first suggestion is the action
        expect(run).toHaveBeenCalledWith({ tag: 'ctx' });
        expect(document.querySelector('dialog.cp').open).toBe(false);
    });

    it('ignores Enter and the arrows while an IME is composing', async () => {
        await openLoaded();
        expect(press('Enter', { isComposing: true }).defaultPrevented).toBe(false);
        expect(press('ArrowDown', { isComposing: true }).defaultPrevented).toBe(false);
        expect(press('Enter', { keyCode: 229 }).defaultPrevented).toBe(false);
        expect(setup.navigate).not.toHaveBeenCalled();
        expect(active()).toBe(options()[0]);
    });

    it('Enter with nothing to choose does nothing', async () => {
        await openLoaded();
        type('zzzq');
        press('Enter');
        expect(setup.navigate).not.toHaveBeenCalled();
        expect(document.querySelector('dialog.cp').open).toBe(true);
    });
});

describe('mouse and focus', () => {
    it('opens the clicked option, and hover moves the selection', async () => {
        await openLoaded();
        const target = options()[2];
        target.dispatchEvent(new MouseEvent('mousemove', { bubbles: true }));
        expect(active()).toBe(target);
        target.click();
        expect(setup.navigate).toHaveBeenCalledWith('keys.html');
    });

    it('keeps focus in the input when the list is pressed', async () => {
        await openLoaded();
        const event = new MouseEvent('mousedown', { bubbles: true, cancelable: true });
        options()[0].dispatchEvent(event);
        expect(event.defaultPrevented).toBe(true);
    });

    it('closes on a click on the backdrop but not inside the box', async () => {
        await openLoaded();
        options()[0].querySelector('.cp-title').click();
        const dialog = document.querySelector('dialog.cp');
        dialog.click();
        expect(dialog.open).toBe(false);
    });

    it('gives focus back to the opener on close, and the Close button works', async () => {
        const opener = document.createElement('button');
        document.body.append(opener);
        opener.focus();
        await openLoaded();
        document.querySelector('.cp-close').click();
        expect(document.querySelector('dialog.cp').open).toBe(false);
        expect(document.activeElement).toBe(opener);
    });

    it('toggle opens then closes, and a reopened palette starts empty', async () => {
        await openLoaded();
        type('gs');
        setup.palette.toggle();
        expect(setup.palette.isOpen).toBe(false);
        setup.palette.toggle();
        expect(input().value).toBe('');
        expect(titles()).toContain('Scroll to top');
    });
});

describe('recent items', () => {
    it('lists what was chosen first, most recent on top, when opened with no query', async () => {
        await openLoaded();
        type('moving');
        press('Enter');
        setup.palette.open();
        await vi.waitFor(() => expect(document.getElementById('cp-group-recent')).not.toBeNull());
        expect(titles()[0]).toBe('Moving');
        expect(document.getElementById('cp-group-recent').textContent).toBe('Recent');
    });
});

describe('urlPath', () => {
    it.each([
        ['start.html', 'start.html'],
        ['start.html#install', 'start.html'],
        ['/docs/start.html?x=1#a', 'start.html'],
        ['/', 'index.html'],
        ['', 'index.html'],
    ])('%s -> %s', (url, expected) => {
        expect(urlPath(url)).toBe(expected);
    });
});

import { describe, expect, it, vi } from 'vitest';
import { RECENT_KEY, RECENT_MAX, createRecents } from '../src/recent.js';
import { safeStorage } from '../src/storage.js';
import { THEME_KEY, createTheme } from '../src/theme.js';

function fakeStore(initial = {}) {
    const data = new Map(Object.entries(initial));
    return {
        getItem: (k) => data.get(k) ?? null,
        setItem: (k, v) => data.set(k, String(v)),
        removeItem: (k) => data.delete(k),
        data,
    };
}

describe('safeStorage', () => {
    it('reads and writes through to the store', () => {
        const store = fakeStore();
        const storage = safeStorage(() => store);
        storage.set('k', 'v');
        expect(store.data.get('k')).toBe('v');
        expect(storage.get('k')).toBe('v');
        storage.remove('k');
        expect(storage.get('k')).toBeNull();
    });

    it('falls back to memory when the storage getter throws', () => {
        const storage = safeStorage(() => {
            throw new DOMException('denied', 'SecurityError');
        });
        storage.set('k', 'v');
        expect(storage.get('k')).toBe('v');
    });

    it('falls back to memory when every call throws (quota, blocked)', () => {
        const boom = vi.fn(() => {
            throw new DOMException('full', 'QuotaExceededError');
        });
        const storage = safeStorage(() => ({ getItem: boom, setItem: boom, removeItem: boom }));
        storage.set('k', 'v');
        expect(storage.get('k')).toBe('v');
        storage.remove('k');
        expect(storage.get('k')).toBeNull();
    });
});

describe('recents', () => {
    const make = (initial) => {
        const store = fakeStore(initial);
        return { recents: createRecents(safeStorage(() => store)), store };
    };

    it('keeps the newest first, without duplicates, up to the cap', () => {
        const { recents } = make();
        for (const id of ['a', 'b', 'c', 'a', 'd', 'e', 'f']) recents.push(id);
        expect(recents.get()).toEqual(['f', 'e', 'd', 'a', 'c']);
        expect(recents.get()).toHaveLength(RECENT_MAX);
    });

    it('clears', () => {
        const { recents, store } = make();
        recents.push('a');
        recents.clear();
        expect(recents.get()).toEqual([]);
        expect(store.data.has(RECENT_KEY)).toBe(false);
    });

    it.each([
        ['not JSON', '{nope'],
        ['null', 'null'],
        ['a bare array', '["a"]'],
        ['the wrong version', '{"v":9,"ids":["a"]}'],
        ['ids not an array', '{"v":1,"ids":"a"}'],
    ])('treats %s as empty', (_, raw) => {
        expect(make({ [RECENT_KEY]: raw }).recents.get()).toEqual([]);
    });

    it('drops bad entries, duplicates and anything past the cap', () => {
        const ids = ['a', 3, null, '', 'a', 'x'.repeat(500), { id: 'b' }, 'b', 'c', 'd', 'e', 'f', 'g'];
        const { recents } = make({ [RECENT_KEY]: JSON.stringify({ v: 1, ids }) });
        expect(recents.get()).toEqual(['a', 'b', 'c', 'd', 'e']);
    });

    it('recovers from a corrupt value on the next push', () => {
        const { recents, store } = make({ [RECENT_KEY]: '{nope' });
        recents.push('a');
        expect(JSON.parse(store.data.get(RECENT_KEY))).toEqual({ v: 1, ids: ['a'] });
    });
});

describe('theme', () => {
    it('cycles system, light, dark, system and sets data-theme', () => {
        const root = document.createElement('html');
        const store = fakeStore();
        const theme = createTheme(root, safeStorage(() => store));
        expect(theme.current()).toBe('system');
        expect(root.dataset.theme).toBeUndefined();
        expect(theme.cycle()).toBe('light');
        expect(root.dataset.theme).toBe('light');
        expect(theme.cycle()).toBe('dark');
        expect(store.data.get(THEME_KEY)).toBe('dark');
        expect(theme.cycle()).toBe('system');
        expect(root.dataset.theme).toBeUndefined();
        expect(store.data.has(THEME_KEY)).toBe(false);
    });

    it('applies the stored theme and ignores junk', () => {
        const root = document.createElement('html');
        createTheme(root, safeStorage(() => fakeStore({ [THEME_KEY]: 'dark' })));
        expect(root.dataset.theme).toBe('dark');
        const other = document.createElement('html');
        createTheme(other, safeStorage(() => fakeStore({ [THEME_KEY]: '<script>' })));
        expect(other.dataset.theme).toBeUndefined();
    });
});

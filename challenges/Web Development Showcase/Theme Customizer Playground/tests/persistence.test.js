import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { decodeHash, encodeHash } from '../src/share.js';
import { BACKUP_KEY, KEY, createStore } from '../src/storage.js';
import { defaults } from '../src/schema.js';
import { themeArb } from './helpers.js';

const fakeStorage = (initial = {}) => {
  const data = { ...initial };
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => void (data[k] = String(v)),
    removeItem: (k) => void delete data[k],
  };
};

describe('share links', () => {
  it('round-trips any valid theme', () => {
    fc.assert(
      fc.property(themeArb, (theme) => {
        const r = decodeHash(encodeHash(theme));
        expect(r.ok).toBe(true);
        expect(r.theme).toEqual(theme);
      }),
    );
  });

  it('is readable and short enough to paste', () => {
    const hash = encodeHash(defaults());
    expect(hash).toMatch(/^#brandL=0\.55&brandLDark=0\.7&/);
    expect(hash.length).toBeLessThan(400);
  });

  it('returns null when the hash is not a theme, and an error when it is a bad one', () => {
    expect(decodeHash('')).toBeNull();
    expect(decodeHash('#section-2')).toBeNull();
    expect(decodeHash('#radius=5em').ok).toBe(false);
    expect(decodeHash('#radius=').ok).toBe(false);
    expect(decodeHash('#radius=500').ok).toBe(false);
    expect(decodeHash('#bodyFont=<script>').ok).toBe(false);
    expect(decodeHash('#radius=12&evil=1')).toMatchObject({ ok: true, ignored: 0 });
  });

  it('partial hashes keep the base theme for the rest', () => {
    expect(decodeHash('#radius=12').theme).toEqual({ ...defaults(), radius: 12 });
  });
});

describe('autosave', () => {
  it('saves and loads a theme', () => {
    const storage = fakeStorage();
    const store = createStore(storage);
    const theme = { ...defaults(), radius: 3, bodyFont: 'mono' };
    expect(store.save(theme)).toBe(true);
    expect(createStore(storage).load()).toEqual({ theme, notice: null });
  });

  it('loads the defaults quietly when nothing is saved', () => {
    expect(createStore(fakeStorage()).load()).toEqual({ theme: defaults(), notice: null });
  });

  it.each([
    ['not JSON', '{oops'],
    ['wrong version', JSON.stringify({ v: 2, theme: defaults() })],
    ['out-of-range value', JSON.stringify({ v: 1, theme: { ...defaults(), radius: 1e9 } })],
    ['wrong type', JSON.stringify({ v: 1, theme: { ...defaults(), brandH: 'red' } })],
    ['a bare number', '42'],
    ['null', 'null'],
  ])('moves aside corrupt data (%s) and loads the defaults', (_, text) => {
    const storage = fakeStorage({ [KEY]: text });
    const { theme, notice } = createStore(storage).load();
    expect(theme).toEqual(defaults());
    expect(notice).toMatch(/set aside/);
    expect(storage.data[BACKUP_KEY]).toBe(text);
    expect(storage.data[KEY]).toBeUndefined();
  });

  it('works without storage, and when storage throws', () => {
    const none = createStore(null);
    expect(none.load().notice).toMatch(/Saving is off/);
    expect(none.save(defaults())).toBe(false);
    const boom = () => {
      throw new Error('denied');
    };
    const throwing = createStore({ getItem: boom, setItem: boom, removeItem: boom });
    expect(throwing.load()).toMatchObject({ theme: defaults(), notice: expect.stringMatching(/Saving is off/) });
    expect(throwing.save(defaults())).toBe(false);
  });
});

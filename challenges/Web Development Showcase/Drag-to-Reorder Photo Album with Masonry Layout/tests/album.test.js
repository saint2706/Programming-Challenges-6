// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest';
import { createAlbum } from '../src/album.js';
import { STORAGE_KEY, clearOrder, loadOrder, saveOrder } from '../src/storage.js';

const photos = ['a', 'b', 'c', 'd'].map((id, i) => ({
  id,
  src: `photos/${id}.webp`,
  title: `Photo ${id.toUpperCase()}`,
  photographer: 'Someone <b>bold</b>',
  width: 800,
  height: 400 + i * 100,
  alt: `alt ${id}`,
}));

function memoryStorage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => void data.set(k, String(v)),
    removeItem: (k) => void data.delete(k),
    data,
  };
}

let grid;
let messages;
const domOrder = () => [...grid.children].map((li) => li.dataset.id);
const press = (id, key) =>
  album.tiles.get(id).dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
let album;
const make = (storage = memoryStorage()) => {
  album = createAlbum({ grid, photos, storage, announce: (m) => messages.push(m), prefersReducedMotion: () => true });
  return storage;
};

beforeEach(() => {
  document.body.innerHTML = '<ul id="album"></ul>';
  grid = document.getElementById('album');
  messages = [];
});

describe('rendering', () => {
  it('renders one tile per photo in canonical order, in the DOM', () => {
    make();
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
    expect(grid.querySelectorAll('img')).toHaveLength(4);
    expect(grid.querySelector('img').alt).toBe('alt a');
  });
  it('never interprets photo data as HTML', () => {
    make();
    expect(grid.querySelector('b')).toBeNull();
    expect(grid.querySelector('.tile-credit').textContent).toBe('by Someone <b>bold</b>');
  });
  it('makes tiles focusable and labels their position', () => {
    make();
    const tile = album.tiles.get('b');
    expect(tile.tabIndex).toBe(0);
    expect(tile.getAttribute('aria-label')).toContain('2 of 4');
  });
  it('restores a saved order and reconciles stale entries', () => {
    make(memoryStorage({ [STORAGE_KEY]: JSON.stringify({ v: 1, order: ['c', 'zzz', 'a'] }) }));
    expect(domOrder()).toEqual(['c', 'a', 'b', 'd']);
  });
  it('survives corrupt storage', () => {
    make(memoryStorage({ [STORAGE_KEY]: '{not json' }));
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
  });
  it('works with no storage at all', () => {
    album = createAlbum({ grid, photos, storage: null, prefersReducedMotion: () => true });
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
  });
});

describe('keyboard reordering', () => {
  it('Space grabs, arrows move, Space drops and persists', () => {
    const storage = make();
    album.tiles.get('a').focus();
    press('a', ' ');
    expect(album.isGrabbed()).toBe(true);
    expect(album.tiles.get('a').dataset.grabbed).toBe('true');
    press('a', 'ArrowRight');
    press('a', 'ArrowRight');
    expect(domOrder()).toEqual(['b', 'c', 'a', 'd']);
    expect(storage.data.has(STORAGE_KEY)).toBe(false); // not saved until dropped
    press('a', ' ');
    expect(album.isGrabbed()).toBe(false);
    expect(JSON.parse(storage.data.get(STORAGE_KEY)).order).toEqual(['b', 'c', 'a', 'd']);
  });
  it('Enter also grabs and drops', () => {
    make();
    press('b', 'Enter');
    press('b', 'ArrowLeft');
    press('b', 'Enter');
    expect(domOrder()).toEqual(['b', 'a', 'c', 'd']);
    expect(album.isGrabbed()).toBe(false);
  });
  it('Up/Down behave like Left/Right', () => {
    make();
    press('b', ' ');
    press('b', 'ArrowDown');
    expect(domOrder()).toEqual(['a', 'c', 'b', 'd']);
    press('b', 'ArrowUp');
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
  });
  it('Home and End jump to the ends', () => {
    make();
    press('c', ' ');
    press('c', 'Home');
    expect(domOrder()).toEqual(['c', 'a', 'b', 'd']);
    press('c', 'End');
    expect(domOrder()).toEqual(['a', 'b', 'd', 'c']);
  });
  it('Escape restores the original order and saves nothing', () => {
    const storage = make();
    press('a', ' ');
    press('a', 'End');
    press('a', 'Escape');
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
    expect(album.isGrabbed()).toBe(false);
    expect(storage.data.has(STORAGE_KEY)).toBe(false);
  });
  it('stops at the ends and says so', () => {
    make();
    press('a', ' ');
    press('a', 'ArrowLeft');
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
    expect(messages.at(-1)).toMatch(/already at the start/);
  });
  it('arrows do nothing unless grabbed', () => {
    make();
    press('a', 'ArrowRight');
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
  });
  it('drops when focus leaves the tile', () => {
    const storage = make();
    press('a', ' ');
    press('a', 'ArrowRight');
    album.tiles.get('a').dispatchEvent(new FocusEvent('focusout', { bubbles: true }));
    expect(album.isGrabbed()).toBe(false);
    expect(JSON.parse(storage.data.get(STORAGE_KEY)).order).toEqual(['b', 'a', 'c', 'd']);
  });
  it('keeps focus on the moved tile', () => {
    make();
    press('a', ' ');
    press('a', 'ArrowRight');
    expect(document.activeElement).toBe(album.tiles.get('a'));
  });
  it('announces grab, move, drop and cancel', () => {
    make();
    press('a', ' ');
    press('a', 'ArrowRight');
    press('a', ' ');
    press('b', ' ');
    press('b', 'Escape');
    expect(messages[0]).toMatch(/^Grabbed Photo A\. Position 1 of 4\./);
    expect(messages[1]).toBe('Moved Photo A. Position 2 of 4.');
    expect(messages[2]).toBe('Dropped Photo A. Position 2 of 4.');
    expect(messages.at(-1)).toMatch(/^Reorder cancelled\. Photo B is back at position 1 of 4\./);
  });
  it('ignores key events from descendants and modified keys', () => {
    make();
    const img = album.tiles.get('a').querySelector('img');
    img.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
    expect(album.isGrabbed()).toBe(false);
    album.tiles.get('a').dispatchEvent(new KeyboardEvent('keydown', { key: ' ', ctrlKey: true, bubbles: true }));
    expect(album.isGrabbed()).toBe(false);
  });
});

describe('pointer drop and reset', () => {
  it('commitDrop reorders, persists and announces', () => {
    const storage = make();
    album.commitDrop('a', 'c', 'right');
    expect(domOrder()).toEqual(['b', 'c', 'a', 'd']);
    expect(JSON.parse(storage.data.get(STORAGE_KEY)).order).toEqual(['b', 'c', 'a', 'd']);
    expect(messages.at(-1)).toBe('Moved Photo A. Position 3 of 4.');
  });
  it('a no-op drop does not announce', () => {
    make();
    album.commitDrop('b', 'a', 'right');
    expect(messages).toEqual([]);
  });
  it('reload restores the dragged order', () => {
    const storage = make();
    album.commitDrop('d', 'a', 'left');
    document.body.innerHTML = '<ul id="album"></ul>';
    grid = document.getElementById('album');
    make(storage);
    expect(domOrder()).toEqual(['d', 'a', 'b', 'c']);
  });
  it('reset restores canonical order and clears storage', () => {
    const storage = make();
    album.commitDrop('d', 'a', 'left');
    album.reset();
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
    expect(album.isDefaultOrder()).toBe(true);
    expect(storage.data.has(STORAGE_KEY)).toBe(false);
  });
  it('reset during a keyboard grab cancels it first', () => {
    make();
    press('a', ' ');
    press('a', 'End');
    album.reset();
    expect(album.isGrabbed()).toBe(false);
    expect(domOrder()).toEqual(['a', 'b', 'c', 'd']);
  });
});

describe('storage helpers', () => {
  it('tolerate storage that throws', () => {
    const angry = {
      getItem: () => {
        throw new Error('denied');
      },
      setItem: () => {
        throw new Error('full');
      },
      removeItem: () => {
        throw new Error('denied');
      },
    };
    expect(loadOrder(angry, ['a', 'b'])).toEqual(['a', 'b']);
    expect(saveOrder(angry, ['a'])).toBe(false);
    expect(() => clearOrder(angry)).not.toThrow();
  });
  it('report save success', () => {
    expect(saveOrder(memoryStorage(), ['a'])).toBe(true);
    expect(saveOrder(null, ['a'])).toBe(true);
  });
});

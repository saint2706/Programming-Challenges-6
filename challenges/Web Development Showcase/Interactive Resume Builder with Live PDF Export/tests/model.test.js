import { describe, expect, it } from 'vitest';
import {
  addBullet,
  addEntry,
  getPath,
  moveBullet,
  moveEntry,
  moveSection,
  parsePath,
  removeBullet,
  removeEntry,
  setLayout,
  setPath,
  setSectionHidden,
  toPathString,
} from '../src/lib/model.js';
import { LIMITS, SECTION_IDS, emptyResume } from '../src/lib/schema.js';

function sample() {
  const doc = emptyResume();
  doc.basics.name = 'Ada';
  doc.work = [
    { name: 'A', position: 'One', url: '', startDate: '', endDate: '', summary: '', highlights: ['a', 'b', 'c'] },
    { name: 'B', position: 'Two', url: '', startDate: '', endDate: '', summary: '', highlights: [] },
    { name: 'C', position: 'Three', url: '', startDate: '', endDate: '', summary: '', highlights: [] },
  ];
  doc.skills = [{ name: 'Languages', level: '', keywords: ['JS'] }];
  return doc;
}

describe('paths', () => {
  it('parses and prints dotted paths with numeric indices', () => {
    expect(parsePath('work.1.highlights.0')).toEqual(['work', 1, 'highlights', 0]);
    expect(toPathString(['work', 1, 'highlights', 0])).toBe('work.1.highlights.0');
    expect(parsePath('basics.name')).toEqual(['basics', 'name']);
  });

  it('reads values by path', () => {
    const doc = sample();
    expect(getPath(doc, 'work.0.highlights.2')).toBe('c');
    expect(getPath(doc, ['basics', 'name'])).toBe('Ada');
    expect(getPath(doc, 'work.9.name')).toBeUndefined();
  });
});

describe('setPath', () => {
  it('returns a new document and never mutates the old one', () => {
    const doc = sample();
    const next = setPath(doc, 'work.1.position', 'Changed');
    expect(next.work[1].position).toBe('Changed');
    expect(doc.work[1].position).toBe('Two');
    expect(next).not.toBe(doc);
  });

  it('shares untouched branches', () => {
    const doc = sample();
    const next = setPath(doc, 'work.1.position', 'Changed');
    expect(next.work[0]).toBe(doc.work[0]);
    expect(next.education).toBe(doc.education);
  });

  it('can set a bullet and a keywords array', () => {
    const doc = sample();
    expect(setPath(doc, 'work.0.highlights.1', 'z').work[0].highlights).toEqual(['a', 'z', 'c']);
    expect(setPath(doc, 'skills.0.keywords', ['JS', 'TS']).skills[0].keywords).toEqual(['JS', 'TS']);
  });

  it('refuses fields that do not exist and prototype-polluting keys', () => {
    const doc = sample();
    expect(() => setPath(doc, 'work.0.nope', 'x')).toThrow(RangeError);
    expect(() => setPath(doc, 'work.7.name', 'x')).toThrow(RangeError);
    expect(() => setPath(doc, '__proto__.polluted', 'x')).toThrow(RangeError);
    expect(() => setPath(doc, 'basics.constructor.prototype.polluted', 'x')).toThrow(RangeError);
    expect(({}).polluted).toBeUndefined();
  });
});

describe('entries', () => {
  it('adds a blank entry at the end', () => {
    const next = addEntry(sample(), 'education');
    expect(next.education).toHaveLength(1);
    expect(next.education[0].institution).toBe('');
  });

  it('refuses to exceed the per-section entry limit', () => {
    let doc = emptyResume();
    for (let i = 0; i < LIMITS.entries; i++) doc = addEntry(doc, 'awards');
    expect(() => addEntry(doc, 'awards')).toThrow(RangeError);
  });

  it('removes an entry', () => {
    const next = removeEntry(sample(), 'work', 1);
    expect(next.work.map((w) => w.name)).toEqual(['A', 'C']);
  });

  it('ignores an out-of-range removal', () => {
    const doc = sample();
    expect(removeEntry(doc, 'work', 9)).toBe(doc);
    expect(removeEntry(doc, 'work', -1)).toBe(doc);
  });

  it('moves an entry and is a no-op at the edges', () => {
    const doc = sample();
    expect(moveEntry(doc, 'work', 0, 2).work.map((w) => w.name)).toEqual(['B', 'C', 'A']);
    expect(moveEntry(doc, 'work', 2, 1).work.map((w) => w.name)).toEqual(['A', 'C', 'B']);
    expect(moveEntry(doc, 'work', 0, -1)).toBe(doc);
    expect(moveEntry(doc, 'work', 2, 3)).toBe(doc);
    expect(moveEntry(doc, 'work', 1, 1)).toBe(doc);
  });
});

describe('bullets', () => {
  it('adds, removes and moves bullets', () => {
    const doc = sample();
    expect(addBullet(doc, 'work', 0, 'highlights', 'd').work[0].highlights).toEqual(['a', 'b', 'c', 'd']);
    expect(removeBullet(doc, 'work', 0, 'highlights', 1).work[0].highlights).toEqual(['a', 'c']);
    expect(moveBullet(doc, 'work', 0, 'highlights', 0, 2).work[0].highlights).toEqual(['b', 'c', 'a']);
    expect(moveBullet(doc, 'work', 0, 'highlights', 0, -1)).toBe(doc);
  });

  it('refuses to exceed the bullet limit', () => {
    let doc = sample();
    for (let i = 0; i < LIMITS.bullets - 3; i++) doc = addBullet(doc, 'work', 0, 'highlights');
    expect(doc.work[0].highlights).toHaveLength(LIMITS.bullets);
    expect(() => addBullet(doc, 'work', 0, 'highlights')).toThrow(RangeError);
  });
});

describe('layout', () => {
  it('moves a section within the order and ignores moves past the ends', () => {
    const doc = emptyResume();
    const down = moveSection(doc, 'summary', 1);
    expect(down['x-layout'].sectionOrder.slice(0, 2)).toEqual(['work', 'summary']);
    const up = moveSection(down, 'summary', -1);
    expect(up['x-layout'].sectionOrder).toEqual(SECTION_IDS);
    expect(moveSection(doc, 'summary', -1)).toBe(doc);
    expect(moveSection(doc, 'awards', 1)).toBe(doc);
    expect(moveSection(doc, 'nope', 1)).toBe(doc);
  });

  it('hides and shows a section idempotently', () => {
    const doc = emptyResume();
    const hidden = setSectionHidden(doc, 'work', true);
    expect(hidden['x-layout'].hidden).toEqual(['work']);
    expect(setSectionHidden(hidden, 'work', true)).toBe(hidden);
    expect(setSectionHidden(hidden, 'work', false)['x-layout'].hidden).toEqual([]);
  });

  it('applies a valid layout patch and rejects an invalid one', () => {
    const doc = emptyResume();
    expect(setLayout(doc, { template: 'sidebar', pageSize: 'letter' })['x-layout']).toMatchObject({
      template: 'sidebar',
      pageSize: 'letter',
    });
    expect(() => setLayout(doc, { template: 'fancy' })).toThrow(RangeError);
    expect(() => setLayout(doc, { scale: 5 })).toThrow(RangeError);
  });
});

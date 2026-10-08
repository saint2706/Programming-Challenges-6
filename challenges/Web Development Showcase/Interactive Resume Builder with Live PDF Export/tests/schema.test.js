import { describe, expect, it } from 'vitest';
import {
  FONT_PAIR_IDS,
  PAGE_SIZES,
  SECTION_IDS,
  TEMPLATE_IDS,
  emptyResume,
  formatIssues,
  normalizeLayout,
  parseLayout,
  parseResume,
} from '../src/lib/schema.js';

describe('constants', () => {
  it('lists the sections, templates, page sizes and font pairs', () => {
    expect(SECTION_IDS).toEqual([
      'summary',
      'work',
      'education',
      'projects',
      'skills',
      'certificates',
      'languages',
      'awards',
    ]);
    expect(TEMPLATE_IDS).toEqual(['classic', 'sidebar', 'compact']);
    expect(PAGE_SIZES).toEqual(['a4', 'letter']);
    expect(FONT_PAIR_IDS).toEqual(['inter', 'serif', 'plex']);
  });
});

describe('emptyResume', () => {
  it('is a valid resume with default layout', () => {
    const doc = emptyResume();
    expect(doc.basics.name).toBe('');
    expect(doc.work).toEqual([]);
    expect(doc['x-layout']).toEqual({
      template: 'classic',
      pageSize: 'a4',
      accent: '#1d4ed8',
      fontPair: 'inter',
      scale: 1,
      sectionOrder: SECTION_IDS,
      hidden: [],
    });
    expect(parseResume(doc).ok).toBe(true);
  });

  it('returns a fresh object every call', () => {
    expect(emptyResume()).not.toBe(emptyResume());
  });
});

describe('parseResume', () => {
  it('accepts partial ISO dates and rejects other formats', () => {
    const ok = (startDate) => parseResume({ work: [{ name: 'A', startDate }] }).ok;
    expect(ok('2020')).toBe(true);
    expect(ok('2020-05')).toBe(true);
    expect(ok('2020-05-17')).toBe(true);
    expect(ok('')).toBe(true);
    expect(ok('05/2020')).toBe(false);
    expect(ok('2020-13')).toBe(false);
    expect(ok('Jun 2025')).toBe(false);
  });

  it('reports field paths for bad input', () => {
    const result = parseResume({ work: [{ name: 'A' }, { name: 'B', startDate: 'soon' }] });
    expect(result.ok).toBe(false);
    expect(result.errors[0].path).toBe('work.1.startDate');
    expect(formatIssues(result.errors)).toContain('work.1.startDate');
  });

  it('rejects javascript: and other unsafe links everywhere a link is allowed', () => {
    expect(parseResume({ basics: { url: 'javascript:alert(1)' } }).ok).toBe(false);
    expect(parseResume({ work: [{ url: 'data:text/html,x' }] }).ok).toBe(false);
    expect(parseResume({ basics: { profiles: [{ network: 'X', url: 'vbscript:x' }] } }).ok).toBe(false);
    expect(parseResume({ basics: { url: 'https://example.com' } }).ok).toBe(true);
    expect(parseResume({ basics: { url: '' } }).ok).toBe(true);
  });

  it('drops unknown keys and cannot be used for prototype pollution', () => {
    const hostile = JSON.parse(
      '{"__proto__":{"polluted":true},"constructor":{"prototype":{"polluted":true}},"basics":{"name":"A","__proto__":{"x":1}},"extra":1}',
    );
    const result = parseResume(hostile);
    expect(result.ok).toBe(true);
    expect(result.data.extra).toBeUndefined();
    expect(Object.keys(result.data)).not.toContain('constructor');
    expect({}.polluted).toBeUndefined();
    expect(result.data.basics.x).toBeUndefined();
  });

  it('rejects a 10,000-entry import quickly', () => {
    const work = Array.from({ length: 10_000 }, (_, i) => ({ name: `Company ${i}` }));
    const started = performance.now();
    const result = parseResume({ work });
    expect(result.ok).toBe(false);
    expect(result.errors[0].path).toBe('work');
    expect(performance.now() - started).toBeLessThan(1000);
  });

  it('rejects oversized strings and bullet lists', () => {
    expect(parseResume({ basics: { summary: 'x'.repeat(5001) } }).ok).toBe(false);
    expect(parseResume({ work: [{ highlights: Array(51).fill('x') }] }).ok).toBe(false);
    expect(parseResume({ work: [{ position: 'x'.repeat(501) }] }).ok).toBe(false);
  });

  it('validates email and phone in basics', () => {
    expect(parseResume({ basics: { email: 'user@example.com' } }).ok).toBe(true);
    expect(parseResume({ basics: { phone: '+1 (555) 123-4567' } }).ok).toBe(true);
    expect(parseResume({ basics: { email: '', phone: '' } }).ok).toBe(true);
    expect(parseResume({ basics: { email: 'not-an-email' } }).ok).toBe(false);
    expect(parseResume({ basics: { email: 'user@' } }).ok).toBe(false);
    expect(parseResume({ basics: { phone: 'call me!' } }).ok).toBe(false);
  });

  it('rejects non-objects and bad layout values', () => {
    expect(parseResume(null).ok).toBe(false);
    expect(parseResume('text').ok).toBe(false);
    expect(parseResume([]).ok).toBe(false);
    expect(parseResume({ 'x-layout': { accent: 'red' } }).ok).toBe(false);
    expect(parseResume({ 'x-layout': { template: 'fancy' } }).ok).toBe(false);
    expect(parseResume({ 'x-layout': { scale: 2 } }).ok).toBe(false);
    expect(parseResume({ 'x-layout': { pageSize: 'a3' } }).ok).toBe(false);
  });

  it('round-trips through JSON', () => {
    const doc = emptyResume();
    doc.basics.name = 'Ada Lovelace';
    doc.work.push({
      name: 'Analytical Engines',
      position: 'Programmer',
      url: '',
      startDate: '1842',
      endDate: '',
      summary: '',
      highlights: ['Wrote the first program'],
    });
    const result = parseResume(JSON.parse(JSON.stringify(doc)));
    expect(result.ok).toBe(true);
    expect(result.data).toEqual(doc);
  });
});

describe('parseLayout', () => {
  it('fills defaults for a partial layout', () => {
    const layout = parseLayout({ template: 'sidebar' });
    expect(layout.template).toBe('sidebar');
    expect(layout.pageSize).toBe('a4');
    expect(layout.sectionOrder).toEqual(SECTION_IDS);
  });

  it('returns null for an invalid layout', () => {
    expect(parseLayout({ scale: 3 })).toBeNull();
    expect(parseLayout({ accent: 'blue' })).toBeNull();
    expect(parseLayout('nope')).toBeNull();
  });
});

describe('normalizeLayout', () => {
  it('repairs the section order: drops unknown and duplicate ids, appends missing ones', () => {
    const layout = normalizeLayout({
      sectionOrder: ['work', 'nope', 'work', 'skills'],
      hidden: ['skills', 'x', 'skills'],
    });
    expect(layout.sectionOrder).toEqual([
      'work',
      'skills',
      'summary',
      'education',
      'projects',
      'certificates',
      'languages',
      'awards',
    ]);
    expect(layout.hidden).toEqual(['skills']);
  });

  it('fills defaults for missing keys', () => {
    expect(normalizeLayout({}).template).toBe('classic');
    expect(normalizeLayout(undefined).sectionOrder).toEqual(SECTION_IDS);
  });
});

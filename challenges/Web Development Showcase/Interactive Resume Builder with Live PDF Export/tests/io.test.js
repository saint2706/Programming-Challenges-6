import { describe, expect, it } from 'vitest';
import { MAX_IMPORT_BYTES, exportFileName, exportJson, importJson } from '../src/lib/io.js';
import { setPath } from '../src/lib/model.js';
import { emptyResume } from '../src/lib/schema.js';
import { longResume } from '../src/lib/sample-long.js';
import { seedResume } from '../src/lib/seed.js';

describe('exportJson / importJson', () => {
  it('round-trips a resume, layout included', () => {
    const doc = { ...longResume(), 'x-layout': { ...longResume()['x-layout'], template: 'sidebar', accent: '#047857' } };
    const result = importJson(exportJson(doc));
    expect(result.ok).toBe(true);
    expect(result.doc).toEqual(doc);
  });

  it('exports readable JSON Resume with the layout under x-layout', () => {
    const text = exportJson(seedResume());
    expect(text.endsWith('\n')).toBe(true);
    const parsed = JSON.parse(text);
    expect(parsed.basics.name).toBe('Rishabh Agrawal');
    expect(parsed['x-layout'].template).toBe('sidebar');
    expect(text).toContain('\n  "basics"');
  });

  it('keeps emoji and CJK text intact through an export and import', () => {
    const doc = setPath(longResume(), 'basics.summary', '日本語 🚀 café Ελληνικά 😀\nsecond line');
    const result = importJson(exportJson(doc));
    expect(result.ok).toBe(true);
    expect(result.doc.basics.summary).toBe('日本語 🚀 café Ελληνικά 😀\nsecond line');
  });

  it('accepts a plain JSON Resume file with no layout and fills the defaults', () => {
    const result = importJson(JSON.stringify({ basics: { name: 'Ada' }, work: [{ name: 'Engines', position: 'Analyst' }] }));
    expect(result.ok).toBe(true);
    expect(result.doc.basics.name).toBe('Ada');
    expect(result.doc['x-layout'].template).toBe('classic');
  });

  it('preserves email and phone and drops unknown fields (photo, other sections)', () => {
    const result = importJson(
      JSON.stringify({
        basics: { name: 'Ada', email: 'ada@example.com', phone: '555-0100', image: 'https://example.com/a.jpg' },
        volunteer: [{ organization: 'X' }],
        meta: { theme: 'flat' },
      }),
    );
    expect(result.ok).toBe(true);
    const text = JSON.stringify(result.doc);
    expect(text).toContain('ada@example.com');
    expect(text).toContain('555-0100');
    expect(text).not.toContain('volunteer');
    expect(text).not.toContain('image');
    expect(result.doc.basics.email).toBe('ada@example.com');
    expect(result.doc.basics.phone).toBe('555-0100');
  });
});

describe('importJson errors', () => {
  it('explains text that is not JSON', () => {
    const result = importJson('{ nope');
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/not valid JSON/i);
  });

  it('rejects JSON that is not an object', () => {
    for (const text of ['[]', '42', 'null', '"x"']) {
      const result = importJson(text);
      expect(result.ok, text).toBe(false);
      expect(result.message).toMatch(/object/i);
    }
  });

  it('lists which fields are wrong', () => {
    const result = importJson(JSON.stringify({ work: [{ startDate: 'soon' }], basics: { url: 'javascript:alert(1)' } }));
    expect(result.ok).toBe(false);
    expect(result.message).toContain('work.0.startDate');
    expect(result.message).toContain('basics.url');
  });

  it('rejects a file that is too large before parsing it', () => {
    const result = importJson(' '.repeat(MAX_IMPORT_BYTES + 1));
    expect(result.ok).toBe(false);
    expect(result.message).toMatch(/too large/i);
  });

  it('rejects a hostile 10,000-entry file and ignores prototype keys', () => {
    const big = importJson(JSON.stringify({ work: Array.from({ length: 10_000 }, () => ({ name: 'x' })) }));
    expect(big.ok).toBe(false);
    expect(big.message).toContain('work');

    const proto = importJson('{"__proto__":{"polluted":true},"basics":{"name":"A"}}');
    expect(proto.ok).toBe(true);
    expect({}.polluted).toBeUndefined();
  });

  it('never returns a document on failure', () => {
    expect(importJson('{ nope').doc).toBeUndefined();
  });
});

describe('exportFileName', () => {
  it('slugs the name', () => {
    expect(exportFileName(setPath(emptyResume(), 'basics.name', 'Ada Lovelace'))).toBe('ada-lovelace-resume.json');
    expect(exportFileName(setPath(emptyResume(), 'basics.name', 'José Núñez'))).toBe('jose-nunez-resume.json');
  });

  it('falls back when there is no usable name', () => {
    expect(exportFileName(emptyResume())).toBe('resume.json');
    expect(exportFileName(setPath(emptyResume(), 'basics.name', '../../  /\\ '))).toBe('resume.json');
    expect(exportFileName(setPath(emptyResume(), 'basics.name', '日本語'))).toBe('resume.json');
  });

  it('keeps a very long name short', () => {
    const name = 'a'.repeat(300);
    expect(exportFileName(setPath(emptyResume(), 'basics.name', name)).length).toBeLessThanOrEqual(60);
  });
});

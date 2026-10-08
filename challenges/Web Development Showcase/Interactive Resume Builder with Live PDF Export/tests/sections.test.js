import { describe, expect, it } from 'vitest';
import { SECTION_IDS, emptyResume, parseResume } from '../src/lib/schema.js';
import { SECTION_META, entryLabel, newEntry } from '../src/lib/sections.js';

describe('SECTION_META', () => {
  it('has a title for every section', () => {
    for (const id of SECTION_IDS) expect(SECTION_META[id].title).toBeTruthy();
  });
});

describe('newEntry', () => {
  it.each(SECTION_IDS.filter((id) => id !== 'summary'))(
    'creates a schema-valid blank entry for %s',
    (section) => {
      const doc = emptyResume();
      doc[section] = [newEntry(section)];
      const result = parseResume(doc);
      expect(result.ok).toBe(true);
      expect(result.data[section][0]).toEqual(newEntry(section));
    },
  );

  it('throws for a section that has no entries', () => {
    expect(() => newEntry('summary')).toThrow(/no entries/);
    expect(() => newEntry('nope')).toThrow(/no entries/);
  });

  it('returns a fresh object each time', () => {
    expect(newEntry('work')).not.toBe(newEntry('work'));
    expect(newEntry('work').highlights).not.toBe(newEntry('work').highlights);
  });
});

describe('entryLabel', () => {
  it('describes an entry for the side panel list', () => {
    expect(entryLabel('work', { position: 'Engineer', name: 'Acme' }, 0)).toBe('Engineer, Acme');
    expect(entryLabel('work', { position: '', name: 'Acme' }, 0)).toBe('Acme');
    expect(entryLabel('education', { institution: 'MIT', studyType: 'BS', area: 'CS' }, 1)).toBe(
      'MIT',
    );
    expect(entryLabel('skills', { name: 'Languages' }, 0)).toBe('Languages');
  });

  it('falls back to a numbered placeholder for an empty entry', () => {
    expect(entryLabel('work', newEntry('work'), 2)).toBe('New position 3');
    expect(entryLabel('certificates', newEntry('certificates'), 0)).toBe('New certificate 1');
  });
});

import { describe, expect, it } from 'vitest';
import { blockIndex, buildRegions, hashString } from '../src/lib/blocks.js';
import { getPath, moveSection, setPath, setSectionHidden } from '../src/lib/model.js';
import { emptyResume } from '../src/lib/schema.js';
import { longResume } from '../src/lib/sample-long.js';
import { seedResume } from '../src/lib/seed.js';
import { DEFAULT_TEMPLATE_ID, TEMPLATES, getTemplate } from '../src/templates/index.js';

const headings = (region) =>
  region.blocks.filter((b) => b.kind === 'heading').map((b) => b.data.section);
const region = (regions, id) => regions.find((r) => r.id === id);
const withTemplate = (doc, template) => ({ ...doc, 'x-layout': { ...doc['x-layout'], template } });
const classicSeed = () => withTemplate(seedResume(), 'classic');

describe('templates registry', () => {
  it('has the three templates and falls back to classic', () => {
    expect(Object.keys(TEMPLATES)).toEqual(['classic', 'sidebar', 'compact']);
    expect(getTemplate('sidebar').id).toBe('sidebar');
    expect(getTemplate('nope').id).toBe(DEFAULT_TEMPLATE_ID);
    expect(getTemplate(undefined).id).toBe('classic');
  });
});

describe('hashString', () => {
  it('is deterministic and sensitive to every character', () => {
    expect(hashString('abc')).toBe(hashString('abc'));
    expect(hashString('abc')).not.toBe(hashString('abd'));
    expect(hashString('')).not.toBe(hashString(' '));
  });
});

describe('buildRegions: classic', () => {
  it('always has a header block, even for an empty resume', () => {
    const regions = buildRegions(emptyResume());
    expect(regions).toHaveLength(1);
    expect(regions[0].id).toBe('main');
    expect(regions[0].blocks.map((b) => b.kind)).toEqual(['header']);
    expect(regions[0].blocks[0].data.name).toEqual({ path: 'basics.name', value: '' });
  });

  it('emits headings in section order and skips empty and hidden sections', () => {
    const doc = classicSeed();
    const main = region(buildRegions(doc), 'main');
    expect(headings(main)).toEqual(['summary', 'work', 'education', 'projects', 'skills', 'certificates', 'languages']);

    const reordered = moveSection(moveSection(doc, 'work', -1), 'education', -1);
    expect(headings(region(buildRegions(reordered), 'main')).slice(0, 3)).toEqual(['work', 'education', 'summary']);

    const hidden = setSectionHidden(doc, 'projects', true);
    expect(headings(region(buildRegions(hidden), 'main'))).not.toContain('projects');
  });

  it('omits the summary section when the summary is blank', () => {
    const doc = setPath(classicSeed(), 'basics.summary', '   ');
    expect(headings(region(buildRegions(doc), 'main'))).not.toContain('summary');
  });

  it('renders only the header when every section is hidden', () => {
    let doc = classicSeed();
    for (const id of doc['x-layout'].sectionOrder) doc = setSectionHidden(doc, id, true);
    const main = region(buildRegions(doc), 'main');
    expect(main.blocks.map((b) => b.kind)).toEqual(['header']);
  });

  it('keeps headings and entry headers with what follows, and splits only bullet lists', () => {
    const main = region(buildRegions(classicSeed()), 'main');
    for (const b of main.blocks) {
      if (b.kind === 'heading') expect(b.keepWithNext).toBe(true);
      if (b.kind === 'work-header') expect(b.keepWithNext).toBe(true);
      if (b.kind === 'bullets') {
        expect(b.keepWithNext).toBe(false);
        expect(b.rows.length).toBeGreaterThan(0);
      } else {
        expect(b.rows).toBeUndefined();
      }
    }
  });

  it('gives bullet rows ids that are real document paths', () => {
    const doc = classicSeed();
    const bullets = region(buildRegions(doc), 'main').blocks.filter((b) => b.kind === 'bullets');
    expect(bullets.length).toBeGreaterThan(0);
    for (const block of bullets) {
      for (const row of block.rows) {
        expect(row.id).toBe(row.path);
        expect(getPath(doc, row.path)).toBe(row.value);
      }
    }
  });

  it('gives every block in every region a unique id', () => {
    for (const template of Object.keys(TEMPLATES)) {
      const regions = buildRegions(withTemplate(longResume(2), template));
      const all = regions.flatMap((r) => r.blocks.map((b) => b.id));
      expect(new Set(all).size).toBe(all.length);
    }
  });

  it('keeps block ids stable under a text edit and changes only that block hash', () => {
    const doc = longResume();
    const edited = setPath(doc, 'work.1.position', 'Principal Engineer');
    const before = blockIndex(buildRegions(doc));
    const after = blockIndex(buildRegions(edited));
    expect([...after.keys()]).toEqual([...before.keys()]);
    const changed = [...after.keys()].filter((id) => after.get(id).hash !== before.get(id).hash);
    expect(changed).toEqual(['work.1.header']);
  });

  it('never links an unsafe url', () => {
    let doc = classicSeed();
    doc = setPath(doc, 'basics.profiles', [{ network: 'Evil', username: 'x', url: 'javascript:alert(1)' }]);
    const header = region(buildRegions(doc), 'main').blocks[0];
    expect(header.data.contact.profiles).toEqual([{ label: 'Evil', href: null }]);
  });

  it('gives entry headers a sanitised link with a host-name label', () => {
    const doc = longResume();
    const project = blockIndex(buildRegions(doc)).get('projects.0.header');
    expect(project.data.link).toEqual({ href: 'https://example.com/tidepool', label: 'example.com/tidepool' });

    const long = setPath(doc, 'projects.0.url', `https://www.example.com/${'a'.repeat(80)}/?q=1#top`);
    const label = blockIndex(buildRegions(long)).get('projects.0.header').data.link.label;
    expect(label.startsWith('example.com/aaaa')).toBe(true);
    expect(label.length).toBeLessThanOrEqual(48);
    expect(label.endsWith('…')).toBe(true);

    const unsafe = setPath(doc, 'projects.0.url', 'javascript:alert(1)');
    expect(blockIndex(buildRegions(unsafe)).get('projects.0.header').data.link).toBeNull();
    expect(blockIndex(buildRegions(doc)).get('work.0.header').data.link).toBeNull();

    const mail = setPath(doc, 'projects.0.url', 'mailto:avery@example.com');
    expect(blockIndex(buildRegions(mail)).get('projects.0.header').data.link.label).toBe('avery@example.com');
  });

  it('survives an enormous unbroken word', () => {
    const doc = setPath(longResume(), 'work.0.highlights.0', 'x'.repeat(20_000));
    const index = blockIndex(buildRegions(doc));
    expect(index.get('work.0.bullets').rows[0].value).toHaveLength(20_000);
  });
});

describe('buildRegions: sidebar', () => {
  it('splits sections between a main and a sidebar region', () => {
    const regions = buildRegions(withTemplate(classicSeed(), 'sidebar'));
    expect(regions.map((r) => r.id)).toEqual(['main', 'sidebar']);
    expect(headings(region(regions, 'main'))).toEqual(['summary', 'work', 'education', 'projects']);
    expect(headings(region(regions, 'sidebar'))).toEqual(['skills', 'certificates', 'languages']);
    expect(region(regions, 'main').blocks[0].data.contact).toBeNull();
    expect(region(regions, 'sidebar').blocks[0].kind).toBe('contact');
  });

  it('leaves the sidebar empty when it has nothing to show', () => {
    const regions = buildRegions(withTemplate(emptyResume(), 'sidebar'));
    expect(region(regions, 'sidebar').blocks).toEqual([]);
  });
});

describe('buildRegions: compact', () => {
  it('is a single region like classic', () => {
    const regions = buildRegions(withTemplate(classicSeed(), 'compact'));
    expect(regions.map((r) => r.id)).toEqual(['main']);
    expect(headings(regions[0])).toContain('work');
  });
});

describe('blockIndex', () => {
  it('maps ids to blocks across regions', () => {
    const regions = buildRegions(withTemplate(classicSeed(), 'sidebar'));
    const index = blockIndex(regions);
    expect(index.get('header').kind).toBe('header');
    expect(index.get('contact').kind).toBe('contact');
    expect(index.size).toBe(regions.reduce((n, r) => n + r.blocks.length, 0));
  });
});

import { describe, expect, it } from 'vitest';
import { parseResume } from '../src/lib/schema.js';
import { longResume } from '../src/lib/sample-long.js';
import { blankResume, seedResume } from '../src/lib/seed.js';

describe('seedResume', () => {
  it('is a valid resume for the real author', () => {
    const result = parseResume(seedResume());
    expect(result.ok).toBe(true);
    expect(result.data.basics.name).toBe('Rishabh Agrawal');
    expect(result.data.work.length).toBeGreaterThan(0);
    expect(result.data.education.length).toBeGreaterThan(0);
    expect(result.data.projects.length).toBeGreaterThan(0);
    expect(result.data.skills.length).toBeGreaterThan(0);
  });

  it('never contains an email address or phone number', () => {
    const doc = seedResume();
    const text = JSON.stringify(doc);
    expect(text).not.toMatch(/@[a-z0-9-]+\.[a-z]{2,}/i);
    expect(text).not.toMatch(/\+?\d[\d\s-]{8,}\d/);
    expect(text).not.toContain('gim.ac.in');
    expect(text).not.toContain('9137095017');
    expect(doc.basics.email).toBe('');
    expect(doc.basics.phone).toBe('');
  });

  it('links out to GitHub and LinkedIn instead', () => {
    const urls = seedResume().basics.profiles.map((p) => p.url);
    expect(urls).toContain('https://github.com/saint2706');
    expect(urls.some((u) => u.startsWith('https://www.linkedin.com/in/'))).toBe(true);
  });

  it('starts in the Sidebar template', () => {
    expect(seedResume()['x-layout'].template).toBe('sidebar');
    expect(blankResume()['x-layout'].template).toBe('classic');
  });

  it('returns a fresh copy each time', () => {
    const a = seedResume();
    a.basics.name = 'changed';
    expect(seedResume().basics.name).toBe('Rishabh Agrawal');
  });
});

describe('blankResume', () => {
  it('is valid and empty', () => {
    const doc = blankResume();
    expect(parseResume(doc).ok).toBe(true);
    expect(doc.basics.name).toBe('');
    expect(doc.work).toEqual([]);
  });
});

describe('longResume', () => {
  it('is a valid fictional resume', () => {
    const doc = longResume();
    expect(parseResume(doc).ok).toBe(true);
    expect(doc.basics.name).toBe('Avery Morgan');
    expect(doc.work.length).toBeGreaterThanOrEqual(6);
    expect(JSON.stringify(doc)).not.toContain('Rishabh');
  });

  it('is deterministic', () => {
    expect(longResume(2)).toEqual(longResume(2));
  });

  it('scales with the copies argument and stays within the schema limits', () => {
    const one = longResume(1);
    const many = longResume(10);
    expect(many.work.length).toBe(one.work.length * 10);
    expect(parseResume(many).ok).toBe(true);
  });

  it('applies layout overrides', () => {
    const doc = longResume(1, { template: 'sidebar', pageSize: 'letter' });
    expect(doc['x-layout']).toMatchObject({ template: 'sidebar', pageSize: 'letter' });
    expect(parseResume(doc).ok).toBe(true);
  });
});

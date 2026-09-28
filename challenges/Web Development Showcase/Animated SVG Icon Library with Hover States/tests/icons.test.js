import { describe, expect, it } from 'vitest';
import { CATEGORIES, ICONS, SMIL_TECHNIQUE_MAX, SMIL_TECHNIQUE_MIN } from '../scripts/icons.js';
import { countSmilAnimations } from '../scripts/smil.js';

const CATEGORY_IDS = new Set(CATEGORIES.map((c) => c.id));
const REQUIRED_FIELDS = ['id', 'name', 'category', 'technique', 'description', 'markup'];

describe('icon set shape', () => {
  it('has between 16 and 20 icons per the challenge brief', () => {
    expect(ICONS.length).toBeGreaterThanOrEqual(16);
    expect(ICONS.length).toBeLessThanOrEqual(20);
  });

  it.each(ICONS.map((icon) => [icon.id, icon]))('icon "%s" has every required field as a non-empty string', (_id, icon) => {
    for (const field of REQUIRED_FIELDS) {
      expect(typeof icon[field], `${field} on ${icon.id}`).toBe('string');
      expect(icon[field].length, `${field} on ${icon.id}`).toBeGreaterThan(0);
    }
  });

  it('has no duplicate icon ids', () => {
    const ids = ICONS.map((icon) => icon.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('only uses technique values of "css" or "smil"', () => {
    for (const icon of ICONS) {
      expect(['css', 'smil']).toContain(icon.technique);
    }
  });

  it('only uses category ids declared in CATEGORIES', () => {
    for (const icon of ICONS) {
      expect(CATEGORY_IDS.has(icon.category), `unknown category "${icon.category}" on ${icon.id}`).toBe(true);
    }
  });
});

describe('SMIL vs CSS split', () => {
  const smilIcons = ICONS.filter((icon) => icon.technique === 'smil');
  const cssIcons = ICONS.filter((icon) => icon.technique === 'css');

  it(`uses SMIL for a handful of icons (${SMIL_TECHNIQUE_MIN}-${SMIL_TECHNIQUE_MAX}), not none and not most`, () => {
    expect(smilIcons.length).toBeGreaterThanOrEqual(SMIL_TECHNIQUE_MIN);
    expect(smilIcons.length).toBeLessThanOrEqual(SMIL_TECHNIQUE_MAX);
  });

  it('every SMIL-technique icon actually contains at least one animate element', () => {
    for (const icon of smilIcons) {
      expect(countSmilAnimations(icon.markup), icon.id).toBeGreaterThan(0);
    }
  });

  it('no CSS-technique icon accidentally contains a SMIL animate element', () => {
    for (const icon of cssIcons) {
      expect(countSmilAnimations(icon.markup), icon.id).toBe(0);
    }
  });
});

describe('markup well-formedness', () => {
  it.each(ICONS.map((icon) => [icon.id, icon]))('icon "%s" markup is a single svg root tagged with its own id', (_id, icon) => {
    expect(icon.markup.startsWith(`<svg id="${icon.id}"`)).toBe(true);
    expect(icon.markup.trim().endsWith('</svg>')).toBe(true);
  });

  it.each(ICONS.map((icon) => [icon.id, icon]))('icon "%s" markup is decorative (aria-hidden) and uses currentColor stroke', (_id, icon) => {
    expect(icon.markup).toContain('aria-hidden="true"');
    expect(icon.markup).toContain('stroke="currentColor"');
  });

  it('every animateMotion SMIL icon references an mpath that exists in the same markup', () => {
    for (const icon of ICONS) {
      const mpathMatch = icon.markup.match(/<mpath href="#([^"]+)"/);
      if (!mpathMatch) continue;
      const referencedId = mpathMatch[1];
      expect(icon.markup, `${icon.id} references #${referencedId}`).toContain(`id="${referencedId}"`);
    }
  });
});

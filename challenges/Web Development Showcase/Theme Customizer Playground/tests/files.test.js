import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import { exportCss, exportTailwind, exportTokens } from '../src/exporters.js';
import { MAX_BYTES, parseThemeFile } from '../src/importers.js';
import { INPUTS, defaults } from '../src/schema.js';
import { coreCss, themeArb } from './helpers.js';

/** Round trips lose at most the 4 decimals the files keep (components) or the OKLCH of a 4-decimal value. */
const same = (a, b) => {
  for (const i of INPUTS) {
    if (i.kind === 'font') expect(b[i.key], i.key).toBe(a[i.key]);
    else expect(b[i.key], i.key).toBeCloseTo(a[i.key], 4);
  }
};

describe('export then import is lossless', () => {
  it('CSS: any valid theme comes back', () => {
    fc.assert(
      fc.property(themeArb, (theme) => {
        const r = parseThemeFile(exportCss(theme, coreCss));
        expect(r.ok, r.errors?.join()).toBe(true);
        same(theme, r.theme);
      }),
      { numRuns: 200 },
    );
  });

  it('design tokens: any valid theme comes back', () => {
    fc.assert(
      fc.property(themeArb, (theme) => {
        const r = parseThemeFile(exportTokens(theme));
        expect(r.ok, r.errors?.join()).toBe(true);
        same(theme, r.theme);
      }),
      { numRuns: 200 },
    );
  });
});

describe('the exported files', () => {
  it('CSS is self-contained: inputs, typed @property, derivation, and no leftover app selector', () => {
    const css = exportCss(defaults(), coreCss);
    expect(css).toMatch(/^\/\*.*\*\/\n:root \{\n {2}--brand-l: 0\.55;/s);
    expect(css).toContain('@property --brand-h');
    expect(css).toContain('light-dark(');
    expect(css).toContain('oklch(from var(--brand)');
    expect(css).not.toContain('theme-scope');
    expect(css).not.toContain('data-scheme');
  });

  it('tokens follow the W3C Design Tokens shapes', () => {
    const tree = JSON.parse(exportTokens(defaults()));
    const brand = tree.theme.color['brand-light'];
    expect(brand.$type).toBe('color');
    expect(brand.$value).toMatchObject({ colorSpace: 'oklch', components: [0.55, 0.15, 264], alpha: 1 });
    expect(brand.$value.hex).toMatch(/^#[0-9a-f]{6}$/);
    expect(tree.theme.radius).toEqual({ $type: 'dimension', $value: { value: 10, unit: 'px' } });
    expect(tree.theme.typography['body-font']).toEqual({ $type: 'fontFamily', $value: ['system-ui', '-apple-system', 'sans-serif'] });
    expect(tree.theme.shadow.$value.offsetY).toEqual({ value: 6, unit: 'px' });
    expect(Object.keys(tree.palette.brand)).toHaveLength(10);
    // every leaf has a $type and a $value
    const leaves = [];
    const walk = (n) => (n.$value !== undefined ? leaves.push(n) : Object.values(n).forEach((v) => typeof v === 'object' && walk(v)));
    walk(tree.theme);
    walk(tree.palette);
    expect(leaves.length).toBeGreaterThan(40);
    for (const leaf of leaves) expect(leaf.$type).toBeTruthy();
  });

  it('tailwind is one @theme block with the variables Tailwind v4 reads', () => {
    const tw = exportTailwind(defaults());
    expect(tw.match(/@theme \{/g)).toHaveLength(1);
    for (const v of ['--color-brand-500:', '--color-surface-dark:', '--font-heading:', '--text-2xl:', '--radius-md: 10px', '--spacing: 4px', '--shadow-card:']) expect(tw).toContain(v);
    expect(tw).toContain('--color-brand-500: oklch(0.62 0.15 264);');
  });

  it('palette values in the tokens file are ignored on import', () => {
    const tree = JSON.parse(exportTokens(defaults()));
    tree.palette.brand['500'].$value.components = [0.1, 0.1, 0.1];
    expect(parseThemeFile(JSON.stringify(tree)).theme).toEqual(defaults());
  });
});

describe('hostile CSS is read for values only', () => {
  it('ignores selectors, other properties, url(), @import and comments, and still reads the real values', () => {
    const css = `/* --radius: 99px; */
      @import url(https://evil.example/x.css);
      body { background: url(javascript:alert(1)); display: none !important }
      :root { --radius: 12px; --brand-h: 120; content: "}"; --unknown: 1 }`;
    const r = parseThemeFile(css);
    expect(r.ok).toBe(true);
    expect(r.theme).toEqual({ ...defaults(), radius: 12, brandH: 120 });
  });

  it('a value that tries to break out is an error, not applied', () => {
    for (const evil of ['0.5; } body { display: none', '0.5 !important', 'calc(1 + 1)', 'var(--x)', 'url(x)', "0.5'", '1e3', '--brand-c']) {
      const r = parseThemeFile(`:root { --brand-c: ${evil} }`);
      expect(r.ok, evil).toBe(false);
    }
  });

  it('wrong or missing units, and out-of-range numbers, are errors', () => {
    for (const css of [':root{--radius:10}', ':root{--radius:10em}', ':root{--brand-h:10px}', ':root{--radius:40px}', ':root{--type-ratio:9}']) {
      expect(parseThemeFile(css).ok, css).toBe(false);
    }
    expect(parseThemeFile(':root{--radius:0}').ok).toBe(true);
  });

  it('fonts must be an exact allow-listed stack', () => {
    expect(parseThemeFile(':root{--font-body: ui-serif, Georgia, serif}').theme.bodyFont).toBe('serif');
    expect(parseThemeFile(':root{--font-body: Papyrus, serif}').ok).toBe(false);
    expect(parseThemeFile(':root{--font-body: system-ui, -apple-system, sans-serif; --font-heading: url(x)}').ok).toBe(false);
  });

  it('a file with no theme values, an empty file and an oversized file are refused', () => {
    expect(parseThemeFile('body { color: red }').ok).toBe(false);
    expect(parseThemeFile('   ').ok).toBe(false);
    expect(parseThemeFile(`:root{--radius:1px}${' '.repeat(MAX_BYTES)}`).ok).toBe(false);
    expect(parseThemeFile(undefined).ok).toBe(false);
  });

  it('never half-applies: one bad value rejects the whole file', () => {
    const r = parseThemeFile(':root{--radius: 12px; --brand-h: 9999}');
    expect(r.ok).toBe(false);
    expect(r.theme).toBeUndefined();
  });
});

describe('hostile JSON', () => {
  const tokens = (patch) => {
    const tree = JSON.parse(exportTokens(defaults()));
    patch(tree);
    return JSON.stringify(tree);
  };

  it('rejects invalid JSON, and JSON with no theme tokens', () => {
    expect(parseThemeFile('{ nope').ok).toBe(false);
    expect(parseThemeFile('{}').ok).toBe(false);
    expect(parseThemeFile('{"theme": 5}').ok).toBe(false);
  });

  it('rejects wrong shapes and units instead of guessing', () => {
    expect(parseThemeFile(tokens((t) => (t.theme.radius.$value.unit = 'rem'))).ok).toBe(false);
    expect(parseThemeFile(tokens((t) => (t.theme.radius.$value = '10px'))).ok).toBe(false);
    expect(parseThemeFile(tokens((t) => (t.theme.color['brand-light'].$value.colorSpace = 'srgb'))).ok).toBe(false);
    expect(parseThemeFile(tokens((t) => (t.theme.color['brand-light'].$value.components = [0.5, 'none', 3]))).ok).toBe(false);
    expect(parseThemeFile(tokens((t) => (t.theme.typography['body-font'].$value = ['Papyrus']))).ok).toBe(false);
    expect(parseThemeFile(tokens((t) => (t.theme.radius = null))).ok).toBe(false);
  });

  it('is not fooled by __proto__ or constructor keys', () => {
    const r = parseThemeFile('{"__proto__": {"theme": {"radius": {"$value": {"value": 1, "unit": "px"}}}}, "constructor": {"x": 1}, "theme": {"radius": {"$value": {"value": 12, "unit": "px"}}}}');
    expect(r.ok).toBe(true);
    expect(r.theme.radius).toBe(12);
    expect({}.radius).toBeUndefined();
    expect(Object.getPrototypeOf(r.theme)).toBe(Object.prototype);
  });

  it('survives arbitrary text without throwing', () => {
    for (const arb of [fc.string(), fc.json()]) {
      fc.assert(
        fc.property(arb, (s) => {
          expect(typeof parseThemeFile(s).ok).toBe('boolean');
        }),
        { numRuns: 300 },
      );
    }
  });
});

// A second implementation of the derived tokens in theme.css. The browser computes them in CSS
// (relative colour syntax, color-mix, light-dark); this computes the same colours in JS, for the
// contrast checks, the exports, and the fallback for browsers without those CSS features. The
// end-to-end tests compare the two in a real Chrome.

import { mixOklch, norm, toHex } from './color.js';
import { fmt } from './schema.js';
import { measure } from './contrast.js';

export const SCHEMES = ['light', 'dark'];

/** Brand lightness at which the label flips from white to near-black. */
const LABEL_FLIP = 0.62;

/** Lightness and chroma share of each ramp step (`--brand-50` to `--brand-900`). */
export const RAMP = [
  [50, 0.97, 0.12],
  [100, 0.935, 0.25],
  [200, 0.87, 0.45],
  [300, 0.79, 0.7],
  [400, 0.7, 0.95],
  [500, 0.62, 1],
  [600, 0.54, 1],
  [700, 0.46, 0.9],
  [800, 0.38, 0.75],
  [900, 0.3, 0.6],
];

/** token key -> CSS custom property, for every derived colour. */
export const COLOR_TOKENS = {
  brand: '--brand',
  brandHover: '--brand-hover',
  brandSubtle: '--brand-subtle',
  onBrand: '--on-brand',
  accent: '--accent',
  bg: '--bg',
  surface: '--surface',
  text: '--text',
  muted: '--muted',
  border: '--border',
};

export function deriveColors(theme, scheme) {
  const dark = scheme === 'dark';
  const brand = { l: dark ? theme.brandLDark : theme.brandL, c: theme.brandC, h: theme.brandH };
  const tint = (l) => ({ l, c: theme.neutralC, h: theme.neutralH });
  return {
    brand,
    brandHover: mixOklch(brand, { l: 0, c: 0, h: 0 }, 14),
    brandSubtle: dark ? { l: 0.28, c: brand.c * 0.4, h: brand.h } : { l: 0.97, c: brand.c * 0.2, h: brand.h },
    onBrand: { l: brand.l < LABEL_FLIP ? 1 : 0.18, c: 0, h: brand.h },
    accent: { ...brand, h: norm(brand.h + theme.accentShift) },
    bg: tint(dark ? 0.17 : 0.975),
    surface: tint(dark ? 0.22 : 1),
    text: tint(dark ? 0.95 : 0.22),
    muted: tint(dark ? 0.72 : 0.5),
    border: tint(dark ? 0.32 : 0.9),
  };
}

export const ramp = (theme) => RAMP.map(([step, l, k]) => ({ step, color: { l, c: theme.brandC * k, h: theme.brandH } }));

/** Resolved values for the browsers that cannot compute them: hex colours and a shadow string, nothing else. */
export function fallbackStyle(theme, scheme) {
  const out = {};
  const colors = deriveColors(theme, scheme);
  for (const [key, css] of Object.entries(COLOR_TOKENS)) out[css] = toHex(colors[key]);
  for (const { step, color } of ramp(theme)) out[`--brand-${step}`] = toHex(color);
  out['--shadow'] = `0 ${fmt(theme.shadowY)}px ${fmt(theme.shadowBlur)}px rgb(0 0 0 / ${fmt(theme.shadowAlpha)})`;
  return out;
}

const SIZE_NAMES = ['sm', 'base', 'lg', 'xl', '2xl', '3xl', '4xl'];
export const typeScale = (theme) => SIZE_NAMES.map((name, i) => ({ name, px: +(theme.typeBase * theme.typeRatio ** (i - 1)).toFixed(2) }));
export const spaceScale = (theme) => [1, 2, 3, 4, 6, 8].map((n) => ({ name: String(n), px: n * theme.space }));
export const radiusScale = (theme) => [['sm', 0.5], ['md', 1], ['lg', 1.5]].map(([name, k]) => ({ name, px: +(theme.radius * k).toFixed(2) }));

/** The pairs the contrast table checks. `fix` is the input that moves the pair, or null when none does. */
const PAIRS = [
  { id: 'text-bg', label: 'Text on page', fg: 'text', bg: 'bg', fix: null },
  { id: 'text-surface', label: 'Text on card', fg: 'text', bg: 'surface', fix: null },
  { id: 'muted-surface', label: 'Muted text on card', fg: 'muted', bg: 'surface', fix: null },
  { id: 'brand-surface', label: 'Brand link on card', fg: 'brand', bg: 'surface', fix: 'brand' },
  { id: 'accent-surface', label: 'Accent on card', fg: 'accent', bg: 'surface', fix: 'brand' },
  { id: 'onbrand-brand', label: 'Button label on brand', fg: 'onBrand', bg: 'brand', fix: 'brand' },
  { id: 'onbrand-hover', label: 'Button label on hover', fg: 'onBrand', bg: 'brandHover', fix: 'brand' },
  { id: 'brand-subtle', label: 'Brand on tinted chip', fg: 'brand', bg: 'brandSubtle', fix: 'brand' },
];

export const fixKey = (pair, scheme) => (pair.fix === 'brand' ? (scheme === 'dark' ? 'brandLDark' : 'brandL') : null);

export function contrastRows(theme) {
  return SCHEMES.flatMap((scheme) => {
    const colors = deriveColors(theme, scheme);
    return PAIRS.map((pair) => ({
      ...pair,
      scheme,
      fixKey: fixKey(pair, scheme),
      fgHex: toHex(colors[pair.fg]),
      bgHex: toHex(colors[pair.bg]),
      ...measure(colors[pair.fg], colors[pair.bg]),
    }));
  });
}

/** Does this pair pass `target` for `theme`? Used by the fix finder with a changed copy of the theme. */
export const pairPasses = (pair, scheme, target) => (theme) => {
  const colors = deriveColors(theme, scheme);
  return target.passes(measure(colors[pair.fg], colors[pair.bg]));
};

export const pairs = PAIRS;
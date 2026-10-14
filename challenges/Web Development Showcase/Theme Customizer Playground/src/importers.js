// Theme files are untrusted. Nothing here ever puts text from a file into a stylesheet or the DOM:
// a file is read only for the values of known inputs, each parsed strictly and then checked by
// `validate`. Everything else in it (selectors, other properties, url(), @import) is ignored.

import { FONTS, INPUTS, validate } from './schema.js';

export const MAX_BYTES = 100_000;

const NUMBER = /^(-?(?:\d+\.?\d*|\.\d+))(px)?$/;
const ABSENT = undefined;
const BAD = null; // present but malformed: `validate` reports it

function fontKey(names) {
  const joined = Array.isArray(names) ? names.join(', ') : '';
  return Object.keys(FONTS).find((k) => FONTS[k].join(', ') === joined) ?? BAD;
}

/** Read the inputs out of CSS custom-property declarations, wherever they sit in the file. */
export function readCss(text) {
  const raw = {};
  const byCss = new Map(INPUTS.map((i) => [i.css, i]));
  const clean = text.replace(/\/\*[\s\S]*?\*\//g, '');
  for (const [, name, value] of clean.matchAll(/(--[a-z0-9-]+)\s*:\s*([^;{}]*)(?=[;}]|$)/gi)) {
    const input = byCss.get(name);
    if (!input) continue;
    const v = value.trim();
    if (input.kind === 'font') raw[input.key] = fontKey(v.split(',').map((s) => s.trim()));
    else {
      const m = NUMBER.exec(v);
      const unitOk = m && (input.unit === 'px' ? m[2] === 'px' || +m[1] === 0 : m[2] === undefined);
      raw[input.key] = unitOk ? Number(m[1]) : BAD;
    }
  }
  return raw;
}

const node = (tree, path) => path.reduce((n, k) => (n !== null && typeof n === 'object' && Object.hasOwn(n, k) ? n[k] : ABSENT), tree);
const value = (tree, path) => {
  const t = node(tree, path);
  return t === ABSENT ? ABSENT : t !== null && typeof t === 'object' && Object.hasOwn(t, '$value') ? t.$value : BAD;
};
const px = (v) => (v === ABSENT ? ABSENT : v !== null && typeof v === 'object' && v.unit === 'px' ? v.value : BAD);
const oklch = (v, i) => (v === ABSENT ? ABSENT : v !== null && typeof v === 'object' && v.colorSpace === 'oklch' && Array.isArray(v.components) ? v.components[i] : BAD);

/** Read the inputs out of a W3C design-tokens file, from the `theme` group only (`palette` is computed). */
export function readTokens(tree) {
  const t = (...path) => value(tree, ['theme', ...path]);
  const shadow = t('shadow');
  const shadowPart = (name) => (shadow === ABSENT ? ABSENT : shadow !== null && typeof shadow === 'object' ? shadow[name] : BAD);
  const family = (v) => (v === ABSENT ? ABSENT : fontKey(v));
  const raw = {
    brandL: oklch(t('color', 'brand-light'), 0),
    brandC: oklch(t('color', 'brand-light'), 1),
    brandH: oklch(t('color', 'brand-light'), 2),
    brandLDark: oklch(t('color', 'brand-dark'), 0),
    accentShift: t('color', 'accent-hue-shift'),
    neutralH: t('color', 'neutral-hue'),
    neutralC: t('color', 'neutral-chroma'),
    radius: px(t('radius')),
    space: px(t('space')),
    shadowY: px(shadowPart('offsetY')),
    shadowBlur: px(shadowPart('blur')),
    shadowAlpha: (() => {
      const c = shadowPart('color');
      return c === ABSENT ? ABSENT : c !== null && typeof c === 'object' ? c.alpha : BAD;
    })(),
    typeBase: px(t('typography', 'base-size')),
    typeRatio: t('typography', 'scale-ratio'),
    headingWeight: t('typography', 'heading-weight'),
    bodyFont: family(t('typography', 'body-font')),
    headingFont: family(t('typography', 'heading-font')),
  };
  return Object.fromEntries(Object.entries(raw).filter(([, v]) => v !== ABSENT));
}

/** Parse a pasted or uploaded theme file (CSS or design-tokens JSON). `{ ok, theme, errors, ignored, format }`. */
export function parseThemeFile(text, base) {
  if (typeof text !== 'string' || !text.trim()) return { ok: false, errors: ['The file is empty.'], format: null };
  if (text.length > MAX_BYTES) return { ok: false, errors: [`The file is larger than ${MAX_BYTES / 1000} kB.`], format: null };
  let raw;
  let format;
  if (text.trimStart().startsWith('{')) {
    format = 'tokens';
    try {
      raw = readTokens(JSON.parse(text));
    } catch {
      return { ok: false, errors: ['The file is not valid JSON.'], format };
    }
  } else {
    format = 'css';
    raw = readCss(text);
  }
  if (!Object.keys(raw).length) return { ok: false, errors: [`No theme values found in this ${format === 'css' ? 'CSS' : 'JSON'} file.`], format };
  return { ...validate(raw, base), format };
}

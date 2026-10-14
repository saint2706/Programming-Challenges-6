// The one list of what a theme is. Everything else (UI, CSS, exports, import, share links, storage)
// is driven from it, and every value that reaches the DOM goes through `validate` first.

/** Font stacks are an allow-list: a theme stores a key, never a CSS string. System fonts only, so nothing downloads. */
export const FONTS = {
  system: ['system-ui', '-apple-system', 'sans-serif'],
  serif: ['ui-serif', 'Georgia', 'serif'],
  mono: ['ui-monospace', 'Consolas', 'monospace'],
  rounded: ['ui-rounded', 'system-ui', 'sans-serif'],
};

const num = (key, css, group, label, min, max, step, def, unit = '') => ({ key, css, group, label, kind: 'number', min, max, step, default: def, unit });
const font = (key, css, label, def) => ({ key, css, group: 'Type', label, kind: 'font', default: def });

export const INPUTS = [
  num('brandL', '--brand-l', 'Color', 'Brand lightness (light mode)', 0.2, 0.95, 0.005, 0.55),
  num('brandLDark', '--brand-l-dark', 'Color', 'Brand lightness (dark mode)', 0.2, 0.95, 0.005, 0.7),
  num('brandC', '--brand-c', 'Color', 'Brand chroma', 0, 0.37, 0.005, 0.15),
  num('brandH', '--brand-h', 'Color', 'Brand hue', 0, 360, 1, 264),
  num('accentShift', '--accent-shift', 'Color', 'Accent hue shift', -180, 180, 1, 150),
  num('neutralH', '--neutral-h', 'Color', 'Neutral tint hue', 0, 360, 1, 264),
  num('neutralC', '--neutral-c', 'Color', 'Neutral tint chroma', 0, 0.04, 0.001, 0.012),
  num('radius', '--radius', 'Shape and space', 'Corner radius', 0, 32, 1, 10, 'px'),
  num('space', '--space', 'Shape and space', 'Spacing unit', 2, 8, 0.5, 4, 'px'),
  num('shadowY', '--shadow-y', 'Shape and space', 'Shadow offset', 0, 24, 1, 6, 'px'),
  num('shadowBlur', '--shadow-blur', 'Shape and space', 'Shadow blur', 0, 48, 1, 18, 'px'),
  num('shadowAlpha', '--shadow-alpha', 'Shape and space', 'Shadow opacity', 0, 0.5, 0.01, 0.12),
  num('typeBase', '--type-base', 'Type', 'Base text size', 14, 20, 0.5, 16, 'px'),
  num('typeRatio', '--type-ratio', 'Type', 'Type scale ratio', 1.1, 1.6, 0.01, 1.25),
  num('headingWeight', '--heading-weight', 'Type', 'Heading weight', 400, 800, 100, 700),
  font('bodyFont', '--font-body', 'Body font', 'system'),
  font('headingFont', '--font-heading', 'Heading font', 'system'),
];

export const defaults = () => Object.fromEntries(INPUTS.map((i) => [i.key, i.default]));

/** Four decimals is finer than any control step and keeps CSS and URLs short. */
export const fmt = (n) => String(+n.toFixed(4));

/** The CSS text for one input. Built only from validated numbers and allow-listed stacks. */
export function cssValue(input, value) {
  return input.kind === 'font' ? FONTS[value].join(', ') : fmt(value) + input.unit;
}

/**
 * Check a candidate theme (any object, e.g. parsed from a file) against the schema. Missing keys keep
 * `base`; unknown keys are counted and dropped; a wrong type or out-of-range value is an error.
 * Never half-applies: the caller gets either `ok` with a whole theme or a list of errors.
 */
export function validate(raw, base = defaults()) {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) return { ok: false, errors: ['The theme is not an object.'], ignored: 0, found: 0 };
  const errors = [];
  const theme = { ...base };
  let found = 0;
  for (const input of INPUTS) {
    if (!Object.hasOwn(raw, input.key)) continue;
    found++;
    const v = raw[input.key];
    if (input.kind === 'font') {
      if (typeof v === 'string' && Object.hasOwn(FONTS, v)) theme[input.key] = v;
      else errors.push(`${input.label}: expected one of ${Object.keys(FONTS).join(', ')}.`);
    } else if (typeof v !== 'number' || !Number.isFinite(v)) errors.push(`${input.label}: expected a number.`);
    else if (v < input.min || v > input.max) errors.push(`${input.label}: ${v} is outside ${input.min} to ${input.max}.`);
    else theme[input.key] = v;
  }
  const known = new Set(INPUTS.map((i) => i.key));
  const ignored = Object.keys(raw).filter((k) => !known.has(k)).length;
  return errors.length ? { ok: false, errors, ignored, found } : { ok: true, theme, errors, ignored, found };
}

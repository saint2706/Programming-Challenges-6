import { toHex } from './color.js';
import { COLOR_TOKENS, deriveColors, radiusScale, ramp, spaceScale, typeScale } from './derive.js';
import { FONTS, INPUTS, cssValue, fmt } from './schema.js';

const oklchText = (c) => `oklch(${fmt(c.l)} ${fmt(c.c)} ${fmt(c.h)})`;
const stack = (key) => FONTS[key].join(', ');

/**
 * A drop-in stylesheet: the inputs on `:root`, then the same derivation theme.css runs in the app
 * (typed @property inputs, relative colours, light-dark), so the file works on its own.
 */
export function exportCss(theme, coreCss) {
  const core = coreCss.split('\n.theme-scope[data-scheme')[0].trimEnd().replaceAll('.theme-scope', ':root');
  const inputs = INPUTS.map((i) => `  ${i.css}: ${cssValue(i, theme[i.key])};`).join('\n');
  return `/* Theme Customizer Playground: the inputs are the :root block below; the rest is computed from them. */\n:root {\n${inputs}\n}\n\n${core}\n`;
}

// W3C Design Tokens Format Module (2025.10): colours as {colorSpace, components}, dimensions as {value, unit}.
const dim = (px) => ({ value: +px.toFixed(4), unit: 'px' });
const colorToken = (c) => ({
  $type: 'color',
  $value: { colorSpace: 'oklch', components: [+c.l.toFixed(4), +c.c.toFixed(4), +c.h.toFixed(4)], alpha: 1, hex: toHex(c) },
});
const number = (value) => ({ $type: 'number', $value: value });
const dimension = (px) => ({ $type: 'dimension', $value: dim(px) });

/** `theme` holds the inputs (this is what import reads); `palette` is computed, for tools that want the results. */
export function exportTokens(theme) {
  const light = deriveColors(theme, 'light');
  const dark = deriveColors(theme, 'dark');
  const semantic = (colors) => Object.fromEntries(Object.keys(COLOR_TOKENS).map((k) => [k, colorToken(colors[k])]));
  const named = (list) => Object.fromEntries(list.map(({ name, px }) => [name, dimension(px)]));
  return JSON.stringify(
    {
      $description: 'Exported from Theme Customizer Playground. "theme" holds the editable inputs; "palette" is computed from them and ignored on import.',
      theme: {
        color: {
          'brand-light': colorToken(light.brand),
          'brand-dark': colorToken(dark.brand),
          'accent-hue-shift': number(theme.accentShift),
          'neutral-hue': number(theme.neutralH),
          'neutral-chroma': number(theme.neutralC),
        },
        radius: dimension(theme.radius),
        space: dimension(theme.space),
        typography: {
          'base-size': dimension(theme.typeBase),
          'scale-ratio': number(theme.typeRatio),
          'heading-weight': { $type: 'fontWeight', $value: theme.headingWeight },
          'body-font': { $type: 'fontFamily', $value: FONTS[theme.bodyFont] },
          'heading-font': { $type: 'fontFamily', $value: FONTS[theme.headingFont] },
        },
        shadow: {
          $type: 'shadow',
          $value: {
            color: { colorSpace: 'srgb', components: [0, 0, 0], alpha: theme.shadowAlpha },
            offsetX: dim(0),
            offsetY: dim(theme.shadowY),
            blur: dim(theme.shadowBlur),
            spread: dim(0),
          },
        },
      },
      palette: {
        brand: Object.fromEntries(ramp(theme).map(({ step, color }) => [String(step), colorToken(color)])),
        light: semantic(light),
        dark: semantic(dark),
        'type-scale': named(typeScale(theme)),
        space: named(spaceScale(theme)),
        radius: named(radiusScale(theme)),
      },
    },
    null,
    2,
  ) + '\n';
}

/** Tailwind CSS v4 theme variables. Resolved values, since Tailwind does not evaluate the CSS derivation. */
export function exportTailwind(theme) {
  const light = deriveColors(theme, 'light');
  const dark = deriveColors(theme, 'dark');
  const lines = [];
  for (const { step, color } of ramp(theme)) lines.push(`  --color-brand-${step}: ${oklchText(color)};`);
  lines.push(`  --color-accent: ${oklchText(light.accent)};`, `  --color-accent-dark: ${oklchText(dark.accent)};`);
  for (const key of ['bg', 'surface', 'text', 'muted', 'border']) {
    lines.push(`  --color-${key}: ${oklchText(light[key])};`, `  --color-${key}-dark: ${oklchText(dark[key])};`);
  }
  lines.push(`  --font-sans: ${stack(theme.bodyFont)};`, `  --font-heading: ${stack(theme.headingFont)};`);
  for (const { name, px } of typeScale(theme)) lines.push(`  --text-${name}: ${px}px;`);
  for (const { name, px } of radiusScale(theme)) lines.push(`  --radius-${name}: ${px}px;`);
  lines.push(`  --spacing: ${fmt(theme.space)}px;`);
  lines.push(`  --shadow-card: 0 ${fmt(theme.shadowY)}px ${fmt(theme.shadowBlur)}px rgb(0 0 0 / ${fmt(theme.shadowAlpha)});`);
  return `/* Tailwind CSS v4. Light values are the defaults; the -dark colours are for dark: variants. */\n@theme {\n${lines.join('\n')}\n}\n`;
}

export const FORMATS = [
  { id: 'css', label: 'CSS variables', file: 'theme.css', mime: 'text/css', build: (theme, coreCss) => exportCss(theme, coreCss) },
  { id: 'tokens', label: 'Design tokens (W3C JSON)', file: 'theme.tokens.json', mime: 'application/json', build: (theme) => exportTokens(theme) },
  { id: 'tailwind', label: 'Tailwind v4 @theme', file: 'theme.tailwind.css', mime: 'text/css', build: (theme) => exportTailwind(theme) },
];

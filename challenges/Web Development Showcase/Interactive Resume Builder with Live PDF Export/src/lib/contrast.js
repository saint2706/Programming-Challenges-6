/** WCAG 2.x contrast math, used to warn when the accent colour is too pale for text. */

const HEX = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i;

/** `#rgb` or `#rrggbb` to `[r, g, b]` (0-255). Throws on anything else. */
export function parseHex(value) {
  const match = typeof value === 'string' ? HEX.exec(value) : null;
  if (!match) throw new Error(`invalid hex color: ${JSON.stringify(value)}`);
  let digits = match[1];
  if (digits.length === 3) digits = [...digits].map((d) => d + d).join('');
  return [0, 2, 4].map((i) => parseInt(digits.slice(i, i + 2), 16));
}

function relativeLuminance(hex) {
  const [r, g, b] = parseHex(hex).map((channel) => {
    const c = channel / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** Contrast ratio between two colours, 1 (identical) to 21 (black on white). */
export function contrastRatio(a, b) {
  const [light, dark] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x);
  return (light + 0.05) / (dark + 0.05);
}

/** True when `foreground` on `background` meets WCAG AA for normal-size text (4.5:1). */
export function meetsAA(foreground, background = '#ffffff') {
  return contrastRatio(foreground, background) >= 4.5;
}

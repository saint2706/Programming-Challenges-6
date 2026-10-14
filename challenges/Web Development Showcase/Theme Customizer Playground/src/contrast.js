import { toRgb8 } from './color.js';

/** WCAG 2.x relative luminance and contrast ratio, on 8-bit sRGB. */
const channel = (v) => {
  const s = v / 255;
  return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
};
export const luminance = ([r, g, b]) => 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);

export function wcagRatio(fg, bg) {
  const [hi, lo] = [luminance(fg), luminance(bg)].sort((a, b) => b - a);
  return (hi + 0.05) / (lo + 0.05);
}

/** APCA-W3 0.0.98G-4g (the constants of the published reference). Positive Lc is dark text on light, negative the reverse. */
const apcaY = ([r, g, b]) => 0.2126729 * (r / 255) ** 2.4 + 0.7151522 * (g / 255) ** 2.4 + 0.072175 * (b / 255) ** 2.4;
const soften = (y) => (y > 0.022 ? y : y + (0.022 - y) ** 1.414);

export function apca(fg, bg) {
  const yt = soften(apcaY(fg));
  const yb = soften(apcaY(bg));
  if (Math.abs(yb - yt) < 0.0005) return 0;
  if (yb > yt) {
    const s = (yb ** 0.56 - yt ** 0.57) * 1.14;
    return s < 0.1 ? 0 : (s - 0.027) * 100;
  }
  const s = (yb ** 0.65 - yt ** 0.62) * 1.14;
  return s > -0.1 ? 0 : (s + 0.027) * 100;
}

export const wcagLevel = (r) => (r >= 7 ? 'AAA' : r >= 4.5 ? 'AA' : r >= 3 ? 'AA large' : 'Fail');
export const apcaLevel = (lc) => {
  const a = Math.abs(lc);
  return a >= 75 ? 'Body text' : a >= 60 ? 'Content' : a >= 45 ? 'Large text' : a >= 30 ? 'Spot text' : 'Fail';
};

/** What "passing" means. The fix finder and the table both use it. */
export const TARGETS = [
  { id: 'aa', label: 'WCAG AA (4.5:1)', metric: 'wcag', min: 4.5 },
  { id: 'aaa', label: 'WCAG AAA (7:1)', metric: 'wcag', min: 7 },
  { id: 'apca60', label: 'APCA Lc 60', metric: 'apca', min: 60 },
  { id: 'apca75', label: 'APCA Lc 75', metric: 'apca', min: 75 },
].map((t) => ({ ...t, passes: (m) => Math.abs(m[t.metric]) >= t.min }));

/** Contrast of two OKLCH colours as painted (clipped to sRGB, rounded to 8 bits). */
export function measure(fg, bg) {
  const f = toRgb8(fg);
  const b = toRgb8(bg);
  return { wcag: wcagRatio(f, b), apca: apca(f, b) };
}

/**
 * The value of one numeric input nearest to the current one for which `passes(theme)` is true, found
 * by stepping outward both ways. Null when nothing in range passes, or when it already passes.
 */
export function nearestPassing(theme, input, passes) {
  if (passes(theme)) return null;
  const cur = theme[input.key];
  const ok = (v) => v >= input.min - 1e-9 && v <= input.max + 1e-9 && passes({ ...theme, [input.key]: v });
  const reach = Math.ceil((input.max - input.min) / input.step) + 1;
  for (let k = 1; k <= reach; k++) {
    const down = +(cur - k * input.step).toFixed(4);
    const up = +(cur + k * input.step).toFixed(4);
    const [downOk, upOk] = [ok(down), ok(up)];
    if (downOk || upOk) return upOk && !downOk ? up : down;
  }
  return null;
}

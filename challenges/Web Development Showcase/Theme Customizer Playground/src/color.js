// OKLCH <-> sRGB, from Björn Ottosson's published matrices (https://bottosson.github.io/posts/oklab/).
// A colour is a plain {l, c, h} object: l in 0..1, c >= 0, h in degrees.

const toLinear = (v) => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
const toGamma = (v) => (v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055);

export const norm = (h) => ((h % 360) + 360) % 360;

export function oklchToLinear({ l, c, h }) {
  const a = c * Math.cos((h * Math.PI) / 180);
  const b = c * Math.sin((h * Math.PI) / 180);
  const l_ = (l + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m_ = (l - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s_ = (l - 0.0894841775 * a - 1.291485548 * b) ** 3;
  return [
    4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.707614701 * s_,
  ];
}

export function linearToOklch([r, g, b]) {
  const l_ = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m_ = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s_ = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  const l = 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_;
  const a = 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_;
  const bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_;
  const c = Math.hypot(a, bb);
  return { l, c, h: c < 1e-7 ? 0 : norm((Math.atan2(bb, a) * 180) / Math.PI) };
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const inGamut = (rgb, eps) => rgb.every((v) => v >= -eps && v <= 1 + eps);

/** The sRGB colour a browser paints for an OKLCH colour: out-of-gamut chroma is clipped per channel. */
export function toRgb(color) {
  return oklchToLinear(color).map((v) => toGamma(clamp01(v)));
}

/** False when the colour is outside sRGB, so `toRgb` has to clip it (what Chrome measurably does). */
export const isInGamut = (color) => inGamut(oklchToLinear(color), 0.0005);

export const toRgb8 = (color) => toRgb(color).map((v) => Math.round(v * 255));

export const toHex = (color) => `#${toRgb8(color).map((v) => v.toString(16).padStart(2, '0')).join('')}`;

/** `#rgb` or `#rrggbb` -> OKLCH, or null when it is not a hex colour. */
export function hexToOklch(text) {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(text).trim());
  if (!m) return null;
  const hex = m[1].length === 3 ? [...m[1]].map((d) => d + d).join('') : m[1];
  const lin = [0, 2, 4].map((i) => toLinear(parseInt(hex.slice(i, i + 2), 16) / 255));
  return linearToOklch(lin);
}

/** `color-mix(in oklch, a, b p%)`: a straight lerp of l, c and h (shorter arc); an achromatic side lends its hue. */
export function mixOklch(a, b, percentB) {
  const t = percentB / 100;
  const aGray = a.c < 1e-6;
  const bGray = b.c < 1e-6;
  let h;
  if (aGray && bGray) h = 0;
  else if (aGray) h = b.h;
  else if (bGray) h = a.h;
  else {
    let d = b.h - a.h;
    if (d > 180) d -= 360;
    else if (d < -180) d += 360;
    h = norm(a.h + d * t);
  }
  return { l: a.l + (b.l - a.l) * t, c: a.c + (b.c - a.c) * t, h };
}

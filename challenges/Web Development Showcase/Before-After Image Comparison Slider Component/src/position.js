// Pure helpers: no DOM, so they are unit tested without a browser.

/** Clamp to the 0..100 slider range. */
export const clamp = (n) => Math.min(100, Math.max(0, n));

const round = (n) => Math.round(n * 1000) / 1000;

/** "40", "40%", 40 -> 40. Anything that is not a finite number -> `fallback`. */
export function parsePosition(value, fallback = 50) {
  const n = typeof value === 'number' ? value : parseFloat(value);
  return Number.isFinite(n) ? clamp(n) : fallback;
}

/**
 * Where a pointer at (x, y) puts the divider, as a percentage of the box measured from the "start"
 * edge: left (right in RTL) for horizontal, top for vertical. `null` if the box has no size.
 * @param {{left:number, top:number, width:number, height:number}} rect
 * @param {number} x
 * @param {number} y
 * @param {{vertical?: boolean, rtl?: boolean}} [mode]
 */
export function positionFromPoint(rect, x, y, { vertical = false, rtl = false } = {}) {
  const size = vertical ? rect.height : rect.width;
  if (!(size > 0)) return null;
  const fraction = vertical ? (y - rect.top) / size : (x - rect.left) / size;
  return clamp((rtl && !vertical ? 1 - fraction : fraction) * 100);
}

/**
 * The position a key press leads to, or `null` when the key is not ours (so it is not swallowed).
 * Arrows move the divider in the direction they point (flipped in RTL). Up/Right also mean "more"
 * on a horizontal slider and Right/Left on a vertical one, as the ARIA slider pattern asks.
 * @param {string} key
 * @param {number} current
 * @param {{vertical?: boolean, rtl?: boolean, step?: number, bigStep?: number, shift?: boolean}} [mode]
 * @returns {number | null}
 */
export function keyTarget(key, current, { vertical = false, rtl = false, step = 1, bigStep = 10, shift = false } = {}) {
  if (key === 'Home') return 0;
  if (key === 'End') return 100;
  if (key === 'PageUp') return round(clamp(current + bigStep));
  if (key === 'PageDown') return round(clamp(current - bigStep));
  let sign;
  if (vertical) {
    sign = { ArrowDown: 1, ArrowUp: -1, ArrowRight: 1, ArrowLeft: -1 }[key];
  } else {
    const right = rtl ? -1 : 1;
    sign = { ArrowRight: right, ArrowLeft: -right, ArrowUp: 1, ArrowDown: -1 }[key];
  }
  return sign ? round(clamp(current + sign * (shift ? bigStep : step))) : null;
}

/** Human text for aria-valuetext, e.g. "40% Before, 60% After". */
export function valueText(position, beforeLabel, afterLabel) {
  const p = Math.round(position);
  return `${p}% ${beforeLabel}, ${100 - p}% ${afterLabel}`;
}

/** "16/9", "16 / 9" or "1.5" -> a CSS aspect-ratio value; `null` for anything else. */
export function parseRatio(value) {
  // Trim first: leading/trailing `\s*` inside the regex overlap and backtrack polynomially.
  const m = /^(\d+(?:\.\d+)?)(?:\s*\/\s*(\d+(?:\.\d+)?))?$/.exec(String(value ?? '').trim());
  if (!m) return null;
  const [w, h] = [Number(m[1]), Number(m[2] ?? 1)];
  return w > 0 && h > 0 ? `${w} / ${h}` : null;
}

/**
 * Only load images over http(s)/blob/file, relative URLs, or `data:image/*`. An `<img>` ignores
 * `javascript:` URLs anyway; this keeps the component from ever being the one to try one.
 * @param {string} value
 * @param {string} [base]
 */
export function isSafeImageUrl(value, base = 'http://localhost/') {
  let url;
  try {
    url = new URL(value, base);
  } catch {
    return false;
  }
  if (url.protocol === 'data:') return /^data:image\//i.test(value.trim());
  return ['http:', 'https:', 'blob:', 'file:'].includes(url.protocol);
}

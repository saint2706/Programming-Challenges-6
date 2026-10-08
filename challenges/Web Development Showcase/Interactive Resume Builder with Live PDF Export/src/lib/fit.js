import { SCALE_MAX, SCALE_MIN } from './schema.js';

/**
 * "Fit to N pages": the largest content scale (font size and spacing multiplier) at which the
 * resume still paginates to at most `target` pages. `pageCountAt(scale)` runs a real
 * measure-and-paginate pass, so it is the expensive part; a binary search keeps it to about seven
 * calls over the 0.82 to 1.15 range. Page count is assumed to be non-decreasing in scale.
 *
 * Returns `{ scale, pageCount, fits }`. When even the smallest scale needs more than `target`
 * pages, `fits` is false and the result is the smallest scale with the page count it reaches.
 */
export function fitToPages(
  target,
  pageCountAt,
  { min = SCALE_MIN, max = SCALE_MAX, step = 0.01 } = {},
) {
  if (!Number.isInteger(target) || target < 1) throw new RangeError('target must be a whole number of pages');
  const scaleOf = (i) => Number((i * step).toFixed(6));
  const lo = Math.round(min / step);
  const hi = Math.round(max / step);

  const floorCount = pageCountAt(scaleOf(lo));
  if (floorCount > target) return { scale: scaleOf(lo), pageCount: floorCount, fits: false };

  let best = lo;
  let bestCount = floorCount;
  let a = lo + 1;
  let b = hi;
  while (a <= b) {
    const mid = (a + b) >> 1;
    const count = pageCountAt(scaleOf(mid));
    if (count <= target) {
      best = mid;
      bestCount = count;
      a = mid + 1;
    } else {
      b = mid - 1;
    }
  }
  return { scale: scaleOf(best), pageCount: bestCount, fits: true };
}

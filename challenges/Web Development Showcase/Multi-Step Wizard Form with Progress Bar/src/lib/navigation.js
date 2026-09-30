import { STEPS } from './schemas.js';

export const STEP_COUNT = STEPS.length;

/** Index of the first step that has not been verified, capped at the last step. */
export function furthestReachable(verified, count = STEP_COUNT) {
  const set = new Set(verified);
  for (let i = 0; i < count; i++) if (!set.has(i)) return i;
  return count - 1;
}

/** Clamp a requested step into [0, furthestReachable]. Non-integers fall back to 0. */
export function clampStep(requested, verified, count = STEP_COUNT) {
  if (!Number.isInteger(requested)) return 0;
  return Math.max(0, Math.min(requested, furthestReachable(verified, count)));
}

/** "#/step/3" -> 2 (zero-based). Anything else -> null. */
export function parseHash(hash) {
  const m = /^#\/step\/(\d{1,3})$/.exec(hash ?? '');
  if (!m) return null;
  const n = Number(m[1]);
  return n >= 1 ? n - 1 : null;
}

export function formatHash(index) {
  return `#/step/${index + 1}`;
}

/** Progress as a whole percentage: verified steps out of all steps. */
export function progressPercent(verified, count = STEP_COUNT) {
  const done = new Set(verified.filter((i) => i >= 0 && i < count)).size;
  return Math.round((done / count) * 100);
}

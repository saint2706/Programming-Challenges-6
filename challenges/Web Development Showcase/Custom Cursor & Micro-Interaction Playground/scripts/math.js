/**
 * Pure, side-effect-free math helpers shared by the cursor/hover effects.
 * Kept dependency-free and DOM-free so they can be unit tested directly.
 */

/** Clamp `value` into the inclusive range [min, max]. */
export function clamp(value, min, max) {
  if (min > max) {
    [min, max] = [max, min];
  }
  return Math.min(Math.max(value, min), max);
}

/** Linear interpolation from `start` to `end` at position `t` (0..1, unclamped). */
export function lerp(start, end, t) {
  return start + (end - start) * t;
}

/**
 * Offset to nudge a magnetic element toward the cursor.
 *
 * @param {{x: number, y: number}} cursorPos - pointer position in the same
 *   coordinate space as `elementCenter` (e.g. both viewport coordinates).
 * @param {{x: number, y: number}} elementCenter - center of the magnetic element.
 * @param {number} radius - distance (px) beyond which the magnet has no effect.
 * @param {number} strength - 0..1 fraction of the raw offset actually applied.
 * @returns {{x: number, y: number}} translation to apply to the element;
 *   `{x: 0, y: 0}` when the cursor is outside `radius`.
 */
export function magneticOffset(cursorPos, elementCenter, radius, strength) {
  const dx = cursorPos.x - elementCenter.x;
  const dy = cursorPos.y - elementCenter.y;
  const distance = Math.hypot(dx, dy);

  if (radius <= 0 || distance > radius) {
    return { x: 0, y: 0 };
  }

  return { x: dx * strength, y: dy * strength };
}

/**
 * Tilt rotation angles for a "tilt on mouse move" card.
 *
 * @param {{x: number, y: number}} pointerPos - pointer position in viewport coordinates.
 * @param {{left: number, top: number, width: number, height: number}} rect -
 *   the card's bounding rect (e.g. from `getBoundingClientRect()`).
 * @param {number} maxTiltDegrees - maximum rotation magnitude in degrees.
 * @returns {{rotateX: number, rotateY: number}} rotation to apply; pointer at
 *   the rect's center yields `{0, 0}`, pointer at an edge yields
 *   `±maxTiltDegrees` on the corresponding axis.
 */
export function tiltAngles(pointerPos, rect, maxTiltDegrees) {
  if (rect.width <= 0 || rect.height <= 0) {
    return { rotateX: 0, rotateY: 0 };
  }

  const px = clamp((pointerPos.x - rect.left) / rect.width, 0, 1);
  const py = clamp((pointerPos.y - rect.top) / rect.height, 0, 1);

  // Normalize to -1..1 around the center, then invert Y so moving the
  // pointer toward the top tilts the card "up" (positive rotateX).
  const normX = px * 2 - 1;
  const normY = py * 2 - 1;

  return {
    // `+ 0` normalizes a `-0` result (e.g. at the exact center) to `0`, since
    // `-0` and `0` are visually identical but fail strict `toEqual` checks.
    rotateX: -normY * maxTiltDegrees + 0,
    rotateY: normX * maxTiltDegrees + 0,
  };
}

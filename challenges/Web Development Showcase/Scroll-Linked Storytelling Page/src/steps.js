// Which narrative step is "current". Input is what an IntersectionObserver reports for a line in the
// middle of the viewport; output is the id that should get aria-current.

/**
 * @param {{ id: string, isIntersecting: boolean }[]} entries every step, in DOM order
 * @param {string | null} current the id active before this update
 * @returns {string | null}
 */
export function pickActiveStep(entries, current) {
    const crossing = entries.filter((e) => e.isIntersecting);
    if (crossing.length === 0) return current;
    if (crossing.some((e) => e.id === current)) return current;
    return crossing[0].id;
}

/** Scroll progress (0 to 1) through `count` steps, mapped to the step index that owns it. */
export function stepForProgress(progress, count) {
    if (count < 1) throw new RangeError('count must be at least 1');
    const p = Number.isFinite(progress) ? Math.min(1, Math.max(0, progress)) : 0;
    return Math.min(count - 1, Math.floor(p * count));
}

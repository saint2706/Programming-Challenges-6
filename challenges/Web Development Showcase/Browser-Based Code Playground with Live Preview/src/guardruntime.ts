/** How long one synchronous run may last before the guard stops it. */
export const LOOP_LIMIT_MS = 2000;

/**
 * The function the guard calls put at the start of every loop body resolve to. It is written with no
 * free variables so it can be turned into source text with `toString()` and placed at the top of
 * each bundle (see `bundle.ts`). It lives in the bundle, not in the sandbox page, on purpose: an
 * error thrown from a script of another origin reaches `window.onerror` stripped of its message
 * ("Script error."), and the person needs to read why their loop was stopped.
 *
 * Reading the clock is the expensive part, so it is done as rarely as is safe. The guard looks at the
 * call count against a mask; when two looks were very close together (a tight loop) the mask widens,
 * so the clock is read every few thousand iterations, and when they were far apart (slow iterations)
 * it narrows again, so a stuck loop with a heavy body is still noticed within the allowance. A call
 * costs about one nanosecond in a tight loop.
 *
 * `defer` runs a callback once the current synchronous run is over (the next macrotask), which is
 * when a fresh run, and a fresh allowance, begins.
 */
export function createGuard(limit: number, now: () => number, defer: (fn: () => void) => void): () => void {
    let calls = 0;
    let mask = 15;
    let last = 0;
    let start = 0;
    let tripped = false;
    const message = () =>
        'Stopped a loop that ran for over ' + limit / 1000 + ' seconds without finishing. It is probably infinite. The playground watches loops in your code so that one bad loop cannot freeze the page.';
    const endOfRun = () => {
        start = 0;
        tripped = false;
    };
    return () => {
        if ((++calls & mask) !== 0) return;
        if (tripped) throw new Error(message());
        const t = now();
        const since = t - last;
        last = t;
        if (since < 4 && mask < 1023) mask = (mask << 1) | 1;
        else if (since > 40 && mask > 15) mask >>= 1;
        if (start === 0) {
            start = t;
            defer(endOfRun);
        } else if (t - start > limit) {
            // Stay tripped for the rest of this run, so code that catches the error and loops again still stops.
            tripped = true;
            throw new Error(message());
        }
    };
}

/** Source text that defines `__pgLoop` in a bundle. */
export const GUARD_RUNTIME = `var __pgLoop = (${createGuard.toString()})(${LOOP_LIMIT_MS}, () => performance.now(), (fn) => setTimeout(fn, 0));`;

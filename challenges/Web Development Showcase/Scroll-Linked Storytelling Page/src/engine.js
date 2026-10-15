// Which animation engine runs the page: native CSS scroll-driven animations or GSAP ScrollTrigger.
// The choice is remembered per browser; storage can be blocked or full, so every access is guarded.

export const ENGINES = ['css', 'gsap'];
export const STORAGE_KEY = 'scroll-story-engine';

/** True when the browser implements animation-timeline (native scroll-driven animations). */
export function cssTimelineSupported(css = globalThis.CSS) {
    try {
        return Boolean(css?.supports?.('animation-timeline', 'view()'));
    } catch {
        return false;
    }
}

/**
 * The engine to run. A stored "css" is ignored where native timelines are missing, so the page never
 * depends on an engine the browser cannot run. Without a usable choice: CSS if supported, else GSAP.
 */
export function resolveEngine({ stored = null, cssSupported }) {
    if (stored === 'gsap') return 'gsap';
    if (stored === 'css' && cssSupported) return 'css';
    return cssSupported ? 'css' : 'gsap';
}

/** The other engine, for the toggle. */
export function otherEngine(engine) {
    return engine === 'css' ? 'gsap' : 'css';
}

export function readStoredEngine(storage) {
    try {
        const value = storage?.getItem(STORAGE_KEY);
        return ENGINES.includes(value) ? value : null;
    } catch {
        return null; // blocked storage (private mode, disabled cookies) is the same as no choice
    }
}

/** Returns true if the choice was saved. */
export function writeStoredEngine(storage, engine) {
    try {
        storage.setItem(STORAGE_KEY, engine);
        return true;
    } catch {
        return false;
    }
}

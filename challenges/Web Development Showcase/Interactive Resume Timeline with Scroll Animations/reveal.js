/**
 * Scroll-reveal wiring.
 *
 * Native CSS Scroll-Driven Animations (`animation-timeline: view()`) do the
 * reveal for free in browsers that support them (~84% global as of 2026) —
 * see style.css's `@supports` block. This module is only the fallback for
 * browsers that don't: a real IntersectionObserver that adds `.is-visible`
 * the first time each item enters the viewport, then stops observing it.
 *
 * Structured so both branches are unit-testable: pass a fake
 * `supportsScrollTimeline` / `IntersectionObserver` to avoid depending on
 * real browser support during tests.
 */
export function initScrollReveal({
  items,
  supportsScrollTimeline = typeof CSS !== 'undefined' && CSS.supports?.('animation-timeline: view()'),
  IntersectionObserverImpl = typeof IntersectionObserver !== 'undefined' ? IntersectionObserver : undefined,
} = {}) {
  if (supportsScrollTimeline) {
    // Native CSS handles the reveal entirely — nothing for JS to do.
    return { observed: 0, usedFallback: false };
  }

  if (!items?.length || !IntersectionObserverImpl) {
    return { observed: 0, usedFallback: true };
  }

  const observer = new IntersectionObserverImpl((entries, obs) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        obs.unobserve(entry.target);
      }
    }
  });

  for (const item of items) {
    observer.observe(item);
  }

  return { observed: items.length, usedFallback: true, observer };
}

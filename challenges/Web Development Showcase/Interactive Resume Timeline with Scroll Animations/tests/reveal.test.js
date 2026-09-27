import { describe, it, expect, vi } from 'vitest';
import { initScrollReveal } from '../reveal.js';

class MockIntersectionObserver {
  constructor(callback) {
    this.callback = callback;
    this.observed = [];
    this.unobserve = vi.fn((target) => {
      this.observed = this.observed.filter((el) => el !== target);
    });
  }

  observe(target) {
    this.observed.push(target);
  }

  trigger(target, isIntersecting) {
    this.callback([{ target, isIntersecting }], this);
  }
}

function makeItem() {
  const el = document.createElement('li');
  el.className = 'timeline-item';
  return el;
}

describe('initScrollReveal — native CSS scroll-timeline branch', () => {
  it('does nothing and creates no observer when the browser supports animation-timeline: view()', () => {
    const items = [makeItem(), makeItem()];
    let constructed = 0;
    class ShouldNotBeCalled {
      constructor() {
        constructed += 1;
      }
    }

    const result = initScrollReveal({
      items,
      supportsScrollTimeline: true,
      IntersectionObserverImpl: ShouldNotBeCalled,
    });

    expect(result).toEqual({ observed: 0, usedFallback: false });
    expect(constructed).toBe(0);
    expect(items[0].classList.contains('is-visible')).toBe(false);
  });
});

describe('initScrollReveal — IntersectionObserver fallback branch', () => {
  it('observes every item when native scroll-timeline is unsupported', () => {
    const items = [makeItem(), makeItem(), makeItem()];

    const result = initScrollReveal({
      items,
      supportsScrollTimeline: false,
      IntersectionObserverImpl: MockIntersectionObserver,
    });

    expect(result.usedFallback).toBe(true);
    expect(result.observed).toBe(3);
    expect(result.observer.observed).toHaveLength(3);
  });

  it('adds is-visible and unobserves once an item intersects', () => {
    const items = [makeItem()];

    const { observer } = initScrollReveal({
      items,
      supportsScrollTimeline: false,
      IntersectionObserverImpl: MockIntersectionObserver,
    });

    expect(items[0].classList.contains('is-visible')).toBe(false);

    observer.trigger(items[0], true);

    expect(items[0].classList.contains('is-visible')).toBe(true);
    expect(observer.unobserve).toHaveBeenCalledTimes(1);
    expect(observer.unobserve).toHaveBeenCalledWith(items[0]);
  });

  it('does not add is-visible for a non-intersecting entry', () => {
    const items = [makeItem()];

    const { observer } = initScrollReveal({
      items,
      supportsScrollTimeline: false,
      IntersectionObserverImpl: MockIntersectionObserver,
    });

    observer.trigger(items[0], false);

    expect(items[0].classList.contains('is-visible')).toBe(false);
    expect(observer.unobserve).not.toHaveBeenCalled();
  });

  it('returns zero observed items and no observer when the items array is empty', () => {
    const result = initScrollReveal({
      items: [],
      supportsScrollTimeline: false,
      IntersectionObserverImpl: MockIntersectionObserver,
    });

    expect(result).toEqual({ observed: 0, usedFallback: true });
  });
});

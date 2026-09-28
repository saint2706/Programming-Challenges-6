import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createTypingHero } from '../src/index.js';

function makeMatchMedia(matches) {
  return (query) => ({
    matches,
    media: query,
    addEventListener() {},
    removeEventListener() {},
  });
}

describe('createTypingHero', () => {
  let el;

  beforeEach(() => {
    vi.useFakeTimers();
    el = document.createElement('div');
    document.body.appendChild(el);
    window.matchMedia = makeMatchMedia(false);
  });

  afterEach(() => {
    vi.useRealTimers();
    document.body.innerHTML = '';
  });

  it('throws without a target element', () => {
    expect(() => createTypingHero(null, { strings: ['Hi'] })).toThrow();
  });

  it('types characters over time and marks the visual text aria-hidden', () => {
    createTypingHero(el, { strings: ['Hi'], typingSpeed: 100, startDelay: 0 }).start();

    const visual = el.querySelector('.typing-hero__visual');
    expect(visual.getAttribute('aria-hidden')).toBe('true');

    // startDelay=0 fires the first tick immediately; give it 1ms so it isn't
    // ambiguous with the fake-timer clock boundary, then stay well short of
    // the next 100ms typingSpeed tick before asserting only 'H' has typed.
    vi.advanceTimersByTime(1);
    expect(visual.textContent).toBe('H');

    vi.advanceTimersByTime(50);
    expect(visual.textContent).toBe('H');

    vi.advanceTimersByTime(60);
    expect(visual.textContent).toBe('Hi');
  });

  it('announces the full phrase once per phrase-change via the sr-only live region, not per keystroke', () => {
    createTypingHero(el, { strings: ['Hi'], typingSpeed: 50, startDelay: 0 }).start();

    const srLive = el.querySelector('.typing-hero__sr-only');
    expect(srLive.getAttribute('aria-live')).toBe('polite');

    vi.advanceTimersByTime(50);
    expect(srLive.textContent).toBe('Hi');

    vi.advanceTimersByTime(50);
    expect(srLive.textContent).toBe('Hi'); // unchanged on the second keystroke
  });

  it('stops scheduling further ticks after stop()', () => {
    const controller = createTypingHero(el, { strings: ['Hello'], typingSpeed: 10, startDelay: 0 });
    controller.start();

    vi.advanceTimersByTime(20);
    const visual = el.querySelector('.typing-hero__visual');
    const textAtStop = visual.textContent;
    controller.stop();

    vi.advanceTimersByTime(1000);
    expect(visual.textContent).toBe(textAtStop);
  });

  it('calls onComplete exactly once when loop is false', () => {
    const onComplete = vi.fn();
    createTypingHero(el, { strings: ['Hi'], typingSpeed: 10, loop: false, onComplete }).start();

    vi.advanceTimersByTime(10); // 'H'
    vi.advanceTimersByTime(10); // 'Hi' -> done
    expect(onComplete).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(10000);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it('destroy() clears the element and stops the timer', () => {
    const controller = createTypingHero(el, { strings: ['Hi'], typingSpeed: 10 });
    controller.start();
    vi.advanceTimersByTime(10);
    controller.destroy();

    expect(el.textContent).toBe('');
    expect(el.classList.contains('typing-hero')).toBe(false);
  });

  it('renders statically and skips animation when prefers-reduced-motion is set', () => {
    window.matchMedia = makeMatchMedia(true);
    createTypingHero(el, { strings: ['Reduced'], typingSpeed: 10 }).start();

    const visual = el.querySelector('.typing-hero__visual');
    const cursor = el.querySelector('.typing-hero__cursor');
    expect(visual.textContent).toBe('Reduced');
    expect(cursor.style.display).toBe('none');

    vi.advanceTimersByTime(10000);
    expect(visual.textContent).toBe('Reduced'); // nothing further scheduled
  });
});

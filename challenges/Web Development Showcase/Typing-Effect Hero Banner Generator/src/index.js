import { createInitialState, advance } from './stateMachine.js';

/**
 * @typedef {object} TypingHeroOptions
 * @property {string[]} strings - phrases to cycle through
 * @property {number} [typingSpeed=60] - ms per typed character
 * @property {number} [deletingSpeed=30] - ms per deleted character
 * @property {number} [pauseBeforeDelete=1200] - ms to hold a fully-typed phrase before deleting
 * @property {number} [pauseBeforeNext=400] - ms to hold an empty string before typing the next phrase
 * @property {number} [startDelay=0] - ms before the first character is typed
 * @property {boolean} [loop=true] - cycle forever; if false, stop after the last phrase is typed once
 * @property {string} [cursorChar='|'] - character rendered as the blinking cursor
 * @property {boolean} [cursorBlink=true] - whether the cursor blinks
 * @property {() => void} [onComplete] - called once, only when loop is false and the last phrase finishes
 *
 * @typedef {object} TypingHeroController
 * @property {() => void} start
 * @property {() => void} stop
 * @property {() => void} destroy
 */

const DEFAULTS = {
  typingSpeed: 60,
  deletingSpeed: 30,
  pauseBeforeDelete: 1200,
  pauseBeforeNext: 400,
  startDelay: 0,
  loop: true,
  cursorChar: '|',
  cursorBlink: true,
  onComplete: undefined,
};

/**
 * Mount a configurable typing/deleting hero-banner effect into `el`.
 *
 * @param {HTMLElement} el - target element; its contents are replaced
 * @param {TypingHeroOptions} options
 * @returns {TypingHeroController}
 */
export function createTypingHero(el, options = {}) {
  if (!el || typeof el.appendChild !== 'function') {
    throw new Error('createTypingHero: `el` must be a DOM element');
  }
  const strings = Array.isArray(options.strings) && options.strings.length > 0 ? options.strings : [''];
  const config = { ...DEFAULTS, ...options, strings };

  const reducedMotionQuery =
    typeof window !== 'undefined' && typeof window.matchMedia === 'function'
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : null;

  let state = createInitialState();
  let timerId = null;
  let running = false;

  el.classList.add('typing-hero');
  el.textContent = '';

  const visual = document.createElement('span');
  visual.className = 'typing-hero__visual';
  visual.setAttribute('aria-hidden', 'true');

  const cursor = document.createElement('span');
  cursor.className = 'typing-hero__cursor';
  cursor.setAttribute('aria-hidden', 'true');
  cursor.textContent = config.cursorChar;
  if (config.cursorBlink) {
    cursor.classList.add('typing-hero__cursor--blink');
  }

  // Screen readers get the *whole* current phrase announced once per
  // phrase-change, not once per keystroke — see renderPhraseIfChanged below.
  const srLive = document.createElement('span');
  srLive.className = 'typing-hero__sr-only';
  srLive.setAttribute('aria-live', 'polite');

  el.append(visual, cursor, srLive);

  function renderVisual() {
    const phrase = config.strings[state.phraseIndex] ?? '';
    visual.textContent = phrase.slice(0, state.charIndex);
  }

  function announcePhraseIfChanged(previousPhraseIndex) {
    if (previousPhraseIndex !== state.phraseIndex || srLive.textContent === '') {
      srLive.textContent = config.strings[state.phraseIndex] ?? '';
    }
  }

  function renderReducedMotion() {
    const phrase = config.strings[0] ?? '';
    visual.textContent = phrase;
    srLive.textContent = phrase;
    cursor.style.display = 'none';
  }

  function tick() {
    const previousPhraseIndex = state.phraseIndex;
    const result = advance(state, config);
    state = result.state;
    renderVisual();
    announcePhraseIfChanged(previousPhraseIndex);

    if (result.done) {
      running = false;
      timerId = null;
      config.onComplete?.();
      return;
    }
    timerId = setTimeout(tick, result.delay);
  }

  function start() {
    if (running) return;
    if (reducedMotionQuery?.matches) {
      renderReducedMotion();
      return;
    }
    running = true;
    timerId = setTimeout(tick, config.startDelay);
  }

  function stop() {
    running = false;
    if (timerId !== null) {
      clearTimeout(timerId);
      timerId = null;
    }
  }

  function destroy() {
    stop();
    el.textContent = '';
    el.classList.remove('typing-hero');
  }

  return { start, stop, destroy };
}

export default createTypingHero;

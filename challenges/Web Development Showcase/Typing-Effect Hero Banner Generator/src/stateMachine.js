/**
 * Pure, DOM-free typing/deleting state machine for the typing hero effect.
 * Every transition is a function of (state, options) -> { state, delay, done },
 * so it can be unit tested without timers, a DOM, or a browser at all.
 *
 * @typedef {'typing'|'pauseBeforeDelete'|'deleting'|'pauseBeforeNext'|'done'} TypingPhase
 *
 * @typedef {object} TypingState
 * @property {number} phraseIndex - index into options.strings
 * @property {number} charIndex - how many characters of the current phrase are "typed"
 * @property {TypingPhase} phase
 *
 * @typedef {object} StepResult
 * @property {TypingState} state - the next state
 * @property {number} delay - ms to wait before the next `advance` call
 * @property {boolean} done - true once the effect has permanently finished (non-looping only)
 */

/** @returns {TypingState} */
export function createInitialState() {
  return { phraseIndex: 0, charIndex: 0, phase: 'typing' };
}

/**
 * @param {TypingState} state
 * @param {{strings: string[], typingSpeed: number, deletingSpeed: number, pauseBeforeDelete: number, pauseBeforeNext: number, loop: boolean}} options
 * @returns {StepResult}
 */
export function advance(state, options) {
  const { strings, typingSpeed, deletingSpeed, pauseBeforeDelete, pauseBeforeNext, loop } = options;
  const isLastPhrase = state.phraseIndex === strings.length - 1;

  switch (state.phase) {
    case 'typing': {
      const phrase = strings[state.phraseIndex] ?? '';
      if (state.charIndex < phrase.length) {
        return {
          state: { ...state, charIndex: state.charIndex + 1 },
          delay: typingSpeed,
          done: false,
        };
      }
      if (isLastPhrase && !loop) {
        return { state: { ...state, phase: 'done' }, delay: 0, done: true };
      }
      return {
        state: { ...state, phase: 'pauseBeforeDelete' },
        delay: pauseBeforeDelete,
        done: false,
      };
    }

    case 'pauseBeforeDelete':
      return { state: { ...state, phase: 'deleting' }, delay: deletingSpeed, done: false };

    case 'deleting': {
      if (state.charIndex > 0) {
        return {
          state: { ...state, charIndex: state.charIndex - 1 },
          delay: deletingSpeed,
          done: false,
        };
      }
      const nextIndex = (state.phraseIndex + 1) % strings.length;
      return {
        state: { phraseIndex: nextIndex, charIndex: 0, phase: 'pauseBeforeNext' },
        delay: pauseBeforeNext,
        done: false,
      };
    }

    case 'pauseBeforeNext': {
      const phrase = strings[state.phraseIndex] ?? '';
      if (phrase.length === 0) {
        // Nothing to type for an empty phrase; treat it as already "typed" and
        // fall straight into the delete pause so the cycle keeps moving.
        return { state: { ...state, phase: 'pauseBeforeDelete' }, delay: pauseBeforeDelete, done: false };
      }
      return {
        state: { ...state, phase: 'typing', charIndex: 1 },
        delay: typingSpeed,
        done: false,
      };
    }

    case 'done':
    default:
      return { state, delay: 0, done: true };
  }
}

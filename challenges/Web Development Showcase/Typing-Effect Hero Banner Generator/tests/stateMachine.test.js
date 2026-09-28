import { describe, expect, it } from 'vitest';
import { advance, createInitialState } from '../src/stateMachine.js';

const baseOptions = {
  strings: ['Hi', 'Bye'],
  typingSpeed: 60,
  deletingSpeed: 30,
  pauseBeforeDelete: 1200,
  pauseBeforeNext: 400,
  loop: true,
};

function runSteps(state, options, count) {
  const history = [state];
  let current = state;
  for (let i = 0; i < count; i += 1) {
    const result = advance(current, options);
    current = result.state;
    history.push(current);
    if (result.done) break;
  }
  return history;
}

describe('createInitialState', () => {
  it('starts typing the first phrase from character 0', () => {
    expect(createInitialState()).toEqual({ phraseIndex: 0, charIndex: 0, phase: 'typing' });
  });
});

describe('advance: typing phase', () => {
  it('types one character per step at typingSpeed', () => {
    const state = createInitialState();
    const result = advance(state, baseOptions);
    expect(result).toEqual({
      state: { phraseIndex: 0, charIndex: 1, phase: 'typing' },
      delay: 60,
      done: false,
    });
  });

  it('moves to pauseBeforeDelete once the phrase is fully typed (loop=true)', () => {
    const state = { phraseIndex: 0, charIndex: 2, phase: 'typing' }; // 'Hi'.length === 2
    const result = advance(state, baseOptions);
    expect(result.state.phase).toBe('pauseBeforeDelete');
    expect(result.delay).toBe(1200);
    expect(result.done).toBe(false);
  });

  it('finishes without deleting when loop=false and the last phrase completes', () => {
    const options = { ...baseOptions, loop: false };
    const state = { phraseIndex: 1, charIndex: 3, phase: 'typing' }; // 'Bye'.length === 3, last index
    const result = advance(state, options);
    expect(result.state.phase).toBe('done');
    expect(result.done).toBe(true);
  });

  it('keeps deleting/cycling past a non-last phrase even when loop=false', () => {
    const options = { ...baseOptions, loop: false };
    const state = { phraseIndex: 0, charIndex: 2, phase: 'typing' }; // 'Hi' finished, not last
    const result = advance(state, options);
    expect(result.state.phase).toBe('pauseBeforeDelete');
    expect(result.done).toBe(false);
  });
});

describe('advance: pauseBeforeDelete -> deleting', () => {
  it('starts deleting after the pause', () => {
    const state = { phraseIndex: 0, charIndex: 2, phase: 'pauseBeforeDelete' };
    const result = advance(state, baseOptions);
    expect(result.state).toEqual({ phraseIndex: 0, charIndex: 2, phase: 'deleting' });
    expect(result.delay).toBe(30);
  });

  it('removes one character per step', () => {
    const state = { phraseIndex: 0, charIndex: 2, phase: 'deleting' };
    const result = advance(state, baseOptions);
    expect(result.state).toEqual({ phraseIndex: 0, charIndex: 1, phase: 'deleting' });
  });

  it('advances to the next phrase (wrapping) once fully deleted', () => {
    const state = { phraseIndex: 0, charIndex: 0, phase: 'deleting' };
    const result = advance(state, baseOptions);
    expect(result.state).toEqual({ phraseIndex: 1, charIndex: 0, phase: 'pauseBeforeNext' });
    expect(result.delay).toBe(400);
  });

  it('wraps from the last phrase back to phrase 0', () => {
    const state = { phraseIndex: 1, charIndex: 0, phase: 'deleting' };
    const result = advance(state, baseOptions);
    expect(result.state.phraseIndex).toBe(0);
    expect(result.state.phase).toBe('pauseBeforeNext');
  });
});

describe('advance: pauseBeforeNext', () => {
  it('types the first character of the next phrase', () => {
    const state = { phraseIndex: 1, charIndex: 0, phase: 'pauseBeforeNext' };
    const result = advance(state, baseOptions);
    expect(result.state).toEqual({ phraseIndex: 1, charIndex: 1, phase: 'typing' });
    expect(result.delay).toBe(60);
  });

  it('skips straight to the delete pause for an empty phrase', () => {
    const options = { ...baseOptions, strings: ['Hi', ''] };
    const state = { phraseIndex: 1, charIndex: 0, phase: 'pauseBeforeNext' };
    const result = advance(state, options);
    expect(result.state.phase).toBe('pauseBeforeDelete');
  });
});

describe('advance: done phase', () => {
  it('is terminal', () => {
    const state = { phraseIndex: 1, charIndex: 3, phase: 'done' };
    const result = advance(state, baseOptions);
    expect(result.done).toBe(true);
    expect(result.state.phase).toBe('done');
  });
});

describe('a full non-looping run', () => {
  it('types "Hi", cycles to "Bye", types it, then stops', () => {
    const options = { ...baseOptions, loop: false };
    const history = runSteps(createInitialState(), options, 200);
    const finalState = history[history.length - 1];
    expect(finalState.phase).toBe('done');
    expect(finalState.phraseIndex).toBe(1);

    // Confirm "Bye" was actually typed out in full at some point before done.
    const typedBye = history.some((s) => s.phraseIndex === 1 && s.phase === 'typing' && s.charIndex === 3);
    expect(typedBye).toBe(true);
  });
});

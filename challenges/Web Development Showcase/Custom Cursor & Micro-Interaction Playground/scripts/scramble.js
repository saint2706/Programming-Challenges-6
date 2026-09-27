const SCRAMBLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*';

/**
 * Computes one frame of a scramble-to-reveal animation.
 *
 * @param {string} target - the final text being revealed.
 * @param {number} revealCount - how many leading characters of `target` are
 *   already resolved (0..target.length).
 * @param {() => number} random - injectable RNG in [0, 1) for deterministic tests.
 * @returns {string} the text to display this frame: the resolved prefix,
 *   followed by random characters for the rest.
 */
export function scrambleStep(target, revealCount, random = Math.random) {
  const clampedReveal = Math.max(0, Math.min(revealCount, target.length));
  let result = target.slice(0, clampedReveal);

  for (let i = clampedReveal; i < target.length; i += 1) {
    if (target[i] === ' ') {
      result += ' ';
      continue;
    }
    const index = Math.floor(random() * SCRAMBLE_CHARS.length);
    result += SCRAMBLE_CHARS[index];
  }

  return result;
}

/** Wires up a hover-triggered scramble/decode effect on a text element. */
export function initScrambleText(el, { frameDelayMs = 30, revealEveryFrame = 1 } = {}) {
  if (!el) {
    return null;
  }

  const target = el.dataset.text ?? el.textContent;
  el.dataset.text = target;
  let timerId = null;

  function stop() {
    if (timerId !== null) {
      clearInterval(timerId);
      timerId = null;
    }
  }

  function play() {
    stop();
    let revealCount = 0;
    timerId = setInterval(() => {
      el.textContent = scrambleStep(target, revealCount);
      revealCount += revealEveryFrame;
      if (revealCount > target.length) {
        el.textContent = target;
        stop();
      }
    }, frameDelayMs);
  }

  el.addEventListener('mouseenter', play);
  el.addEventListener('focus', play);

  return function destroy() {
    stop();
    el.removeEventListener('mouseenter', play);
    el.removeEventListener('focus', play);
  };
}

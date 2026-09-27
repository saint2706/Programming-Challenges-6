import { magneticOffset } from './math.js';

/** Wires up "magnetic" nudge-toward-cursor behavior on a button-like element. */
export function initMagneticButton(button, { radius = 60, strength = 0.4 } = {}) {
  if (!button) {
    return null;
  }

  function handleMove(event) {
    const rect = button.getBoundingClientRect();
    const center = { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
    const cursor = { x: event.clientX, y: event.clientY };
    const { x, y } = magneticOffset(cursor, center, radius, strength);
    button.style.transform = `translate(${x}px, ${y}px)`;
  }

  function handleLeave() {
    button.style.transform = 'translate(0, 0)';
  }

  button.addEventListener('mousemove', handleMove);
  button.addEventListener('mouseleave', handleLeave);

  return function destroy() {
    button.removeEventListener('mousemove', handleMove);
    button.removeEventListener('mouseleave', handleLeave);
  };
}

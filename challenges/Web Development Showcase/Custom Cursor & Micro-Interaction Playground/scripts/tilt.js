import { tiltAngles } from './math.js';

/** Wires up a mouse-move 3D tilt effect on a card element. */
export function initTiltCard(card, { maxTiltDegrees = 12 } = {}) {
  if (!card) {
    return null;
  }

  function handleMove(event) {
    const rect = card.getBoundingClientRect();
    const pointer = { x: event.clientX, y: event.clientY };
    const { rotateX, rotateY } = tiltAngles(pointer, rect, maxTiltDegrees);
    card.style.transform = `perspective(600px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
  }

  function handleLeave() {
    card.style.transform = 'perspective(600px) rotateX(0deg) rotateY(0deg)';
  }

  card.addEventListener('mousemove', handleMove);
  card.addEventListener('mouseleave', handleLeave);

  return function destroy() {
    card.removeEventListener('mousemove', handleMove);
    card.removeEventListener('mouseleave', handleLeave);
  };
}

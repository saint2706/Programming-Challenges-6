import { initCursorFollower } from './cursor.js';
import { initMagneticButton } from './magnetic.js';
import { initTiltCard } from './tilt.js';
import { initSpotlightCard } from './spotlight.js';
import { initScrambleText } from './scramble.js';
import { initCursorTrail } from './trail.js';

function supportsFinePointer() {
  return window.matchMedia('(hover: hover) and (pointer: fine)').matches;
}

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

document.addEventListener('DOMContentLoaded', () => {
  const finePointer = supportsFinePointer();
  const reducedMotion = prefersReducedMotion();

  // Custom cursor only makes sense with a real mouse, and only animates
  // continuously when the visitor hasn't asked for reduced motion.
  if (finePointer && !reducedMotion) {
    document.body.classList.add('has-custom-cursor');
    initCursorFollower({
      dot: document.querySelector('#cursor-dot'),
      ring: document.querySelector('#cursor-ring'),
    });
  }

  const magneticButton = document.querySelector('[data-effect="magnetic"]');
  if (magneticButton && !reducedMotion) {
    initMagneticButton(magneticButton);
  }

  const tiltCard = document.querySelector('[data-effect="tilt"]');
  if (tiltCard && !reducedMotion) {
    initTiltCard(tiltCard);
  }

  // Spotlight and scramble are subtle/static-friendly enough to keep even
  // under reduced motion (no continuous transform animation involved).
  const spotlightCard = document.querySelector('[data-effect="spotlight"]');
  if (spotlightCard) {
    initSpotlightCard(spotlightCard);
  }

  document.querySelectorAll('[data-effect="scramble"]').forEach((el) => initScrambleText(el));

  const trailContainer = document.querySelector('[data-effect="trail"]');
  if (trailContainer && !reducedMotion) {
    initCursorTrail(trailContainer);
  }
});

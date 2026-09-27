/**
 * Damped custom-cursor follower (dot + ring) driven by GSAP's `quickTo`.
 * The only effect on this page allowed to depend on GSAP — everything else
 * in this playground is dependency-free vanilla JS.
 */
export function initCursorFollower({
  dot,
  ring,
  gsapInstance = window.gsap,
  doc = document,
} = {}) {
  if (!dot || !ring || !gsapInstance) {
    return null;
  }

  const dotX = gsapInstance.quickTo(dot, 'x', { duration: 0.1, ease: 'power3' });
  const dotY = gsapInstance.quickTo(dot, 'y', { duration: 0.1, ease: 'power3' });
  const ringX = gsapInstance.quickTo(ring, 'x', { duration: 0.5, ease: 'power3' });
  const ringY = gsapInstance.quickTo(ring, 'y', { duration: 0.5, ease: 'power3' });

  function handleMove(event) {
    dotX(event.clientX);
    dotY(event.clientY);
    ringX(event.clientX);
    ringY(event.clientY);
  }

  function handleDown() {
    gsapInstance.to(ring, { scale: 0.75, duration: 0.2 });
  }

  function handleUp() {
    gsapInstance.to(ring, { scale: 1, duration: 0.2 });
  }

  doc.addEventListener('mousemove', handleMove);
  doc.addEventListener('mousedown', handleDown);
  doc.addEventListener('mouseup', handleUp);

  return function destroy() {
    doc.removeEventListener('mousemove', handleMove);
    doc.removeEventListener('mousedown', handleDown);
    doc.removeEventListener('mouseup', handleUp);
  };
}

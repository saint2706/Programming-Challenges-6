/** Wires up a cursor-following radial-gradient spotlight via CSS custom properties. */
export function initSpotlightCard(card) {
  if (!card) {
    return null;
  }

  function handleMove(event) {
    const rect = card.getBoundingClientRect();
    card.style.setProperty('--x', `${event.clientX - rect.left}px`);
    card.style.setProperty('--y', `${event.clientY - rect.top}px`);
  }

  card.addEventListener('mousemove', handleMove);

  return function destroy() {
    card.removeEventListener('mousemove', handleMove);
  };
}

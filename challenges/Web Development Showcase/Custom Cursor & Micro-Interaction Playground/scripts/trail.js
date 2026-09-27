/**
 * Spawns fading particle trail elements that follow the cursor inside `container`.
 * Throttled to at most one spawn per `minSpawnIntervalMs` so fast mouse
 * movement doesn't flood the DOM with particles.
 */
export function initCursorTrail(container, { minSpawnIntervalMs = 40, particleLifeMs = 600 } = {}) {
  if (!container) {
    return null;
  }

  let lastSpawn = 0;

  function spawnParticle(x, y) {
    const particle = document.createElement('span');
    particle.className = 'trail-particle';
    particle.style.left = `${x}px`;
    particle.style.top = `${y}px`;
    container.appendChild(particle);
    setTimeout(() => particle.remove(), particleLifeMs);
  }

  function handleMove(event) {
    const now = performance.now();
    if (now - lastSpawn < minSpawnIntervalMs) {
      return;
    }
    lastSpawn = now;
    const rect = container.getBoundingClientRect();
    spawnParticle(event.clientX - rect.left, event.clientY - rect.top);
  }

  container.addEventListener('mousemove', handleMove);

  return function destroy() {
    container.removeEventListener('mousemove', handleMove);
  };
}

import { timelineEntries } from './data.js';
import { renderTimeline } from './render.js';
import { initScrollReveal } from './reveal.js';

document.addEventListener('DOMContentLoaded', () => {
  const mount = document.getElementById('timeline-mount');
  const timeline = renderTimeline(timelineEntries, document);
  mount.appendChild(timeline);

  const items = Array.from(timeline.querySelectorAll('.timeline-item'));
  initScrollReveal({ items });
});

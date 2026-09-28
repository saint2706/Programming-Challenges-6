import { CATEGORIES, ICONS } from './icons.js';
import { stripSmilAnimations } from './smil.js';

const gallery = document.getElementById('gallery');
const filterBar = document.getElementById('filters');

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function iconMarkupForRender(icon) {
  // SMIL doesn't obey CSS media queries, so under reduced motion we never
  // insert the <animate*> elements at all rather than trying to pause them.
  return icon.technique === 'smil' && prefersReducedMotion ? stripSmilAnimations(icon.markup) : icon.markup;
}

function renderPanel(icon) {
  const panel = document.createElement('button');
  panel.type = 'button';
  panel.className = 'icon-panel';
  panel.dataset.icon = icon.id;
  panel.dataset.category = icon.category;
  panel.dataset.technique = icon.technique;
  panel.setAttribute('aria-label', `${icon.name} — ${icon.technique === 'smil' ? 'SMIL' : 'CSS'} hover animation`);

  const stage = document.createElement('span');
  stage.className = 'icon-stage';
  // icon.markup comes from this repo's own static icons.js data module, never
  // from user input, so inserting it as markup here carries no XSS risk.
  stage.innerHTML = iconMarkupForRender(icon);

  const nameEl = document.createElement('span');
  nameEl.className = 'icon-name';
  nameEl.textContent = icon.name;

  const techniqueEl = document.createElement('span');
  techniqueEl.className = 'icon-technique';
  techniqueEl.textContent = icon.technique === 'smil' ? 'SMIL' : 'CSS';

  panel.append(stage, nameEl, techniqueEl);

  const summary = document.createElement('summary');
  summary.textContent = 'View source';

  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.textContent = icon.markup;
  pre.append(code);

  const copyBtn = document.createElement('button');
  copyBtn.type = 'button';
  copyBtn.className = 'copy-btn';
  copyBtn.textContent = 'Copy SVG';
  copyBtn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(icon.markup);
      const original = copyBtn.textContent;
      copyBtn.textContent = 'Copied!';
      setTimeout(() => {
        copyBtn.textContent = original;
      }, 1200);
    } catch {
      copyBtn.textContent = 'Copy failed — select manually';
    }
  });

  const details = document.createElement('details');
  details.className = 'source';
  details.append(summary, pre, copyBtn);

  const card = document.createElement('div');
  card.className = 'icon-card';
  card.append(panel, details);
  return card;
}

function renderGallery(filterCategory = 'all') {
  gallery.textContent = '';
  const visible = filterCategory === 'all' ? ICONS : ICONS.filter((icon) => icon.category === filterCategory);
  const fragment = document.createDocumentFragment();
  visible.forEach((icon) => fragment.append(renderPanel(icon)));
  gallery.append(fragment);
  gallery.setAttribute('aria-busy', 'false');
}

function renderFilters() {
  const allBtn = document.createElement('button');
  allBtn.type = 'button';
  allBtn.className = 'filter-btn active';
  allBtn.textContent = `All (${ICONS.length})`;
  allBtn.dataset.filter = 'all';
  filterBar.append(allBtn);

  CATEGORIES.forEach(({ id, label }) => {
    const count = ICONS.filter((icon) => icon.category === id).length;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'filter-btn';
    btn.textContent = `${label} (${count})`;
    btn.dataset.filter = id;
    filterBar.append(btn);
  });

  filterBar.addEventListener('click', (event) => {
    const btn = event.target.closest('.filter-btn');
    if (!btn) return;
    filterBar.querySelectorAll('.filter-btn').forEach((b) => b.classList.toggle('active', b === btn));
    renderGallery(btn.dataset.filter);
  });
}

renderFilters();
renderGallery();

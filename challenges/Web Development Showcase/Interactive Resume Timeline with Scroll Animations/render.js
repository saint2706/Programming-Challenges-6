/**
 * Pure rendering helpers — no DOM-global access, so this is trivial to unit test.
 * `document` is passed in explicitly rather than referenced as a global.
 */

const CATEGORY_LABEL = {
  education: 'Education',
  experience: 'Experience',
  certification: 'Certification',
};

/**
 * Build the <li> element for a single timeline entry.
 * @param {object} entry
 * @param {Document} doc
 * @returns {HTMLLIElement}
 */
export function renderTimelineItem(entry, doc = document) {
  const li = doc.createElement('li');
  li.className = `timeline-item timeline-item--${entry.type}`;
  li.setAttribute('data-category', entry.type);

  const badge = doc.createElement('span');
  badge.className = 'timeline-item__badge';
  badge.textContent = CATEGORY_LABEL[entry.type] ?? entry.type;
  li.appendChild(badge);

  const date = doc.createElement('time');
  date.className = 'timeline-item__date';
  date.textContent = entry.date;
  li.appendChild(date);

  const title = doc.createElement('h3');
  title.className = 'timeline-item__title';
  title.textContent = entry.title;
  li.appendChild(title);

  const org = doc.createElement('p');
  org.className = 'timeline-item__org';
  org.textContent = entry.org;
  li.appendChild(org);

  if (entry.description) {
    const desc = doc.createElement('p');
    desc.className = 'timeline-item__description';
    desc.textContent = entry.description;
    li.appendChild(desc);
  }

  if (entry.highlights?.length) {
    const list = doc.createElement('ul');
    list.className = 'timeline-item__highlights';
    for (const highlight of entry.highlights) {
      const item = doc.createElement('li');
      item.textContent = highlight;
      list.appendChild(item);
    }
    li.appendChild(list);
  }

  return li;
}

/**
 * Build the full <ol class="timeline"> element from an entries array.
 * @param {object[]} entries
 * @param {Document} doc
 * @returns {HTMLOListElement}
 */
export function renderTimeline(entries, doc = document) {
  const ol = doc.createElement('ol');
  ol.className = 'timeline';
  for (const entry of entries) {
    ol.appendChild(renderTimelineItem(entry, doc));
  }
  return ol;
}

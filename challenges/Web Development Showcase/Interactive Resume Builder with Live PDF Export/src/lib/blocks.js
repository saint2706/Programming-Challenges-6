import { getTemplate } from '../templates/index.js';
import { formatDate, formatRange } from './dates.js';
import { mailHref, safeHref, telHref } from './links.js';
import { SECTION_META } from './sections.js';

/**
 * Turns a resume plus its layout into regions of blocks, the units the paginator places.
 *
 * A block is `{ id, kind, keepWithNext, data, hash, rows? }`:
 *   - `id` is derived from the document path (`work.1.header`), so it stays the same while the
 *     text inside is edited and Svelte keeps the DOM node (and the caret) alive;
 *   - `hash` fingerprints `kind` + `data` and keys the measured-height cache, so a keystroke
 *     re-measures only the block it touched;
 *   - `data` is everything `Block.svelte` needs to render it. Editable text is a
 *     `{ path, value }` field; links are sanitised here (`href` is null when unsafe);
 *   - list blocks (`bullets`) carry `rows` and may split between rows; headings and entry
 *     headers are `keepWithNext`.
 */

/** FNV-1a, 32-bit, as 8 hex digits. */
export function hashString(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

const field = (path, value) => ({ path, value });
const isBlank = (value) => typeof value !== 'string' || value.trim() === '';

function block(id, kind, data, { keepWithNext = false, rows } = {}) {
  const made = { id, kind, keepWithNext, data, hash: hashString(JSON.stringify([kind, data, rows])) };
  if (rows) made.rows = rows;
  return made;
}

function hostname(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '') || url.replace(/^(mailto|tel):/, '');
  } catch {
    return url;
  }
}

const MAX_LINK_LABEL = 48;

/** `github.com/user/repo` for a web link, the address for mailto: and tel:. */
function linkLabel(href) {
  if (/^(mailto|tel):/i.test(href)) return href.replace(/^(mailto|tel):/i, '');
  try {
    const { hostname: host, pathname } = new URL(href);
    const path = pathname === '/' ? '' : pathname.replace(/\/$/, '');
    const text = `${host.replace(/^www\./, '')}${path}`;
    return text.length > MAX_LINK_LABEL ? `${text.slice(0, MAX_LINK_LABEL - 1)}…` : text;
  } catch {
    return href;
  }
}

/** `{ href, label }` for a safe link, otherwise null. */
function linkOf(url) {
  const href = safeHref(url);
  return href ? { href, label: linkLabel(href) } : null;
}

function contactData(basics) {
  const { location, email, phone, url, profiles } = basics;
  const place = [location.city, location.region, location.countryCode].filter(Boolean).join(', ');
  const emailData = isBlank(email) ? null : { label: email.trim(), href: mailHref(email) };
  const phoneData = isBlank(phone) ? null : { label: phone.trim(), href: telHref(phone) };
  const website = isBlank(url) ? null : { label: hostname(url.trim()), href: safeHref(url) };
  const links = profiles
    .filter((p) => p.network || p.username || p.url)
    .map((p) => ({ label: p.network || p.username || hostname(p.url), href: safeHref(p.url) }));
  if (!place && !emailData && !phoneData && !website && links.length === 0) return null;
  return { location: place, email: emailData, phone: phoneData, website, profiles: links };
}

function headerBlock(doc, withContact) {
  const { basics } = doc;
  return block('header', 'header', {
    name: field('basics.name', basics.name),
    label: field('basics.label', basics.label),
    contact: withContact ? contactData(basics) : null,
  });
}

function bulletRows(section, index, key, list) {
  return list.map((value, j) => {
    const path = `${section}.${index}.${key}.${j}`;
    return { id: path, path, value };
  });
}

function bulletsBlock(section, index, key, list) {
  const rows = bulletRows(section, index, key, list);
  return block(`${section}.${index}.bullets`, 'bullets', { section, index, key }, { rows });
}

function textBlock(id, path, value) {
  return block(id, 'text', field(path, value));
}

function entryBlocks(section, entry, i) {
  const at = (key) => field(`${section}.${i}.${key}`, entry[key]);
  const out = [];
  if (section === 'work') {
    out.push(
      block(
        `work.${i}.header`,
        'work-header',
        { position: at('position'), name: at('name'), dates: formatRange(entry.startDate, entry.endDate), link: linkOf(entry.url) },
        { keepWithNext: true },
      ),
    );
    if (!isBlank(entry.summary)) out.push(textBlock(`work.${i}.text`, `work.${i}.summary`, entry.summary));
    if (entry.highlights.length) out.push(bulletsBlock('work', i, 'highlights', entry.highlights));
  } else if (section === 'education') {
    out.push(
      block(`education.${i}.header`, 'education-header', {
        institution: at('institution'),
        studyType: at('studyType'),
        area: at('area'),
        score: at('score'),
        dates: formatRange(entry.startDate, entry.endDate),
        link: linkOf(entry.url),
      }),
    );
  } else if (section === 'projects') {
    out.push(
      block(
        `projects.${i}.header`,
        'project-header',
        { name: at('name'), dates: entry.endDate ? formatRange(entry.startDate, entry.endDate) : formatDate(entry.startDate), link: linkOf(entry.url) },
        { keepWithNext: true },
      ),
    );
    if (!isBlank(entry.description)) out.push(textBlock(`projects.${i}.text`, `projects.${i}.description`, entry.description));
    if (entry.highlights.length) out.push(bulletsBlock('projects', i, 'highlights', entry.highlights));
  } else if (section === 'skills') {
    out.push(block(`skills.${i}`, 'skill', { name: at('name'), keywords: field(`skills.${i}.keywords`, entry.keywords) }));
  } else if (section === 'certificates') {
    out.push(
      block(`certificates.${i}`, 'certificate', {
        name: at('name'),
        issuer: at('issuer'),
        date: formatDate(entry.date),
        link: linkOf(entry.url),
      }),
    );
  } else if (section === 'languages') {
    out.push(block(`languages.${i}`, 'language', { language: at('language'), fluency: at('fluency') }));
  } else if (section === 'awards') {
    out.push(
      block(
        `awards.${i}.header`,
        'award',
        { title: at('title'), awarder: at('awarder'), date: formatDate(entry.date) },
        { keepWithNext: !isBlank(entry.summary) },
      ),
    );
    if (!isBlank(entry.summary)) out.push(textBlock(`awards.${i}.text`, `awards.${i}.summary`, entry.summary));
  }
  return out;
}

function sectionBlocks(doc, id) {
  const heading = block(`heading.${id}`, 'heading', { section: id, title: SECTION_META[id].title }, { keepWithNext: true });
  if (id === 'summary') {
    return isBlank(doc.basics.summary) ? [] : [heading, textBlock('summary.text', 'basics.summary', doc.basics.summary)];
  }
  const entries = doc[id];
  if (!entries.length) return [];
  return [heading, ...entries.flatMap((entry, i) => entryBlocks(id, entry, i))];
}

/** The regions of blocks for `doc`, following its template, section order and hidden sections. */
export function buildRegions(doc) {
  const layout = doc['x-layout'];
  const template = getTemplate(layout.template);
  const visible = layout.sectionOrder.filter((id) => !layout.hidden.includes(id));
  return template.regions.map((spec) => {
    const blocks = [];
    if (spec.header) blocks.push(headerBlock(doc, spec.header === 'full'));
    if (spec.contact) {
      const contact = contactData(doc.basics);
      if (contact) blocks.push(block('contact', 'contact', contact));
    }
    const ids = spec.sections === 'all' ? visible : visible.filter((id) => spec.sections.includes(id));
    for (const id of ids) blocks.push(...sectionBlocks(doc, id));
    return { id: spec.id, blocks };
  });
}

/** Map from block id to block across all regions. */
export function blockIndex(regions) {
  return new Map(regions.flatMap((r) => r.blocks).map((b) => [b.id, b]));
}

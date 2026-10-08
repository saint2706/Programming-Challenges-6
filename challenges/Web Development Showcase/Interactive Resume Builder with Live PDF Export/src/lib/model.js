/**
 * Pure edit operations on a resume document. Every function returns a new document (untouched
 * branches are shared) and never mutates its input, so the UI can keep the document in
 * `$state.raw` and replace it wholesale. Invalid operations throw `RangeError`, or return the
 * same document when they are harmless no-ops (moving past the end of a list, removing an index
 * that does not exist).
 */

import { LIMITS, SECTION_IDS, parseLayout } from './schema.js';
import { newEntry } from './sections.js';

const FORBIDDEN_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

/** `'work.1.highlights.0'` to `['work', 1, 'highlights', 0]`. */
export function parsePath(path) {
  if (Array.isArray(path)) return path;
  return String(path)
    .split('.')
    .map((part) => (/^\d+$/.test(part) ? Number(part) : part));
}

export function toPathString(path) {
  return parsePath(path).join('.');
}

export function getPath(doc, path) {
  let node = doc;
  for (const key of parsePath(path)) {
    if (node === null || typeof node !== 'object') return undefined;
    node = node[key];
  }
  return node;
}

function update(node, keys, value, original) {
  const [head, ...rest] = keys;
  if (
    typeof head === 'string' && FORBIDDEN_KEYS.has(head)
  ) {
    throw new RangeError(`illegal path segment: ${head}`);
  }
  if (node === null || typeof node !== 'object' || !Object.hasOwn(node, head)) {
    throw new RangeError(`no such field: ${toPathString(original)}`);
  }
  const copy = Array.isArray(node) ? node.slice() : { ...node };
  copy[head] = rest.length ? update(node[head], rest, value, original) : value;
  return copy;
}

/** Set an existing field. Creating new fields or indices is deliberately not possible. */
export function setPath(doc, path, value) {
  const keys = parsePath(path);
  if (keys.length === 0) throw new RangeError('empty path');
  return update(doc, keys, value, keys);
}

function entriesOf(doc, section) {
  const list = doc[section];
  if (!Array.isArray(list)) throw new RangeError(`section "${section}" has no entries`);
  return list;
}

export function addEntry(doc, section, entry = newEntry(section)) {
  const list = entriesOf(doc, section);
  if (list.length >= LIMITS.entries) {
    throw new RangeError(`a section holds at most ${LIMITS.entries} entries`);
  }
  return { ...doc, [section]: [...list, entry] };
}

export function removeEntry(doc, section, index) {
  const list = entriesOf(doc, section);
  if (!Number.isInteger(index) || index < 0 || index >= list.length) return doc;
  return { ...doc, [section]: list.filter((_, i) => i !== index) };
}

function moved(list, from, to) {
  if (
    !Number.isInteger(from) ||
    !Number.isInteger(to) ||
    from === to ||
    from < 0 ||
    to < 0 ||
    from >= list.length ||
    to >= list.length
  ) {
    return null;
  }
  const copy = list.slice();
  const [item] = copy.splice(from, 1);
  copy.splice(to, 0, item);
  return copy;
}

export function moveEntry(doc, section, from, to) {
  const next = moved(entriesOf(doc, section), from, to);
  return next ? { ...doc, [section]: next } : doc;
}

function withBullets(doc, section, index, field, change) {
  const entry = entriesOf(doc, section)[index];
  if (!entry || !Array.isArray(entry[field])) throw new RangeError(`no list "${field}" at ${section}.${index}`);
  const next = change(entry[field]);
  if (next === null) return doc;
  return setPath(doc, [section, index, field], next);
}

export function addBullet(doc, section, index, field = 'highlights', text = '') {
  return withBullets(doc, section, index, field, (list) => {
    if (list.length >= LIMITS.bullets) {
      throw new RangeError(`a list holds at most ${LIMITS.bullets} bullets`);
    }
    return [...list, text];
  });
}

export function removeBullet(doc, section, index, field, bulletIndex) {
  return withBullets(doc, section, index, field, (list) =>
    Number.isInteger(bulletIndex) && bulletIndex >= 0 && bulletIndex < list.length
      ? list.filter((_, i) => i !== bulletIndex)
      : null,
  );
}

export function moveBullet(doc, section, index, field, from, to) {
  return withBullets(doc, section, index, field, (list) => moved(list, from, to));
}

function withLayout(doc, layout) {
  return { ...doc, 'x-layout': layout };
}

/** Shift a section up (`delta` -1) or down (+1) in the section order. */
export function moveSection(doc, id, delta) {
  const order = doc['x-layout'].sectionOrder;
  const from = order.indexOf(id);
  const next = from === -1 ? null : moved(order, from, from + delta);
  return next ? withLayout(doc, { ...doc['x-layout'], sectionOrder: next }) : doc;
}

export function setSectionHidden(doc, id, hidden) {
  if (!SECTION_IDS.includes(id)) return doc;
  const current = doc['x-layout'].hidden;
  if (current.includes(id) === hidden) return doc;
  const next = hidden ? [...current, id] : current.filter((s) => s !== id);
  return withLayout(doc, { ...doc['x-layout'], hidden: next });
}

/** Merge a partial layout (template, pageSize, accent, fontPair, scale, ...). */
export function setLayout(doc, patch) {
  const layout = parseLayout({ ...doc['x-layout'], ...patch });
  if (!layout) throw new RangeError('invalid layout');
  return withLayout(doc, layout);
}

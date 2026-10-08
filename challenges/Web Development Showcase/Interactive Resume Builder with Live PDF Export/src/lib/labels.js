import { SECTION_META } from './sections.js';

const BASICS = { name: 'Name', label: 'Headline', summary: 'Summary' };

/** An accessible name for an in-place field, from its document path. */
export function fieldLabel(path) {
  const parts = String(path).split('.');
  if (parts[0] === 'basics') return BASICS[parts[1]] ?? path;

  const meta = SECTION_META[parts[0]];
  const entry = Number(parts[1]);
  if (!meta?.fields || !Number.isInteger(entry)) return path;
  const where = `${meta.title} entry ${entry + 1}`;

  const field = meta.fields.find((f) => f.key === parts[2]);
  if (!field) return path;
  if (field.type === 'bullets' && parts[3] !== undefined) return `Bullet ${Number(parts[3]) + 1}, ${where}`;
  return `${field.label}, ${where}`;
}

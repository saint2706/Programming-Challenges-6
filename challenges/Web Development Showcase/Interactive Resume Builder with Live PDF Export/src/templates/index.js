import classic from './classic.js';
import compact from './compact.js';
import sidebar from './sidebar.js';

export const TEMPLATES = { classic, sidebar, compact };
export const DEFAULT_TEMPLATE_ID = 'classic';

/** The template with this id, or Classic for an unknown id. */
export function getTemplate(id) {
  return TEMPLATES[id] ?? TEMPLATES[DEFAULT_TEMPLATE_ID];
}

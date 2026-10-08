import { formatIssues, parseResume } from './schema.js';

/**
 * JSON import and export. Import treats the file as hostile: size-capped, parsed, then validated
 * and normalised by the schema (which strips every key it does not model, including `__proto__`).
 * Nothing is applied unless the whole file is valid.
 */

export const MAX_IMPORT_BYTES = 1_000_000;

/** Pretty JSON Resume with the builder's layout under `x-layout`. */
export function exportJson(doc) {
  return `${JSON.stringify(doc, null, 2)}\n`;
}

/**
 * @returns {{ok: true, doc: object} | {ok: false, message: string}}
 */
export function importJson(text) {
  if (typeof text !== 'string' || text.length > MAX_IMPORT_BYTES) {
    return { ok: false, message: `That file is too large to import (the limit is ${MAX_IMPORT_BYTES / 1000} KB).` };
  }
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    return { ok: false, message: 'That file is not valid JSON.' };
  }
  const result = parseResume(data);
  if (!result.ok) {
    const lines = formatIssues(result.errors);
    const message = result.errors[0].path === '' ? 'Expected a JSON object (a JSON Resume file).' : `Some fields are not valid:\n${lines}`;
    return { ok: false, message };
  }
  return { ok: true, doc: result.data };
}

/** A safe download file name from the person's name: `ada-lovelace-resume.json`. */
export function exportFileName(doc) {
  const slug = String(doc.basics.name ?? '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40)
    .replace(/-+$/g, '');
  return slug ? `${slug}-resume.json` : 'resume.json';
}

/**
 * DOM-free CSV serializer.
 *
 * - RFC 4180: fields containing , " CR or LF are quoted, quotes doubled, CRLF
 *   row terminator (including after the last row).
 * - Formula-injection guard (OWASP "CSV Injection"): a TEXT cell whose first
 *   character is = + - @ TAB or CR gets a leading single quote so Excel/Sheets
 *   show it literally instead of evaluating it.
 *   Trade-off: the exported cell now differs from the source ("'=1+1"), so a
 *   round-trip needs to strip the quote. Accepted, because the alternative is
 *   executing attacker-supplied formulas.
 *   Numeric/currency/boolean columns are never guarded: -5 is a number, not an
 *   attack, and it is serialized from a real `number`, so a formula can't be
 *   smuggled through it.
 * - Optional UTF-8 BOM so Excel detects the encoding of non-ASCII data.
 */
export const BOM = '﻿';
const FORMULA_START = /^[=+\-@\t\r]/;

export function guardFormula(text) {
  return FORMULA_START.test(text) ? `'${text}` : text;
}

export function quoteField(text) {
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Serialize one cell to its (not yet quoted) text. */
export function cellText(column, value, { guard = true } = {}) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : '';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  const text = String(value);
  const textual = column.type !== 'number' && column.type !== 'currency' && column.type !== 'boolean';
  return guard && textual ? guardFormula(text) : text;
}

/**
 * @param {{id: string, header: string, type: string}[]} columns
 * @param {Record<string, unknown>[]} rows
 * @param {{bom?: boolean, guard?: boolean}} [options]
 */
export function toCsv(columns, rows, { bom = false, guard = true } = {}) {
  const lines = new Array(rows.length + 1);
  lines[0] = columns.map((c) => quoteField(guardFormula(c.header))).join(',');
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    let line = '';
    for (let c = 0; c < columns.length; c++) {
      if (c > 0) line += ',';
      line += quoteField(cellText(columns[c], row[columns[c].id], { guard }));
    }
    lines[i + 1] = line;
  }
  return (bom ? BOM : '') + lines.join('\r\n') + '\r\n';
}

/** Trigger a browser download; the object URL is revoked right after the click. */
export function downloadText(text, filename, { doc = document, urlApi = URL } = {}) {
  const blob = new Blob([text], { type: 'text/csv;charset=utf-8' });
  const url = urlApi.createObjectURL(blob);
  const a = doc.createElement('a');
  a.href = url;
  a.download = filename;
  a.hidden = true;
  doc.body.appendChild(a);
  try {
    a.click();
  } finally {
    a.remove();
    setTimeout(() => urlApi.revokeObjectURL(url), 0);
  }
}

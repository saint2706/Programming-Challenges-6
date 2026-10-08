/** Display formatting for JSON Resume's partial ISO dates (YYYY, YYYY-MM, YYYY-MM-DD). */

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const PARTIAL = /^(\d{4})(?:-(0[1-9]|1[0-2])(?:-(0[1-9]|[12]\d|3[01]))?)?$/;

/** `2023-05` to `May 2023`, `2020` to `2020`, anything invalid or empty to `''`. */
export function formatDate(iso) {
  const match = typeof iso === 'string' ? PARTIAL.exec(iso) : null;
  if (!match) return '';
  const [, year, month, day] = match;
  if (!month) return year;
  const label = `${MONTHS[Number(month) - 1]} ${year}`;
  return day ? `${MONTHS[Number(month) - 1]} ${Number(day)}, ${year}` : label;
}

/** `May 2023 – Jul 2023`; a start with no end reads `… – Present`. */
export function formatRange(start, end) {
  const from = formatDate(start);
  const to = formatDate(end);
  if (from && to) return from === to ? from : `${from} – ${to}`;
  if (from) return `${from} – Present`;
  return to;
}

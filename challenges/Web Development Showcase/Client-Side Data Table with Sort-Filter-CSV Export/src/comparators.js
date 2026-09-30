/**
 * Type-aware comparators. They only ever see present values: missing values
 * (null / '' / undefined) are pushed last by table-core's `sortUndefined: 'last'`,
 * which keeps them last in BOTH directions (a plain comparator can't do that,
 * because descending order just negates it).
 */
const collator = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });

export const compareNumber = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
export const compareText = (a, b) => collator.compare(a, b);
/** ISO-8601 dates (YYYY-MM-DD) sort correctly as plain strings. */
export const compareDate = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
export const compareBoolean = (a, b) => (a === b ? 0 : a ? 1 : -1);

export function comparatorFor(type) {
  switch (type) {
    case 'number':
    case 'currency':
      return compareNumber;
    case 'date':
      return compareDate;
    case 'boolean':
      return compareBoolean;
    default:
      return compareText;
  }
}

/** True for values that should sort last regardless of direction. */
export const isMissing = (v) => v === null || v === undefined || v === '';

/**
 * Filter helpers (pure). Numeric and date columns accept a tiny expression
 * language so a plain text box can express ranges:
 *   >50000  >=50000  <100  <=100  =42   40000..60000
 * A bare value is a prefix match on the raw value ("2021-0" matches 2021-01..09).
 */
const ISO_PARTIAL = /^\d{4}(-\d{2}(-\d{2})?)?$/;

// Partial ISO dates expand to the first / last day they cover.
const lo = (d) => (d.length === 4 ? `${d}-01-01` : d.length === 7 ? `${d}-01` : d);
const hi = (d) => (d.length === 4 ? `${d}-12-31` : d.length === 7 ? `${d}-31` : d);

function operand(kind, text) {
  const t = text.trim();
  if (kind === 'date') return ISO_PARTIAL.test(t) ? t : null;
  if (t === '') return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

const isPresent = (v) => v !== null && v !== undefined && v !== '';

/**
 * @param {'number'|'date'} kind
 * @returns {{valid: boolean, test: (v: number|string|null|undefined) => boolean}}
 *   An invalid expression matches everything (and `valid: false` lets the UI flag it).
 */
export function parseRangeFilter(kind, raw) {
  const text = raw.trim();
  if (text === '') return { valid: true, test: () => true };
  const invalid = { valid: false, test: () => true };
  const present = (fn) => (v) => isPresent(v) && fn(v);
  const LO = kind === 'date' ? lo : (x) => x;
  const HI = kind === 'date' ? hi : (x) => x;

  const m = /^(>=|<=|>|<|=)\s*(.+)$/.exec(text);
  if (m) {
    const x = operand(kind, m[2]);
    if (x === null) return invalid;
    switch (m[1]) {
      case '>=':
        return { valid: true, test: present((v) => v >= LO(x)) };
      case '>':
        return { valid: true, test: present((v) => v > HI(x)) };
      case '<=':
        return { valid: true, test: present((v) => v <= HI(x)) };
      case '<':
        return { valid: true, test: present((v) => v < LO(x)) };
      default:
        return kind === 'date'
          ? { valid: true, test: present((v) => String(v).startsWith(x)) }
          : { valid: true, test: present((v) => v === x) };
    }
  }
  const range = /^(.+?)\.\.(.+)$/.exec(text);
  if (range) {
    const a = operand(kind, range[1]);
    const b = operand(kind, range[2]);
    if (a === null || b === null) return invalid;
    return { valid: true, test: present((v) => v >= LO(a) && v <= HI(b)) };
  }
  if (kind === 'date' && !/^[\d-]+$/.test(text)) return invalid;
  if (kind === 'number' && !/^-?\d*\.?\d*$/.test(text)) return invalid;
  return { valid: true, test: present((v) => String(v).startsWith(text)) };
}

/** Lowercased haystack of a row's searchable text, cached per row object. */
const haystacks = new WeakMap();
export function searchHaystack(row) {
  let h = haystacks.get(row);
  if (h === undefined) {
    h = Object.values(row)
      .map((v) => (v === null || v === undefined ? '' : typeof v === 'boolean' ? (v ? 'yes true' : 'no false') : String(v)))
      .join('\u0001')
      .toLowerCase();
    haystacks.set(row, h);
  }
  return h;
}

/** All whitespace-separated tokens must appear somewhere in the row (AND). */
export function matchesGlobal(row, query) {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return true;
  const h = searchHaystack(row);
  return tokens.every((t) => h.includes(t));
}

export function debounce(fn, ms) {
  let timer;
  const debounced = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  debounced.cancel = () => clearTimeout(timer);
  return debounced;
}

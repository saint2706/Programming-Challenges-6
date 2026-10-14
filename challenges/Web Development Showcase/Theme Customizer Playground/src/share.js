import { INPUTS, fmt, validate } from './schema.js';

/** A theme as a URL hash: `#brandL=0.55&brandH=264&bodyFont=serif&...`. Readable, and checked on the way back in. */
export function encodeHash(theme) {
  const params = new URLSearchParams();
  for (const input of INPUTS) params.set(input.key, input.kind === 'font' ? theme[input.key] : fmt(theme[input.key]));
  return `#${params}`;
}

const PLAIN_NUMBER = /^-?\d+(\.\d+)?$/;

/** `null` when the hash carries no theme; otherwise the `validate` result (which may be an error). */
export function decodeHash(hash, base) {
  const params = new URLSearchParams(String(hash).replace(/^#/, ''));
  const raw = {};
  for (const input of INPUTS) {
    const v = params.get(input.key);
    if (v === null) continue;
    raw[input.key] = input.kind === 'font' ? v : PLAIN_NUMBER.test(v) ? Number(v) : NaN;
  }
  return Object.keys(raw).length ? validate(raw, base) : null;
}

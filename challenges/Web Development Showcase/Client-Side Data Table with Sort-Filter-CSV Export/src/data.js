import { DEPARTMENTS } from './columns.js';
import { mulberry32 } from './prng.js';

const FIRST = ['Ada', 'Zoë', 'Søren', 'Renée', 'Li', 'Aarav', 'Mateo', 'Fatima', 'Noor', 'Yuki', 'Olu', 'Ines', 'Jürgen', 'Chloé', 'Sam', 'Priya', 'Dmitri', 'Amélie'];
const LAST = ['Smith', 'Müller', "O'Brien", 'Nguyen', 'Khan', 'García', 'Ivanov', '李', 'Okafor', 'Tanaka', 'Silva', 'Kowalski', 'Hernández', 'Patel', 'Lindqvist', 'Brown'];
const CITIES = ['Mumbai', 'Berlin', 'São Paulo', 'Lagos', 'Toronto', 'Seoul', 'Kraków', 'Nairobi', 'Austin', 'Lyon', 'Osaka', 'Bengaluru', 'Reykjavík', 'Sydney'];

/** The awkward CSV cases live in notes; weighted so plain prose still dominates. */
export const AWKWARD_NOTES = [
  'Prefers "async" standups, then leaves',
  'Owns billing, invoicing, and refunds',
  'Line one\nline two',
  'Windows\r\nline ending',
  '=HYPERLINK("http://example.invalid","click")',
  '+1 555 0100 (desk)',
  '-negative vibes only',
  '@team-lead',
  '\t=tab-prefixed',
  '  leading and trailing  ',
  'Émigré — naïve café ☕ 你好',
];
const PLAIN_NOTES = ['Mentors new hires', 'On-call this quarter', 'Relocating soon', 'Speaks at meetups', 'Remote-first'];

// Strip diacritics, then anything non a-z (CJK surnames collapse to a fallback).
const slug = (s) => s.normalize('NFD').replace(/[^a-zA-Z]/g, '').toLowerCase() || 'user';

const pick = (rnd, list) => list[Math.floor(rnd() * list.length)];
const pad2 = (n) => String(n).padStart(2, '0');

/** Generate `count` deterministic rows. Values are raw (null = missing). */
export function generateRows(count, seed = 20260929) {
  const rnd = mulberry32(seed);
  const rows = new Array(count);
  for (let i = 0; i < count; i++) {
    const first = pick(rnd, FIRST);
    const last = pick(rnd, LAST);
    const roll = rnd();
    let notes;
    if (roll < 0.1) notes = null;
    else if (roll < 0.18) notes = '';
    else if (roll < 0.4) notes = pick(rnd, AWKWARD_NOTES);
    else notes = pick(rnd, PLAIN_NOTES);

    const year = 2015 + Math.floor(rnd() * 11);
    const month = 1 + Math.floor(rnd() * 12);
    const day = 1 + Math.floor(rnd() * 28);
    rows[i] = {
      id: i + 1,
      name: `${first} ${last}`,
      email: `${slug(first)}.${slug(last)}${i + 1}@example.com`,
      city: rnd() < 0.02 ? null : pick(rnd, CITIES),
      department: pick(rnd, DEPARTMENTS),
      salary: rnd() < 0.03 ? null : Math.round((30000 + rnd() * 220000) * 100) / 100,
      joined: rnd() < 0.03 ? null : `${year}-${pad2(month)}-${pad2(day)}`,
      active: rnd() < 0.8,
      notes,
    };
  }
  return rows;
}

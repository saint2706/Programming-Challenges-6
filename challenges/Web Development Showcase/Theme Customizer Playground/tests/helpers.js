import fc from 'fast-check';
import { readFileSync } from 'node:fs';
import { FONTS, INPUTS } from '../src/schema.js';

/** Any valid theme: every input at a random step within its range. */
export const themeArb = fc.record(
  Object.fromEntries(
    INPUTS.map((i) => [
      i.key,
      i.kind === 'font'
        ? fc.constantFrom(...Object.keys(FONTS))
        : fc.integer({ min: 0, max: Math.floor((i.max - i.min) / i.step + 1e-9) }).map((n) => +(i.min + n * i.step).toFixed(4)),
    ]),
  ),
);

export const coreCss = readFileSync(new URL('../src/theme.css', import.meta.url), 'utf8');

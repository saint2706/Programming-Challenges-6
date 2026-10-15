// Turns the committed GISTEMP v4 CSV into the JSON the page imports. Run `pnpm run data` after
// replacing data/GLB.Ts+dSST.csv with a fresh download; the page never fetches anything at run time.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseGistemp } from '../src/gistemp.js';

const RETRIEVED = '2026-10-15';
const SOURCE = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv';

const here = (path) => fileURLToPath(new URL(path, import.meta.url));
const csv = readFileSync(here('../data/GLB.Ts+dSST.csv'), 'utf8');
const { title, years } = parseGistemp(csv);
const last = years.at(-1).year;

const out = {
    dataset: 'NASA GISS GISTEMP v4, Land-Ocean Temperature Index, Global Means (annual J-D)',
    dataset_title: title,
    unit: 'degrees Celsius, anomaly relative to the 1951-1980 base period',
    source: SOURCE,
    retrieved: RETRIEVED,
    licence:
        'Public domain (NASA GISS data). Attribute: NASA GISS, GISTEMP v4. https://data.giss.nasa.gov/gistemp/',
    note: `Annual means for complete years only (1880-${last}); months marked *** are not used. The raw CSV is committed as data/GLB.Ts+dSST.csv.`,
    years,
};
writeFileSync(here('../src/data/gistemp-annual.json'), `${JSON.stringify(out, null, 2)}\n`);
console.log(`wrote ${years.length} years (${years[0].year}-${last})`);

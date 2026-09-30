// Node benchmark of the DOM-free pipeline (no rendering): `npm run bench [rows]`.
// "cold" = first use of a column/filter (builds lazy sort keys / lowercase caches);
// "warm" = the same interaction again, which is what a user feels after the first click.
import { performance } from 'node:perf_hooks';
import { COLUMNS } from '../src/columns.js';
import { toCsv } from '../src/csv.js';
import { generateRows } from '../src/data.js';
import { createTableModel } from '../src/tableModel.js';

const N = Number(process.argv[2] ?? 100_000);
const time = (label, fn) => {
  const t0 = performance.now();
  const out = fn();
  const ms = performance.now() - t0;
  console.log(`${label.padEnd(50)} ${ms.toFixed(1).padStart(8)} ms${out === undefined ? '' : `   (${out})`}`);
};

let rows;
time(`generate ${N} rows`, () => (rows = generateRows(N)).length);
const m = createTableModel({ rows });
const step = (label, fn) => time(label, () => (fn(), `${m.count} rows`));

step('no sort/filter (identity order)', () => m.order);
step('cold: sort salary asc', () => m.toggleSort('salary') || m.order);
step('warm: salary desc', () => m.toggleSort('salary') || m.order);
step('warm: salary off', () => m.toggleSort('salary') || m.order);
step('cold: sort name asc (text, 288 distinct)', () => m.toggleSort('name') || m.order);
m.toggleSort('name'), m.toggleSort('name');
step('cold: sort email asc (text, 100k distinct)', () => m.toggleSort('email') || m.order);
m.toggleSort('email'), m.toggleSort('email');
step('cold: sort joined asc (date)', () => m.toggleSort('joined') || m.order);
m.toggleSort('joined'), m.toggleSort('joined');
step('multi: department, then salary asc', () => {
  m.toggleSort('department');
  m.toggleSort('salary', true);
  m.order;
});
m.resetAll();
step('cold: global search "smith"', () => m.setGlobalFilter('smith') || m.order);
step('warm: global search "smith x"', () => m.setGlobalFilter('smith x') || m.order);
step('warm: global search "garcía berlin" (2 tokens)', () => m.setGlobalFilter('garcía berlin') || m.order);
m.resetAll();
step('column filter: salary >=100000', () => m.setColumnFilter('salary', '>=100000') || m.order);
step('+ department = Data', () => m.setColumnFilter('department', 'Data') || m.order);
step('+ sort name asc on the filtered set', () => m.toggleSort('name') || m.order);
m.resetAll();
time('CSV: visibleRows() + serialize (all rows)', () => {
  const csv = toCsv(COLUMNS, m.visibleRows(), { bom: true });
  return `${(csv.length / 1e6).toFixed(1)} MB`;
});

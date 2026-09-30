// Baseline: what @tanstack/table-core's own row models cost at 100k rows, to justify
// the index-permutation engine in src/engine.js. Run: `npm run bench:baseline [rows]`.
import { performance } from 'node:perf_hooks';
import {
  createTable,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
} from '@tanstack/table-core';
import { COLUMNS } from '../src/columns.js';
import { generateRows } from '../src/data.js';

const N = Number(process.argv[2] ?? 100_000);
const rows = generateRows(N);
const heap = () => process.memoryUsage().heapUsed / 1e6;
const time = (label, fn) => {
  const h0 = heap();
  const t0 = performance.now();
  const out = fn();
  const ms = performance.now() - t0;
  console.log(`${label.padEnd(46)} ${ms.toFixed(0).padStart(7)} ms   heap +${(heap() - h0).toFixed(0)} MB   (${out})`);
};

let state = { sorting: [], columnFilters: [], globalFilter: undefined };
const table = createTable({
  data: rows,
  columns: COLUMNS.map((c) => ({
    id: c.id,
    accessorKey: c.id,
    header: c.header,
    sortUndefined: 'last',
    sortingFn: c.type === 'text' || c.type === 'enum' ? 'alphanumeric' : 'auto',
  })),
  getCoreRowModel: getCoreRowModel(),
  getSortedRowModel: getSortedRowModel(),
  getFilteredRowModel: getFilteredRowModel(),
  state,
  onStateChange() {},
  renderFallbackValue: null,
  autoResetAll: false,
});
const set = (patch) => {
  state = { ...state, ...patch };
  table.setOptions((o) => ({ ...o, state }));
};

global.gc?.();
time('getCoreRowModel (wrap every row)', () => table.getCoreRowModel().rows.length);
set({ sorting: [{ id: 'salary', desc: false }] });
time('getSortedRowModel: salary asc', () => table.getSortedRowModel().rows.length);
set({ sorting: [{ id: 'name', desc: false }] });
time('getSortedRowModel: name asc', () => table.getSortedRowModel().rows.length);
set({ sorting: [], columnFilters: [{ id: 'city', value: 'ber' }] });
time('getFilteredRowModel: city contains "ber"', () => table.getFilteredRowModel().rows.length);

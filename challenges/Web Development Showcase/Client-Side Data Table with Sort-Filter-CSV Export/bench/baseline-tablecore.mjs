// Baseline: what @tanstack/table-core's own row models cost at 100k rows, to justify
// the index-permutation engine in src/engine.js. Run: `npm run bench:baseline [rows]`.
import { performance } from 'node:perf_hooks';
import {
  columnFilteringFeature,
  constructTable,
  createFilteredRowModel,
  createSortedRowModel,
  filterFns,
  rowSortingFeature,
  sortFns,
  tableFeatures,
} from '@tanstack/table-core';
import { storeReactivityBindings } from '@tanstack/table-core/store-reactivity-bindings';
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

const features = tableFeatures({
  coreReactivityFeature: storeReactivityBindings(),
  rowSortingFeature,
  columnFilteringFeature,
  sortedRowModel: createSortedRowModel(),
  filteredRowModel: createFilteredRowModel(),
  sortFns,
  filterFns,
});
const table = constructTable({
  features,
  data: rows,
  columns: COLUMNS.map((c) => ({
    id: c.id,
    accessorKey: c.id,
    header: c.header,
    sortUndefined: 'last',
    sortFn: c.type === 'text' || c.type === 'enum' ? 'alphanumeric' : 'auto',
  })),
  renderFallbackValue: null,
  autoResetAll: false,
});
const set = ({ sorting = [], columnFilters = [] }) => {
  table.setSorting(sorting);
  table.setColumnFilters(columnFilters);
};

global.gc?.();
time('getCoreRowModel (wrap every row)', () => table.getCoreRowModel().rows.length);
set({ sorting: [{ id: 'salary', desc: false }] });
time('getSortedRowModel: salary asc', () => table.getSortedRowModel().rows.length);
set({ sorting: [{ id: 'name', desc: false }] });
time('getSortedRowModel: name asc', () => table.getSortedRowModel().rows.length);
set({ sorting: [], columnFilters: [{ id: 'city', value: 'ber' }] });
time('getFilteredRowModel: city contains "ber"', () => table.getFilteredRowModel().rows.length);

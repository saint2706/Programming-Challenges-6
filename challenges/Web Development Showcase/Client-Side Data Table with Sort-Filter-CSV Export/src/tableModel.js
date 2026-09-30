import { createTable, getCoreRowModel } from '@tanstack/table-core';
import { COLUMNS } from './columns.js';
import { createEngine } from './engine.js';

/**
 * Headless, DOM-free table model.
 *
 * `@tanstack/table-core` owns the state machine (sorting cycle asc -> desc ->
 * off, shift multi-sort, column/global filter state). Row computation is manual
 * (`manualSorting` / `manualFiltering`) and delegated to `createEngine`, because
 * table-core's per-row objects are too heavy for 100k rows (see README).
 */
export function createTableModel({ rows, columns = COLUMNS, onChange = () => {} }) {
  const engine = createEngine(rows, columns);
  let state;
  let view = null;

  const table = createTable({
    data: [], // rows live in the engine; table-core never materializes them
    columns: columns.map((c) => ({
      id: c.id,
      header: c.header,
      accessorFn: (row) => row[c.id], // required for getCanSort/getCanFilter
      filterFn: () => true,
      meta: c,
    })),
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
    manualFiltering: true,
    enableMultiSort: true,
    enableSortingRemoval: true,
    sortDescFirst: false, // first click is always ascending, for every type
    autoResetAll: false,
    renderFallbackValue: null,
    state: {},
    onStateChange() {},
  });
  state = table.initialState;
  const sync = () =>
    table.setOptions((prev) => ({
      ...prev,
      state,
      onStateChange: (updater) => {
        state = typeof updater === 'function' ? updater(state) : updater;
        view = null;
        sync();
        onChange(model);
      },
    }));
  sync();

  const model = {
    table,
    total: rows.length,
    /** Row indices (Int32Array) of the current filtered + sorted view. */
    get order() {
      view ??= engine.query(table.getState());
      return view;
    },
    get count() {
      return model.order.length;
    },
    getRow(i) {
      return rows[model.order[i]];
    },
    /** Raw row objects of the whole current view (used for CSV export). */
    visibleRows() {
      const order = model.order;
      const out = new Array(order.length);
      for (let i = 0; i < order.length; i++) out[i] = rows[order[i]];
      return out;
    },
    get sorting() {
      return table.getState().sorting;
    },
    setGlobalFilter(text) {
      table.setGlobalFilter(text ? text : undefined);
    },
    setColumnFilter(id, value) {
      table.getColumn(id).setFilterValue(value === '' ? undefined : value);
    },
    toggleSort(id, multi = false) {
      table.getColumn(id).toggleSorting(undefined, multi);
    },
    resetAll() {
      table.resetSorting(true);
      table.resetColumnFilters(true);
      table.resetGlobalFilter(true);
    },
  };
  return model;
}

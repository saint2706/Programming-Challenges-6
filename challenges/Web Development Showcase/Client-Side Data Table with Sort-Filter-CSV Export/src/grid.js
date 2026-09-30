import {
  Virtualizer,
  elementScroll,
  observeElementOffset,
  observeElementRect,
} from '@tanstack/virtual-core';
import { COLUMNS, formatCell, isNumeric } from './columns.js';
import { toCsv, downloadText } from './csv.js';
import { debounce, parseRangeFilter } from './filters.js';
import { createTableModel } from './tableModel.js';

const SORT_GLYPH = { asc: '▲', desc: '▼' };

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v; // never innerHTML: cell data is untrusted
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (k.startsWith('aria-') || k === 'role' || k === 'type' || k === 'id' || k === 'placeholder' || k === 'title') {
      node.setAttribute(k, v);
    } else node[k] = v;
  }
  for (const c of children) node.append(c);
  return node;
}

/**
 * Mount the data table into `root`.
 *
 * Layout: one scroll container (`role=grid`) holds a sticky header and a tall
 * spacer whose absolutely-positioned rows are windowed by @tanstack/virtual-core.
 * Focus stays on the grid; the active cell is exposed with aria-activedescendant.
 */
export function mountDataTable(root, { rows, columns = COLUMNS, rowHeight = 36, overscan = 6, debounceMs = 150 }) {
  const uid = `dt${Math.random().toString(36).slice(2, 7)}`;
  const totalWidth = columns.reduce((s, c) => s + c.width, 0);
  const template = columns.map((c) => `${c.width}px`).join(' ');

  const model = createTableModel({ rows, columns, onChange: () => refresh(true) });

  // ---- static chrome -------------------------------------------------------
  const search = el('input', { type: 'search', id: `${uid}-search`, placeholder: 'Search all columns…', 'aria-label': 'Search all columns' });
  const bomBox = el('input', { type: 'checkbox', id: `${uid}-bom`, checked: true });
  const exportBtn = el('button', { type: 'button', class: 'btn primary', text: 'Export CSV' });
  const resetBtn = el('button', { type: 'button', class: 'btn', text: 'Reset' });
  const status = el('p', { class: 'status', role: 'status', 'aria-live': 'polite' });
  const toolbar = el('div', { class: 'toolbar' }, [
    search,
    el('label', { class: 'check', htmlFor: `${uid}-bom`, title: 'Adds a UTF-8 byte-order mark so Excel reads accents/CJK correctly' }, [bomBox, ' Excel BOM']),
    exportBtn,
    resetBtn,
  ]);

  const headerCells = new Map();
  const filterInputs = new Map();
  const headerRow = el('div', { class: 'row head', role: 'row', 'aria-rowindex': '1' });
  headerRow.style.gridTemplateColumns = template;
  headerRow.style.width = `${totalWidth}px`;
  columns.forEach((col, ci) => {
    const button = el('button', { type: 'button', class: 'sort', 'aria-label': `Sort by ${col.header}` });
    button.append(el('span', { class: 'label', text: col.header }), el('span', { class: 'glyph', 'aria-hidden': 'true' }));
    button.addEventListener('click', (e) => model.toggleSort(col.id, e.shiftKey));

    let filter;
    if (col.type === 'enum' || col.type === 'boolean') {
      const options = col.type === 'enum' ? col.options.map((o) => [o, o]) : [['true', 'Yes'], ['false', 'No']];
      filter = el('select', { class: 'filter', 'aria-label': `Filter ${col.header}` }, [
        el('option', { value: '', text: 'All' }),
        ...options.map(([value, label]) => el('option', { value, text: label })),
      ]);
      filter.addEventListener('change', () => model.setColumnFilter(col.id, filter.value));
    } else {
      const hint = col.type === 'date' ? '>=2021' : isNumeric(col.type) ? '>100' : 'contains…';
      filter = el('input', { type: 'text', class: 'filter', placeholder: hint, 'aria-label': `Filter ${col.header}` });
      const apply = debounce(() => {
        model.setColumnFilter(col.id, filter.value.trim());
        if (isNumeric(col.type) || col.type === 'date') {
          const kind = col.type === 'date' ? 'date' : 'number';
          filter.setAttribute('aria-invalid', String(!parseRangeFilter(kind, filter.value).valid));
        }
      }, debounceMs);
      filter.addEventListener('input', apply);
    }
    filterInputs.set(col.id, filter);
    const cell = el('div', { class: `cell headcell${isNumeric(col.type) ? ' num' : ''}`, role: 'columnheader', 'aria-colindex': String(ci + 1), 'aria-sort': 'none' }, [button, filter]);
    headerCells.set(col.id, cell);
    headerRow.append(cell);
  });

  const body = el('div', { class: 'body', role: 'rowgroup' });
  body.style.width = `${totalWidth}px`;
  const scroller = el('div', {
    class: 'grid',
    role: 'grid',
    tabIndex: 0,
    'aria-label': 'Employee data table',
    'aria-colcount': String(columns.length),
    'aria-multiselectable': 'false',
  }, [headerRow, body]);
  root.replaceChildren(toolbar, scroller, status);

  // ---- virtualization --------------------------------------------------------
  let headHeight = 0;
  const virtualizer = new Virtualizer({
    count: 0,
    getScrollElement: () => scroller,
    estimateSize: () => rowHeight,
    overscan,
    scrollMargin: 0,
    scrollPaddingStart: 0,
    observeElementRect,
    observeElementOffset,
    scrollToFn: elementScroll,
    onChange: () => renderRows(),
  });
  const unmount = virtualizer._didMount();

  const rowCache = new Map(); // index -> element, valid for the current view
  let active = { row: 0, col: 0 };

  const cellId = (r, c) => `${uid}-r${r}-c${c}`;

  function buildRow(index) {
    const data = model.getRow(index);
    const row = el('div', { class: index % 2 ? 'row alt' : 'row', role: 'row', 'aria-rowindex': String(index + 2), dataset: { index: String(index) } });
    row.style.gridTemplateColumns = template;
    row.style.height = `${rowHeight}px`;
    columns.forEach((col, ci) => {
      const cell = el('div', {
        class: `cell${isNumeric(col.type) ? ' num' : ''}`,
        role: 'gridcell',
        id: cellId(index, ci),
        'aria-colindex': String(ci + 1),
        text: formatCell(col, data[col.id]),
      });
      // Notes can contain newlines; show them on one line but keep the full text on hover.
      if (col.type === 'text') cell.title = String(data[col.id] ?? '');
      row.append(cell);
    });
    return row;
  }

  function markActive() {
    scroller.querySelectorAll('.is-active').forEach((n) => n.classList.remove('is-active'));
    const node = document.getElementById(cellId(active.row, active.col));
    if (node) {
      node.classList.add('is-active');
      scroller.setAttribute('aria-activedescendant', node.id);
    } else {
      scroller.removeAttribute('aria-activedescendant');
    }
  }

  function renderRows() {
    const items = virtualizer.getVirtualItems();
    const keep = new Set();
    const fragment = [];
    for (const item of items) {
      keep.add(item.index);
      let row = rowCache.get(item.index);
      if (!row) {
        row = buildRow(item.index);
        rowCache.set(item.index, row);
      }
      row.style.transform = `translateY(${item.start - virtualizer.options.scrollMargin}px)`;
      fragment.push(row);
    }
    for (const index of rowCache.keys()) if (!keep.has(index)) rowCache.delete(index);
    body.replaceChildren(...fragment);
    markActive();
  }

  function updateHeader() {
    const sorting = model.sorting;
    for (const col of columns) {
      const cell = headerCells.get(col.id);
      const pos = sorting.findIndex((s) => s.id === col.id);
      const glyph = cell.querySelector('.glyph');
      if (pos === -1) {
        cell.setAttribute('aria-sort', 'none');
        glyph.textContent = '';
      } else {
        const dir = sorting[pos].desc ? 'desc' : 'asc';
        cell.setAttribute('aria-sort', dir === 'asc' ? 'ascending' : 'descending');
        glyph.textContent = SORT_GLYPH[dir] + (sorting.length > 1 ? String(pos + 1) : '');
      }
    }
  }

  let lastMs = 0;
  function refresh(resetScroll) {
    const t0 = performance.now();
    rowCache.clear();
    const count = model.count; // forces the engine to (re)compute the view
    lastMs = performance.now() - t0;
    headHeight = headerRow.offsetHeight;
    virtualizer.setOptions({
      ...virtualizer.options,
      count,
      scrollMargin: headHeight,
      scrollPaddingStart: headHeight,
    });
    virtualizer._willUpdate();
    virtualizer.measure();
    body.style.height = `${Math.round(virtualizer.getTotalSize())}px`;
    scroller.setAttribute('aria-rowcount', String(count + 1));
    if (resetScroll) {
      scroller.scrollTop = 0;
      scroller.dispatchEvent(new Event('scroll')); // sync the virtualizer now, not a frame later
      active = { row: 0, col: active.col };
    }
    active.row = Math.min(active.row, Math.max(0, count - 1));
    updateHeader();
    status.textContent = count === 0
      ? `No rows match. (${model.total.toLocaleString()} total)`
      : `Showing ${count.toLocaleString()} of ${model.total.toLocaleString()} rows · view computed in ${lastMs.toFixed(0)} ms`;
    renderRows();
  }

  // ---- keyboard navigation (ARIA grid pattern) -------------------------------
  function moveTo(row, col) {
    const count = model.count;
    if (count === 0) return;
    active = { row: Math.max(0, Math.min(count - 1, row)), col: Math.max(0, Math.min(columns.length - 1, col)) };
    virtualizer.scrollToIndex(active.row, { align: 'auto' });
    renderRows();
  }

  scroller.addEventListener('keydown', (e) => {
    if (e.target !== scroller) return; // typing in a filter input, etc.
    const page = Math.max(1, Math.floor((scroller.clientHeight - headHeight) / rowHeight) - 1);
    const { row, col } = active;
    const handlers = {
      ArrowDown: () => moveTo(row + 1, col),
      ArrowUp: () => moveTo(row - 1, col),
      ArrowRight: () => moveTo(row, col + 1),
      ArrowLeft: () => moveTo(row, col - 1),
      PageDown: () => moveTo(row + page, col),
      PageUp: () => moveTo(row - page, col),
      Home: () => (e.ctrlKey ? moveTo(0, 0) : moveTo(row, 0)),
      End: () => (e.ctrlKey ? moveTo(model.count - 1, columns.length - 1) : moveTo(row, columns.length - 1)),
    };
    const handler = handlers[e.key];
    if (handler) {
      e.preventDefault();
      handler();
    }
  });
  scroller.addEventListener('click', (e) => {
    const cell = e.target.closest?.('[role=gridcell]');
    if (!cell) return;
    const [, r, c] = /-r(\d+)-c(\d+)$/.exec(cell.id) ?? [];
    if (r !== undefined) {
      active = { row: Number(r), col: Number(c) };
      markActive();
    }
  });

  // ---- toolbar ---------------------------------------------------------------
  search.addEventListener('input', debounce(() => model.setGlobalFilter(search.value.trim()), debounceMs));
  exportBtn.addEventListener('click', () => {
    // Exports the WHOLE current filtered+sorted view, not just the rendered window.
    const csv = toCsv(columns, model.visibleRows(), { bom: bomBox.checked });
    downloadText(csv, 'data-table-export.csv');
  });
  resetBtn.addEventListener('click', () => {
    model.resetAll();
    search.value = '';
    for (const input of filterInputs.values()) {
      input.value = '';
      input.removeAttribute('aria-invalid');
    }
  });

  refresh(false);
  const resizeObserver = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => refresh(false));
  resizeObserver?.observe(headerRow);

  return {
    model,
    virtualizer,
    scroller,
    exportCsv: (bom = false) => toCsv(columns, model.visibleRows(), { bom }),
    destroy() {
      resizeObserver?.disconnect();
      unmount();
      root.replaceChildren();
    },
  };
}

import './style.css';
import { generateRows } from './data.js';
import { mountDataTable } from './grid.js';

const root = document.getElementById('app');
const sizeSelect = document.getElementById('row-count');
let table = null;

function mount(count) {
  table?.destroy();
  const rows = generateRows(count);
  table = mountDataTable(root, { rows });
}

sizeSelect.addEventListener('change', () => mount(Number(sizeSelect.value)));
mount(Number(sizeSelect.value));

// Handy for poking at the model from the devtools console.
window.dataTable = () => table;

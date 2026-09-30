/**
 * Column metadata shared by the table model, the DOM grid and the CSV exporter.
 * `type` drives comparators, filter syntax, alignment and CSV formula-guarding.
 */
export const DEPARTMENTS = [
  'Engineering',
  'Data',
  'Design',
  'Sales',
  'Support',
  'Finance',
  'Legal',
  'Operations',
];

export const COLUMNS = [
  { id: 'id', header: 'ID', type: 'number', width: 84 },
  { id: 'name', header: 'Name', type: 'text', width: 200 },
  { id: 'email', header: 'Email', type: 'text', width: 250 },
  { id: 'city', header: 'City', type: 'text', width: 150 },
  { id: 'department', header: 'Department', type: 'enum', width: 150, options: DEPARTMENTS },
  { id: 'salary', header: 'Salary', type: 'currency', width: 130 },
  { id: 'joined', header: 'Joined', type: 'date', width: 130 },
  { id: 'active', header: 'Active', type: 'boolean', width: 100 },
  { id: 'notes', header: 'Notes', type: 'text', width: 290 },
];

export const isNumeric = (type) => type === 'number' || type === 'currency';

const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
const dateFmt = new Intl.DateTimeFormat('en-GB', {
  day: '2-digit',
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
});

/** Display string for a cell (UI only; CSV exports raw values). */
export function formatCell(column, value) {
  if (value === null || value === undefined) return '';
  switch (column.type) {
    case 'currency':
      return currency.format(value);
    case 'date':
      return dateFmt.format(new Date(`${value}T00:00:00Z`));
    case 'boolean':
      return value ? 'Yes' : 'No';
    default:
      return String(value);
  }
}

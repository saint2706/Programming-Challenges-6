import './style.css';
import './theme.css';
import coreCss from './theme.css?raw';
import { hexToOklch, isInGamut, toHex } from './color.js';
import { TARGETS, apcaLevel, nearestPassing, wcagLevel } from './contrast.js';
import { SCHEMES, contrastRows, deriveColors, fallbackStyle, pairPasses, pairs } from './derive.js';
import { FORMATS } from './exporters.js';
import { MAX_BYTES, parseThemeFile } from './importers.js';
import { PRESETS } from './presets.js';
import { FONTS, INPUTS, cssValue, defaults, fmt } from './schema.js';
import { decodeHash, encodeHash } from './share.js';
import { createStore } from './storage.js';

const $ = (selector) => document.querySelector(selector);

/** Build an element. Text always goes in as text, never as markup. */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'text') node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

/** The CSS features the derivation needs. Without them the colours are computed in JS (`?fallback` forces that, for testing). */
const NATIVE =
  !new URLSearchParams(location.search).has('fallback') &&
  CSS.supports('color', 'oklch(from red l c h)') &&
  CSS.supports('color', 'light-dark(red, blue)') &&
  CSS.supports('color', 'color-mix(in oklch, red, blue)');

let storage = null;
try {
  storage = window.localStorage;
} catch {
  // blocked: the store reports it
}
const store = createStore(storage);

let theme = defaults();
let target = TARGETS[0];
let animateTimer = 0;

const say = (message) => {
  $('#notice').textContent = message;
};

// ---- preview panes ----------------------------------------------------------------------------

const panes = SCHEMES.map((scheme) => {
  const pane = el('section', { class: 'pane theme-scope', 'data-scheme': scheme, 'aria-label': `${scheme === 'light' ? 'Light' : 'Dark'} mode preview` });
  pane.append($('#sample').content.cloneNode(true));
  pane.addEventListener('click', (event) => event.target.closest('a') && event.preventDefault());
  $('#panes').append(pane);
  return pane;
});

function applyToPanes(animate) {
  clearTimeout(animateTimer);
  for (const pane of panes) {
    pane.classList.toggle('animating', animate);
    for (const input of INPUTS) pane.style.setProperty(input.css, cssValue(input, theme[input.key]));
    if (!NATIVE) for (const [name, value] of Object.entries(fallbackStyle(theme, pane.dataset.scheme))) pane.style.setProperty(name, value);
  }
  if (animate) animateTimer = setTimeout(() => panes.forEach((p) => p.classList.remove('animating')), 500);
}

// ---- controls ---------------------------------------------------------------------------------

const fields = new Map();
const FONT_LABELS = { system: 'System sans', serif: 'Serif', mono: 'Monospace', rounded: 'Rounded' };
const TRACKS = { brandL: 'l', brandLDark: 'l', brandC: 'c', brandH: 'h', neutralH: 'h' };

function buildControls() {
  const form = $('#controls');
  form.addEventListener('submit', (event) => event.preventDefault());
  for (const group of [...new Set(INPUTS.map((i) => i.group))]) {
    const set = el('fieldset', {}, el('legend', { text: group }));
    if (group === 'Color') set.append(hexField());
    for (const input of INPUTS.filter((i) => i.group === group)) set.append(field(input));
    if (group === 'Color') set.append(el('p', { id: 'gamut-note', class: 'warn', hidden: '' }));
    form.append(set);
  }
}

function field(input) {
  const id = `in-${input.key}`;
  const wrap = el('div', { class: 'field' }, el('label', { for: id, text: input.label }));
  if (input.kind === 'font') {
    const select = el('select', { id });
    for (const name of Object.keys(FONTS)) select.append(el('option', { value: name, text: FONT_LABELS[name] }));
    select.addEventListener('change', () => commit({ [input.key]: select.value }));
    fields.set(input.key, { select });
    wrap.append(select);
    return wrap;
  }
  const attrs = { min: input.min, max: input.max, step: input.step };
  const range = el('input', { type: 'range', id, ...attrs, ...(TRACKS[input.key] ? { 'data-track': TRACKS[input.key] } : {}) });
  const number = el('input', { type: 'number', id: `num-${input.key}`, 'aria-label': `${input.label}, exact value`, ...attrs });
  range.addEventListener('input', () => commit({ [input.key]: Number(range.value) }));
  number.addEventListener('input', () => {
    const v = Number(number.value);
    const ok = number.value.trim() !== '' && Number.isFinite(v) && v >= input.min && v <= input.max;
    number.toggleAttribute('aria-invalid', !ok);
    if (ok) commit({ [input.key]: v });
  });
  number.addEventListener('change', () => {
    number.removeAttribute('aria-invalid');
    number.value = fmt(theme[input.key]);
  });
  fields.set(input.key, { range, number });
  wrap.append(el('div', { class: 'field-row' }, range, number));
  return wrap;
}

/** Brand colour from a hex code: sets lightness (light mode), chroma and hue. */
function hexField() {
  const input = el('input', { type: 'text', id: 'brand-hex', spellcheck: 'false', autocomplete: 'off', placeholder: '#3b4cca' });
  const apply = () => {
    const c = hexToOklch(input.value);
    if (!c) {
      input.setAttribute('aria-invalid', 'true');
      say('That is not a hex colour. Use #rgb or #rrggbb.');
      return;
    }
    input.removeAttribute('aria-invalid');
    const l = INPUTS.find((i) => i.key === 'brandL');
    const ch = INPUTS.find((i) => i.key === 'brandC');
    commit({ brandL: Math.min(l.max, Math.max(l.min, +c.l.toFixed(3))), brandC: Math.min(ch.max, +c.c.toFixed(3)), brandH: Math.round(c.h) });
    say(`Brand colour set from ${input.value.trim()}.`);
  };
  input.addEventListener('change', apply);
  return el('div', { class: 'field' }, el('label', { for: 'brand-hex', text: 'Brand colour from hex' }), input);
}

function syncControls() {
  for (const input of INPUTS) {
    const f = fields.get(input.key);
    const value = theme[input.key];
    if (f.select) f.select.value = value;
    else {
      f.range.value = value;
      if (document.activeElement !== f.number) f.number.value = fmt(value);
    }
  }
  const form = $('#controls').style;
  form.setProperty('--ui-l', fmt(theme.brandL));
  form.setProperty('--ui-c', fmt(theme.brandC));
  form.setProperty('--ui-h', fmt(theme.brandH));
  const clipped = SCHEMES.map((s) => ({ s, c: deriveColors(theme, s).brand })).filter(({ c }) => !isInGamut(c));
  const note = $('#gamut-note');
  note.hidden = !clipped.length;
  note.textContent = clipped.map(({ s, c }) => `The ${s} brand colour is outside sRGB; browsers clip it, so it paints as ${toHex(c)}.`).join(' ');
}

// ---- contrast ---------------------------------------------------------------------------------

/** Colour a cell only for the metric the fix target is about. */
const verdict = (metric, value) => (target.metric !== metric ? '' : Math.abs(value) >= target.min ? 'pass' : 'fail');
const labelOf = (key) => INPUTS.find((i) => i.key === key).label.toLowerCase();

function renderContrast() {
  const rows = contrastRows(theme);
  const body = $('#contrast tbody');
  body.replaceChildren(
    ...rows.map((row) => {
      const pass = target.passes(row);
      const swatch = el('span', { class: 'swatch', text: 'Aa' });
      swatch.style.color = row.fgHex;
      swatch.style.background = row.bgHex;
      const fixCell = el('td');
      if (pass) fixCell.textContent = 'Passes';
      else if (!row.fixKey) fixCell.textContent = 'Derived from the neutral inputs';
      else {
        const input = INPUTS.find((i) => i.key === row.fixKey);
        const value = nearestPassing(theme, input, pairPasses(pairs.find((p) => p.id === row.id), row.scheme, target));
        if (value === null) fixCell.textContent = 'No value in range passes';
        else {
          const button = el('button', { type: 'button', class: 'fix', text: `Set to ${fmt(value)}`, 'aria-label': `Set ${labelOf(row.fixKey)} to ${fmt(value)} to fix ${row.label} in ${row.scheme} mode` });
          button.addEventListener('click', () => {
            commit({ [row.fixKey]: value });
            say(`Set ${labelOf(row.fixKey)} to ${fmt(value)}.`);
            $('#contrast-region').focus();
          });
          fixCell.append(button);
        }
      }
      const tr = el('tr', { 'data-pair': row.id, 'data-scheme': row.scheme, 'data-pass': String(pass) });
      tr.append(
        el('td', { text: row.scheme === 'dark' ? 'Dark' : 'Light' }),
        el('td', { text: row.label }),
        el('td', {}, swatch),
        el('td', { class: verdict('wcag', row.wcag), text: `${row.wcag.toFixed(2)}:1 ${wcagLevel(row.wcag)}` }),
        el('td', { class: verdict('apca', row.apca), text: `Lc ${row.apca.toFixed(1)} ${apcaLevel(row.apca)}` }),
        fixCell,
      );
      return tr;
    }),
  );
}

// ---- export, import, share ---------------------------------------------------------------------

let format = FORMATS[0];

function renderExport() {
  $('#export-text').value = format.build(theme, coreCss);
  $('#share-url').value = new URL(encodeHash(theme), location.href).href;
}

const flash = (button, text) => {
  const old = button.textContent;
  button.textContent = text;
  setTimeout(() => (button.textContent = old), 1500);
};

async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    flash(button, 'Copied');
    say('Copied to the clipboard.');
  } catch {
    say('Copying is blocked in this browser. Select the text and copy it yourself.');
  }
}

function showImport(result, name) {
  const box = $('#import-result');
  box.replaceChildren();
  if (result.ok) {
    box.append(el('p', { class: 'good', text: `Imported ${result.found} values from ${name} (${result.format === 'css' ? 'CSS' : 'design tokens'}). The rest keep their current setting.` }));
    return;
  }
  const list = el('ul');
  for (const message of result.errors.slice(0, 8)) list.append(el('li', { text: message }));
  if (result.errors.length > 8) list.append(el('li', { text: `and ${result.errors.length - 8} more.` }));
  box.append(el('p', { class: 'bad', text: `Could not import ${name}. Nothing was changed.` }), list);
}

function importText(text, name) {
  const result = parseThemeFile(text, theme);
  showImport(result, name);
  if (result.ok) commit(result.theme, { animate: true });
}

function buildExportControls() {
  for (const f of FORMATS) {
    const radio = el('input', { type: 'radio', name: 'format', value: f.id });
    radio.checked = f === format;
    radio.addEventListener('change', () => {
      format = f;
      renderExport();
    });
    $('#formats').append(el('label', {}, radio, ' ', el('span', { text: f.label })), ' ');
  }
  $('#copy').addEventListener('click', (e) => copy($('#export-text').value, e.currentTarget));
  $('#copy-link').addEventListener('click', (e) => copy($('#share-url').value, e.currentTarget));
  $('#download').addEventListener('click', () => {
    const url = URL.createObjectURL(new Blob([$('#export-text').value], { type: format.mime }));
    const a = el('a', { href: url, download: format.file });
    a.click();
    URL.revokeObjectURL(url);
  });
  $('#import-apply').addEventListener('click', () => importText($('#import-text').value, 'the pasted text'));
  $('#import-file').addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    if (file.size > MAX_BYTES) showImport({ ok: false, errors: [`The file is larger than ${MAX_BYTES / 1000} kB.`] }, file.name);
    else importText(await file.text(), file.name);
    event.target.value = '';
  });
}

// ---- state ------------------------------------------------------------------------------------

function render(animate = false) {
  applyToPanes(animate);
  syncControls();
  renderContrast();
  renderExport();
}

/** A user change: apply it, then keep it (autosave and the address bar). */
function commit(patch, { animate = false } = {}) {
  theme = { ...theme, ...patch };
  render(animate);
  store.save(theme);
  history.replaceState(null, '', encodeHash(theme));
}

function buildToolbar() {
  for (const preset of PRESETS) {
    const button = el('button', { type: 'button', text: preset.name });
    button.addEventListener('click', () => {
      commit(preset.theme, { animate: true });
      say(`Applied the ${preset.name} preset.`);
    });
    $('#presets').append(button);
  }
  $('#reset').addEventListener('click', () => {
    commit(defaults(), { animate: true });
    say('Reset to the default theme.');
  });
  for (const t of TARGETS) $('#target').append(el('option', { value: t.id, text: t.label }));
  $('#target').addEventListener('change', (e) => {
    target = TARGETS.find((t) => t.id === e.target.value);
    renderContrast();
  });
  for (const radio of document.querySelectorAll('input[name="view"]')) {
    radio.addEventListener('change', () => {
      $('#panes').dataset.view = radio.value;
      panes.forEach((p) => (p.hidden = radio.value !== 'both' && p.dataset.scheme !== radio.value));
    });
  }
}

function start() {
  buildControls();
  buildToolbar();
  buildExportControls();
  $('#fallback-banner').hidden = NATIVE;

  const loaded = store.load();
  theme = loaded.theme;
  const shared = decodeHash(location.hash, theme);
  if (shared?.ok) {
    theme = shared.theme;
    say('Loaded the theme from the link.');
  } else if (shared) {
    say(`The link's theme was not used: ${shared.errors[0]}`);
  } else if (loaded.notice) say(loaded.notice);
  render();
}

start();

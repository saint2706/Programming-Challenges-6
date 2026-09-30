import { describe, expect, it, vi } from 'vitest';
import { BOM, cellText, downloadText, guardFormula, quoteField, toCsv } from '../src/csv.js';
import { AWKWARD_NOTES } from '../src/data.js';

const col = (id, type = 'text', header = id) => ({ id, type, header });

/** Minimal RFC 4180 parser, independent of the serializer, for round-trip checks. */
function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') {
      row.push(field);
      field = '';
    } else if (ch === '\r' && text[i + 1] === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
      i++;
    } else field += ch;
  }
  return rows;
}

describe('quoteField (RFC 4180)', () => {
  it.each([
    ['plain', 'plain'],
    ['a,b', '"a,b"'],
    ['say "hi"', '"say ""hi"""'],
    ['line1\nline2', '"line1\nline2"'],
    ['line1\r\nline2', '"line1\r\nline2"'],
    ['cr\ronly', '"cr\ronly"'],
    ['', ''],
    ['  padded  ', '  padded  '],
  ])('quoteField(%j) -> %j', (input, expected) => {
    expect(quoteField(input)).toBe(expected);
  });
});

describe('formula-injection guard', () => {
  it.each(['=1+1', '+1', '-1+2', '@SUM(A1)', '\t=x', '\r=x'])('prefixes %j with a single quote', (t) => {
    expect(guardFormula(t)).toBe(`'${t}`);
  });

  it.each(['plain', "'already", 'a=b', '1+1', ' =leading space', ''])('leaves %j alone', (t) => {
    expect(guardFormula(t)).toBe(t);
  });

  it('guards text cells but never numeric, currency or boolean ones', () => {
    expect(cellText(col('n', 'text'), '=cmd')).toBe("'=cmd");
    expect(cellText(col('d', 'date'), '-2020')).toBe("'-2020");
    expect(cellText(col('n', 'number'), -5)).toBe('-5');
    expect(cellText(col('s', 'currency'), -1234.5)).toBe('-1234.5');
    expect(cellText(col('b', 'boolean'), false)).toBe('false');
  });

  it('can be disabled', () => {
    expect(cellText(col('t'), '=1', { guard: false })).toBe('=1');
  });

  it('guards headers too', () => {
    expect(toCsv([col('x', 'text', '=evil')], [])).toBe("'=evil\r\n");
  });

  it('guards before quoting, so a guarded cell with a comma is quoted once', () => {
    expect(toCsv([col('t')], [{ t: '=1,2' }])).toBe('t\r\n"\'=1,2"\r\n');
  });
});

describe('cellText', () => {
  it('maps null/undefined/NaN/Infinity to an empty field', () => {
    for (const v of [null, undefined, NaN, Infinity, -Infinity]) expect(cellText(col('x', 'number'), v)).toBe('');
  });

  it('serializes raw values, not display formatting', () => {
    expect(cellText(col('s', 'currency'), 1234.5)).toBe('1234.5');
    expect(cellText(col('b', 'boolean'), true)).toBe('true');
    expect(cellText(col('j', 'date'), '2021-03-04')).toBe('2021-03-04');
  });
});

describe('toCsv', () => {
  const columns = [col('id', 'number', 'ID'), col('name', 'text', 'Name'), col('note', 'text', 'Note')];

  it('emits a header, CRLF terminators and a trailing CRLF', () => {
    const csv = toCsv(columns, [
      { id: 1, name: 'A', note: 'x' },
      { id: 2, name: 'B', note: null },
    ]);
    expect(csv).toBe('ID,Name,Note\r\n1,A,x\r\n2,B,\r\n');
  });

  it('is header-only for zero rows', () => {
    expect(toCsv(columns, [])).toBe('ID,Name,Note\r\n');
  });

  it('adds a UTF-8 BOM on request only', () => {
    expect(toCsv(columns, [], { bom: true }).startsWith(BOM)).toBe(true);
    expect(toCsv(columns, []).startsWith(BOM)).toBe(false);
    expect(BOM).toBe('﻿');
  });

  it('round-trips every awkward note through an independent parser', () => {
    const rows = AWKWARD_NOTES.map((note, i) => ({ id: i, name: `n${i}`, note }));
    const parsed = parseCsv(toCsv(columns, rows));
    expect(parsed).toHaveLength(rows.length + 1);
    parsed.slice(1).forEach((fields, i) => {
      expect(fields).toHaveLength(3); // no stray delimiters from commas / quotes / newlines
      expect(fields[2]).toBe(guardFormula(AWKWARD_NOTES[i]));
    });
  });

  it('keeps non-ASCII text intact', () => {
    expect(toCsv([col('t')], [{ t: 'Émigré ☕ 你好' }])).toBe('t\r\nÉmigré ☕ 你好\r\n');
  });

  it('only exports the columns and rows it is given (the filtered + sorted view, in order)', () => {
    expect(toCsv([columns[1]], [{ name: 'Z' }, { name: 'A' }])).toBe('Name\r\nZ\r\nA\r\n');
  });
});

describe('downloadText', () => {
  it('clicks a hidden anchor with the blob URL, then cleans up and revokes', () => {
    vi.useFakeTimers();
    const click = vi.fn();
    const anchor = {
      click,
      remove: vi.fn(),
      set href(v) {
        this._href = v;
      },
      set download(v) {
        this._dl = v;
      },
      set hidden(_v) {},
    };
    const doc = { createElement: vi.fn(() => anchor), body: { appendChild: vi.fn() } };
    const urlApi = { createObjectURL: vi.fn(() => 'blob:fake'), revokeObjectURL: vi.fn() };

    downloadText('a,b\r\n', 'out.csv', { doc, urlApi });

    expect(urlApi.createObjectURL.mock.calls[0][0].type).toBe('text/csv;charset=utf-8');
    expect(anchor._href).toBe('blob:fake');
    expect(anchor._dl).toBe('out.csv');
    expect(click).toHaveBeenCalledOnce();
    expect(anchor.remove).toHaveBeenCalledOnce();
    expect(urlApi.revokeObjectURL).not.toHaveBeenCalled(); // deferred so the download can start
    vi.runAllTimers();
    expect(urlApi.revokeObjectURL).toHaveBeenCalledWith('blob:fake');
    vi.useRealTimers();
  });
});

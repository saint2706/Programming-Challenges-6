import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import {
    MAX_FILES,
    MAX_FILE_BYTES,
    MAX_NAME,
    MAX_PROJECT_BYTES,
    entryOf,
    extensionOf,
    filesProblem,
    freshName,
    isEntry,
    kindOf,
    nameProblem,
    parseProject,
} from '../src/project.ts';
import { TEMPLATES } from '../src/templates.ts';

const page = { name: 'index.html', code: '<p>hi</p>' };

describe('file names', () => {
    test('kinds follow the extension', () => {
        expect(kindOf('a.html')).toBe('html');
        expect(kindOf('A.CSS')).toBe('css');
        expect(['a.js', 'a.mjs', 'a.jsx', 'a.ts', 'a.tsx'].map(kindOf)).toEqual(Array(5).fill('script'));
        expect(kindOf('data.json')).toBe('json');
        expect(kindOf('a.exe')).toBeNull();
        expect(kindOf('noext')).toBeNull();
        expect(extensionOf('a.b.TS')).toBe('ts');
    });

    test.each([
        ['', /name/i],
        ['x'.repeat(MAX_NAME) + '.js', /at most/],
        ['../evil.js', /letters/],
        ['dir/file.js', /letters/],
        ['a b.js', /letters/],
        ['.hidden.js', /letters/],
        ['bad\u0000.js', /letters/],
        ['file.exe', /Files can end/],
        ['file', /Files can end/],
    ])('%j is refused', (name, why) => {
        expect(nameProblem(name)).toMatch(why);
    });

    test.each(['a.constructor', 'a.__proto__', 'a.toString', 'a.hasOwnProperty', 'a.valueOf', 'main.prototype'])('%j is not a file type, even though Object.prototype has a member of that name', (name) => {
        expect(kindOf(name)).toBeNull();
        expect(nameProblem(name)).toMatch(/Files can end/);
    });

    test.each(['main.ts', 'style.css', 'a-b_c.1.js', 'index.html', 'data.json', 'Äb.js'.replace('Ä', 'A')])('%j is allowed', (name) => {
        expect(nameProblem(name)).toBeNull();
    });

    test('only main.* is the entry', () => {
        expect(['main.js', 'main.mjs', 'main.jsx', 'main.ts', 'main.tsx', 'MAIN.TS'].every(isEntry)).toBe(true);
        expect(['app.js', 'main.css', 'main.html', 'xmain.js', 'main.json'].some(isEntry)).toBe(false);
        expect(entryOf({ network: false, files: [page, { name: 'main.ts', code: '' }] })?.name).toBe('main.ts');
        expect(entryOf({ network: false, files: [page] })).toBeUndefined();
    });

    test('a fresh name skips the ones taken', () => {
        expect(freshName([page])).toBe('untitled.js');
        expect(freshName([page, { name: 'untitled.js', code: '' }, { name: 'UNTITLED-2.js', code: '' }])).toBe('untitled-3.js');
    });
});

describe('a set of files', () => {
    test('needs exactly one index.html and at most one main script', () => {
        expect(filesProblem([page])).toBeNull();
        expect(filesProblem([])).toMatch(/exactly one/);
        expect(filesProblem([{ name: 'main.js', code: '' }])).toMatch(/exactly one/);
        expect(filesProblem([page, { name: 'about.html', code: '' }])).toMatch(/index\.html/);
        expect(filesProblem([page, { name: 'main.js', code: '' }, { name: 'main.ts', code: '' }])).toMatch(/Only one main/);
    });

    test('names are unique whatever their case', () => {
        expect(filesProblem([page, { name: 'a.js', code: '' }, { name: 'A.js', code: '' }])).toMatch(/Two files/);
    });

    test('count and size limits', () => {
        const many = Array.from({ length: MAX_FILES }, (_, i) => ({ name: `f${i}.js`, code: '' }));
        expect(filesProblem([page, ...many])).toMatch(/At most/);
        expect(filesProblem([page, { name: 'big.js', code: 'x'.repeat(MAX_FILE_BYTES + 1) }])).toMatch(/over/);
        const chunks = Array.from({ length: 4 }, (_, i) => ({ name: `c${i}.js`, code: 'x'.repeat(MAX_FILE_BYTES) }));
        expect(MAX_FILE_BYTES * 3).toBeGreaterThan(MAX_PROJECT_BYTES);
        expect(filesProblem([page, ...chunks])).toMatch(/project is over/);
    });
});

describe('parseProject trusts nothing', () => {
    const valid = { network: true, files: [page, { name: 'main.js', code: 'x' }] };

    test('accepts a valid project and reads the network flag strictly', () => {
        expect(parseProject(valid)?.network).toBe(true);
        expect(parseProject({ ...valid, network: 'yes' })?.network).toBe(false);
        expect(parseProject({ files: valid.files })?.network).toBe(false);
    });

    test.each([
        ['null', null],
        ['a string', 'x'],
        ['no files', { files: [] }],
        ['files not an array', { files: {} }],
        ['a file that is null', { files: [null] }],
        ['a file with a numeric name', { files: [{ name: 1, code: '' }] }],
        ['a file with code that is not text', { files: [page, { name: 'a.js', code: 5 }] }],
        ['a traversal name', { files: [page, { name: '../x.js', code: '' }] }],
        ['no index.html', { files: [{ name: 'main.js', code: '' }] }],
        ['too many files', { files: Array.from({ length: 50 }, (_, i) => ({ name: `f${i}.js`, code: '' })) }],
    ])('rejects %s', (_label, value) => {
        expect(parseProject(value)).toBeNull();
    });

    test('keeps only known fields, so nothing extra or inherited rides along', () => {
        const parsed = parseProject(JSON.parse('{"files":[{"name":"index.html","code":"x","__proto__":{"polluted":1},"extra":1}],"network":false,"__proto__":{"polluted":1}}'));
        expect(parsed).toEqual({ network: false, files: [{ name: 'index.html', code: 'x' }] });
        expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    });

    test('never throws, whatever it is given', () => {
        fc.assert(fc.property(fc.anything(), (value) => void parseProject(value)));
    });
});

test('every template is a valid project with exactly one entry or none', () => {
    for (const t of TEMPLATES) {
        expect(filesProblem(t.project.files), t.id).toBeNull();
        expect(parseProject(t.project), t.id).not.toBeNull();
        expect(t.project.network).toBe(false); // starters never need the network
    }
});

import * as esbuild from 'esbuild';
import { describe, expect, test } from 'vitest';
import { BUNDLE, Mapper, fileName, parseFrames } from '../src/sourcemap.ts';

describe('parseFrames', () => {
    test('reads Chrome frames, with and without a function name', () => {
        const stack = `Error: boom
    at explode (bundle.js:12:7)
    at bundle.js:30:1
    at https://example.com/a.js:1:2`;
        expect(parseFrames(stack)).toEqual([
            { file: 'bundle.js', line: 12, column: 7 },
            { file: 'bundle.js', line: 30, column: 1 },
            { file: 'https://example.com/a.js', line: 1, column: 2 },
        ]);
    });

    test('reads Firefox frames', () => {
        expect(parseFrames('explode@bundle.js:12:7\n@bundle.js:30:1')).toEqual([
            { file: 'bundle.js', line: 12, column: 7 },
            { file: 'bundle.js', line: 30, column: 1 },
        ]);
    });

    test('skips lines that are not frames and survives junk', () => {
        expect(parseFrames('Error: x\nnot a frame\n\n   at (anonymous)')).toEqual([]);
        expect(parseFrames('')).toEqual([]);
        expect(parseFrames('a:b:c:d\n:::\n(:1:2)')).toBeDefined();
    });
});

test('fileName: the person sees their file name, and built-in packages stay visibly built in', () => {
    expect(fileName('pg:/main.tsx')).toBe('main.tsx');
    expect(fileName('../../pg:/main.tsx')).toBe('main.tsx');
    expect(fileName('/main.tsx')).toBe('main.tsx');
    expect(fileName('main.tsx')).toBe('main.tsx');
    expect(fileName('vendor:preact/hooks')).toBe('vendor:preact/hooks');
    expect(fileName('../../vendor:@preact/signals')).toBe('vendor:@preact/signals');
});

describe('Mapper', () => {
    const source = `const a = 1;

function boom() {
    throw new Error('x');
}
boom();
`;

    async function build() {
        const result = await esbuild.build({
            stdin: { contents: source, sourcefile: 'pg:/main.ts', loader: 'ts' },
            bundle: true,
            write: false,
            outfile: 'bundle.js',
            sourcemap: 'external',
            format: 'esm',
            logLevel: 'silent',
        });
        return { code: result.outputFiles.find((f) => f.path.endsWith('.js'))!.text, map: result.outputFiles.find((f) => f.path.endsWith('.map'))!.text };
    }

    test('maps a position in the bundle back to the line in the original', async () => {
        const { code, map } = await build();
        const at = code.split('\n').findIndex((l) => l.includes('throw new Error('));
        const mapper = new Mapper(map);
        const original = mapper.original(at + 1, code.split('\n')[at]!.indexOf('throw') + 1);
        expect(original).toMatchObject({ file: 'main.ts', line: 4 });
    });

    test('rewrites a stack so the person reads their own file and line', async () => {
        const { code, map } = await build();
        const at = code.split('\n').findIndex((l) => l.includes('throw new Error(')) + 1;
        const stack = `Error: x\n    at boom (${BUNDLE}:${at}:9)\n    at https://other.example/x.js:5:5`;
        const text = new Mapper(map).rewriteStack(stack);
        expect(text).toContain('main.ts:4:');
        expect(text).toContain('https://other.example/x.js:5:5'); // other frames are left alone
        expect(text).not.toContain(BUNDLE);
    });

    test('firstOriginal skips frames that are not in the bundle', async () => {
        const { code, map } = await build();
        const at = code.split('\n').findIndex((l) => l.includes('boom();') && !l.includes('function')) + 1;
        const stack = `Error: x\n    at lib (https://x.example/l.js:1:1)\n    at ${BUNDLE}:${at}:1`;
        expect(new Mapper(map).firstOriginal(stack)).toMatchObject({ file: 'main.ts', line: 6 });
    });

    test('a stack that lies (line 0, huge numbers, negative columns) is unknown, never an exception', async () => {
        const { map } = await build();
        const mapper = new Mapper(map);
        for (const [l, c] of [[0, 0], [0, 5], [-1, 1], [1e9, 1], [1, -5], [NaN, 1], [1.5, 1]] as const) {
            expect(() => mapper.original(l, c), `${l}:${c}`).not.toThrow();
        }
        expect(() => mapper.rewriteStack(`at ${BUNDLE}:0:0
at ${BUNDLE}:99999999:1
at ${BUNDLE}:1:99999`)).not.toThrow();
        expect(() => mapper.firstOriginal(`at ${BUNDLE}:0:0`)).not.toThrow();
    });

    test('with no map, or a broken one, it says "unknown" rather than throwing', () => {
        expect(new Mapper(null).original(1, 1)).toBeNull();
        expect(new Mapper('{not json').original(1, 1)).toBeNull();
        expect(new Mapper(null).rewriteStack(`at ${BUNDLE}:1:1`)).toBe(`at ${BUNDLE}:1:1`);
        expect(new Mapper(null).firstOriginal('x')).toBeNull();
    });
});

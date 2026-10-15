import * as esbuild from 'esbuild';
import fc from 'fast-check';
import vendor from 'virtual:vendor';
import { describe, expect, test } from 'vitest';
import { bundleProject } from '../src/bundle.ts';
import { GUARD, guardLoops } from '../src/loopguard.ts';
import { Mapper } from '../src/sourcemap.ts';

/** Guard a function body (so it may `return`), run it with the guard counting its calls, and say what it did. */
function guardedRun(body: string, plain = false): { result: unknown; guards: number; loops: number; code: string } {
    const source = `function __body() { ${body} }`;
    const guarded = plain ? { code: source, loops: 0 } : guardLoops(source, null);
    let guards = 0;
    const result = new Function('__pgLoop', `${guarded.code}; return __body();`)(() => void guards++);
    return { result, guards, loops: guarded.loops, code: guarded.code };
}

describe('every kind of loop gets a guard', () => {
    test.each([
        ['for', 'let s = 0; for (let i = 0; i < 5; i++) { s += i; } return s;', 10, 5],
        ['while', 'let i = 0, s = 0; while (i < 5) { s += i++; } return s;', 10, 5],
        ['do-while', 'let i = 0, s = 0; do { s += i++; } while (i < 5); return s;', 10, 5],
        ['for-in', 'let s = ""; for (const k in { a: 1, b: 2, c: 3 }) { s += k; } return s;', 'abc', 3],
        ['for-of', 'let s = 0; for (const x of [1, 2, 3]) { s += x; } return s;', 6, 3],
        ['no braces: for', 'let s = 0; for (let i = 0; i < 5; i++) s += i; return s;', 10, 5],
        ['no braces: while', 'let i = 0; while (i < 5) i++; return i;', 5, 5],
        ['no braces: do-while', 'let i = 0; do i++; while (i < 5); return i;', 5, 5],
        ['empty body', 'let i = 0; for (; i < 5; i++); return i;', 5, 5],
        ['nested, no braces', 'let n = 0; for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) n++; return n;', 12, 15],
        ['labelled break and continue', 'let n = 0; outer: for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) { if (j === 2) continue outer; if (i === 3) break outer; n++; } } return n;', 6, 14],
        ['inside a function, a closure, a class', 'class A { f() { let n = 0; for (const x of [1,2]) { n += x; } return n; } } return new A().f() + ((() => { let n = 0; while (n < 3) n++; return n; })());', 6, 5],
        ['an if without braces around a loop', 'let n = 0; if (true) for (let i = 0; i < 3; i++) n++; else n = 99; return n;', 3, 3],
    ])('%s', (_label, body, expected, guards) => {
        const run = guardedRun(body);
        expect(run.result).toEqual(expected);
        expect(run.guards).toBe(guards);
        expect(run.code.split(GUARD).length - 1).toBe(run.loops);
    });

    test('async code and generators keep working', async () => {
        const body = `
            async function f() { let n = 0; for await (const x of (async function* () { yield 1; yield 2; })()) { n += x; } return n; }
            function* g() { let i = 0; while (i < 3) yield i++; }
            return f().then((n) => [n, [...g()]]);`;
        const run = guardedRun(body);
        expect(await (run.result as Promise<unknown>)).toEqual([3, [0, 1, 2]]);
        expect(run.loops).toBe(2);
    });

    test('code with no loops comes back untouched', () => {
        const code = 'const a = 1;\nconsole.log(a);\n';
        expect(guardLoops(code, null)).toEqual({ code, map: null, loops: 0 });
    });

    test('code that is not valid JavaScript throws, so the caller can fall back', () => {
        expect(() => guardLoops('for (;;', null)).toThrow();
    });
});

describe('the guard never changes what a program computes', () => {
    // Programs built from loops over small ranges, with break/continue, nesting and mixed brace styles.
    const header = fc.constantFrom('for (let V = 0; V < 4; V++)', 'for (const V of [1, 2, 3, 4])', 'for (const V in { a: 1, b: 2, c: 3 })', 'for (let V = 0; V < 4; V += 2)', 'while (t < 12 && ++t)');
    const simple = fc.constantFrom('t += 1;', 't += 2;', 'if (t % 3 === 0) continue;', 'if (t > 40) break;');
    const statement = (depth: number): fc.Arbitrary<string> =>
        depth === 0
            ? simple
            : fc.oneof(
                  { weight: 3, arbitrary: simple },
                  {
                      weight: 2,
                      arbitrary: fc.tuple(header, fc.array(statement(depth - 1), { minLength: 1, maxLength: 3 }), fc.boolean()).map(([head, body, braces]) => {
                          const loop = head.replace(/V/g, `v${depth}`);
                          return braces || body.length > 1 ? `${loop} { ${body.join(' ')} }` : `${loop} ${body[0]}`;
                      }),
                  },
              );

    test('same result with and without guards, for random loop programs', () => {
        fc.assert(
            fc.property(fc.array(statement(2), { minLength: 1, maxLength: 4 }), (parts) => {
                // `continue` and `break` need an enclosing loop, so everything sits inside one.
                const body = `let t = 0; for (let top = 0; top < 3; top++) { ${parts.join(' ')} } return t;`;
                expect(guardedRun(body).result).toBe(guardedRun(body, true).result);
            }),
            { numRuns: 300 },
        );
    });
});

describe('only the person\'s own loops are guarded, and the source map stays exact', () => {
    async function build(files: Record<string, string>) {
        const r = await bundleProject({ network: false, files: [{ name: 'index.html', code: '' }, ...Object.entries(files).map(([name, code]) => ({ name, code }))] }, esbuild, vendor);
        if (!r.ok) throw new Error(JSON.stringify(r.errors));
        return r;
    }

    test('a loop in the person\'s file is guarded; the loops inside bundled Preact are not', async () => {
        const mine = await build({ 'main.tsx': `import { render } from 'preact'; for (let i = 0; i < 3; i++) { console.log(i); } render(<p>x</p>, document.body);` });
        expect(mine.loops).toBe(1);
        const none = await build({ 'main.tsx': `import { render } from 'preact'; render(<p>x</p>, document.body);` });
        expect(none.loops).toBe(0); // Preact has dozens of loops, and none are touched
    });

    test('loops in an imported file of the person\'s are guarded too', async () => {
        const r = await build({ 'main.ts': `import { sum } from './sum'; console.log(sum(3));`, 'sum.ts': `export function sum(n: number) { let s = 0; while (n > 0) s += n--; return s; }` });
        expect(r.loops).toBe(1);
    });

    test('after guards are added, a position still maps to the right place in the original', async () => {
        const source = `const before = 1;\nfor (let i = 0; i < 3; i++) total += i; const AFTER_MARK = 2;\nconsole.log(before, AFTER_MARK);\n`;
        const r = await build({ 'main.ts': `let total = 0;\n${source}` });
        expect(r.loops).toBe(1);
        const lines = r.code.split('\n');
        const mapper = new Mapper(r.map);
        // Every identifier we care about maps back to where it was typed, including ones after the inserted braces and guard.
        for (const token of ['AFTER_MARK', 'before']) {
            const lineIndex = lines.findIndex((l) => l.includes(token) && l.includes('console.log') === token.includes('before'));
            const column = lines[lineIndex]!.indexOf(token);
            const original = mapper.original(lineIndex + 1, column + 1)!;
            const typed = (`let total = 0;\n${source}`).split('\n')[original.line - 1]!;
            expect(typed.slice(original.column - 1, original.column - 1 + token.length), token).toBe(token);
        }
    });
});

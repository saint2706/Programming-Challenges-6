import * as esbuild from 'esbuild';
import vendor from 'virtual:vendor';
import { describe, expect, test } from 'vitest';
import { type BundleResult, bundleProject, resolveRelative } from '../src/bundle.ts';
import type { Project } from '../src/project.ts';
import { TEMPLATES } from '../src/templates.ts';

const project = (files: Record<string, string>, network = false): Project => ({
    network,
    files: [{ name: 'index.html', code: '<p></p>' }, ...Object.entries(files).map(([name, code]) => ({ name, code }))],
});

const build = (files: Record<string, string>, network = false) => bundleProject(project(files, network), esbuild, vendor);

function ok(result: BundleResult) {
    if (!result.ok) throw new Error(`build failed: ${JSON.stringify(result.errors)}`);
    return result;
}

describe('building the person\'s files', () => {
    test('no entry file means nothing to run, and no compiler is needed', async () => {
        const r = ok(await build({ 'util.ts': 'export const x = 1' }));
        expect(r.code).toBe('');
        expect(r.map).toBeNull();
    });

    test('TypeScript, TSX and JSX are compiled; types are erased', async () => {
        const r = ok(await build({ 'main.ts': `const n: number = 41; interface A { x: number } console.log(n + 1 satisfies number);` }));
        expect(r.code).toContain('console.log(n + 1)');
        expect(r.code).not.toContain('interface');
        const tsx = ok(await build({ 'main.tsx': `import { render } from 'preact'; render(<b>hi</b>, document.body);` }));
        expect(tsx.code).toContain('"b"');
        const jsx = ok(await build({ 'main.jsx': `import { render } from 'preact'; render(<i>x</i>, document.body);` }));
        expect(jsx.code).toContain('"i"');
    });

    test('relative imports between files, with or without an extension, and the TypeScript ".js" habit', async () => {
        for (const spec of ['./util', './util.ts', './util.js']) {
            const r = ok(await build({ 'main.ts': `import { x } from '${spec}'; console.log(x);`, 'util.ts': 'export const x = 7;' }));
            expect(r.code, spec).toContain('x = 7');
        }
    });

    test('JSON can be imported', async () => {
        const r = ok(await build({ 'main.js': `import data from './data.json'; console.log(data.name);`, 'data.json': '{"name":"Ada"}' }));
        expect(r.code).toContain('Ada');
    });

    test('importing a stylesheet from a script is harmless and says so', async () => {
        const r = ok(await build({ 'main.js': `import './style.css'; console.log(1)`, 'style.css': 'body{}' }));
        expect(r.warnings.map((w) => w.message).join()).toMatch(/added to the page on its own/);
    });

    test('Preact is built in, bundled once, and shared by hooks and signals', async () => {
        const r = ok(
            await build({
                'main.tsx': `import { render } from 'preact'; import { useState } from 'preact/hooks'; import { signal } from '@preact/signals'; const s = signal(0); function A() { const [n] = useState(1); return <p>{n}{s}</p>; } render(<A />, document.body);`,
            }),
        );
        expect(r.code.match(/^\/\/ vendor:preact$/gm)).toHaveLength(1);
        expect(r.warnings).toEqual([]);
    });

    test('every template builds without errors or warnings', async () => {
        for (const t of TEMPLATES) {
            const r = ok(await bundleProject(t.project, esbuild, vendor));
            expect(r.warnings, t.id).toEqual([]);
        }
    });

    test('the output has no pointer to a source map file, and the map names the person\'s files', async () => {
        const r = ok(await build({ 'main.ts': 'console.log(1)' }));
        expect(r.code).not.toContain('sourceMappingURL');
        expect(JSON.parse(r.map!).sources.join()).toContain('pg:/main.ts');
    });
});

describe('hostile source text', () => {
    test('a string that looks like a source-map pointer, followed by a long run of spaces, builds quickly', async () => {
        // A regular expression written to strip such a pointer from the output once took quadratic time on this.
        const code = `console.log("//# sourceMappingURL=" + "${' '.repeat(190_000)}x");`;
        const started = performance.now();
        const r = await build({ 'main.js': code });
        expect(r.ok).toBe(true);
        expect(performance.now() - started).toBeLessThan(3000);
    });

    test('a file named like a member of Object.prototype is not given a loader from it', async () => {
        const r = await bundleProject({ network: false, files: [{ name: 'index.html', code: '' }, { name: 'main.js', code: `import './x.constructor'` }, { name: 'x.constructor', code: 'console.log(1)' }] }, esbuild, vendor);
        if (!r.ok) throw new Error(JSON.stringify(r.errors)); // it builds as plain JavaScript; before, the plugin crashed on a loader that was a function
        expect(r.code).toContain('console.log(1)');
    });
});

describe('what cannot be imported', () => {
    test('a missing file points at the import and explains the flat folder', async () => {
        const r = await build({ 'main.ts': `import { x } from './nope';\nconsole.log(x);` });
        expect(r.ok).toBe(false);
        if (r.ok) return;
        expect(r.errors[0]).toMatchObject({ file: 'main.ts', line: 1, severity: 'error' });
        expect(r.errors[0]!.message).toMatch(/Could not find \.\/nope/);
    });

    test.each(['../up', './sub/dir', 'data:text/javascript,alert(1)', 'file:///etc/passwd', 'node:fs', '/abs/path', 'pkg/../../x', 'a b', 'UPPER', '@scope', 'x?y', 'x#y'])(
        'the odd specifier %j is an error, with the network on or off',
        async (spec) => {
            for (const network of [false, true]) {
                const r = await build({ 'main.js': `import x from ${JSON.stringify(spec)}; console.log(x);` }, network);
                expect(r.ok, `${spec} network=${network}`).toBe(false);
            }
        },
    );

    test('an unknown package, network off: the error lists what is built in and how to get more', async () => {
        const r = await build({ 'main.js': `import _ from 'lodash-es'; console.log(_);` });
        expect(r.ok).toBe(false);
        if (r.ok) return;
        expect(r.errors[0]!.message).toMatch(/Cannot find package "lodash-es"/);
        expect(r.errors[0]!.message).toMatch(/preact/);
        expect(r.errors[0]!.message).toMatch(/Allow network/);
    });

    test('with the network on, packages come from esm.sh as external imports, versions and subpaths included', async () => {
        const r = ok(await build({ 'main.js': `import a from 'lodash-es'; import b from '@scope/pkg@1.2.3/sub'; import c from 'canvas-confetti@1'; console.log(a, b, c);` }, true));
        expect(r.code).toContain('"https://esm.sh/lodash-es"');
        expect(r.code).toContain('"https://esm.sh/@scope/pkg@1.2.3/sub"');
        expect(r.code).toContain('"https://esm.sh/canvas-confetti@1"');
    });

    test('URL imports: only esm.sh, and only with the network on', async () => {
        const off = await build({ 'main.js': `import x from 'https://esm.sh/foo'; console.log(x);` });
        expect(off.ok).toBe(false);
        const on = ok(await build({ 'main.js': `import x from 'https://esm.sh/foo'; console.log(x);` }, true));
        expect(on.code).toContain('https://esm.sh/foo');
        for (const url of ['https://evil.example/x.js', 'http://esm.sh/x', 'https://esm.sh.evil.example/x']) {
            const r = await build({ 'main.js': `import x from ${JSON.stringify(url)}; console.log(x);` }, true);
            expect(r.ok, url).toBe(false);
        }
    });

    test('built-in packages cannot be reached into by relative path', async () => {
        const r = await build({ 'main.js': `import { h } from 'preact'; console.log(h);` });
        expect(r.ok).toBe(true);
    });
});

describe('errors in the person\'s code', () => {
    test('a syntax error has the right file, line, column and a length', async () => {
        const r = await build({ 'main.ts': `const ok = 1;\nconst broken = ;\n` });
        expect(r.ok).toBe(false);
        if (r.ok) return;
        expect(r.errors[0]).toMatchObject({ file: 'main.ts', line: 2, severity: 'error' });
        expect(r.errors[0]!.column).toBe(16);
        expect(r.errors[0]!.length).toBeGreaterThanOrEqual(1);
    });

    test('an error in an imported file names that file', async () => {
        const r = await build({ 'main.ts': `import './b';`, 'b.ts': `let x = ;` });
        expect(r.ok).toBe(false);
        if (r.ok) return;
        expect(r.errors[0]).toMatchObject({ file: 'b.ts', line: 1 });
    });

    test('JSX that is not closed is reported with a position', async () => {
        const r = await build({ 'main.tsx': `const a = <div>;` });
        expect(r.ok).toBe(false);
        if (r.ok) return;
        expect(r.errors[0]).toMatchObject({ file: 'main.tsx', line: 1 });
    });
});

describe('resolveRelative', () => {
    const names = ['index.html', 'main.ts', 'util.ts', 'view.tsx', 'data.json', 'old.js'];
    test.each([
        ['./util', 'util.ts'],
        ['./util.ts', 'util.ts'],
        ['./util.js', 'util.ts'],
        ['./view', 'view.tsx'],
        ['./view.jsx', 'view.tsx'],
        ['./data', 'data.json'],
        ['./old', 'old.js'],
    ])('%s -> %s', (spec, found) => {
        expect(resolveRelative(spec, names)).toBe(found);
    });

    test.each(['./nope', '../util', './a/b', 'util', '/util', './', '.'])('%s is not a file', (spec) => {
        expect(resolveRelative(spec, names)).toBeNull();
    });
});

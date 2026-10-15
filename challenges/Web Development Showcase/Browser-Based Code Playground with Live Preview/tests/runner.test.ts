import { describe, expect, test, vi } from 'vitest';
import type { BundleResult } from '../src/bundle.ts';
import type { RunPayload } from '../src/preview.ts';
import type { Project } from '../src/project.ts';
import { Runner } from '../src/runner.ts';

const project = (over: { html?: string; css?: string; js?: string; network?: boolean } = {}): Project => ({
    network: over.network ?? false,
    files: [
        { name: 'index.html', code: over.html ?? '<p>x</p>' },
        { name: 'style.css', code: over.css ?? 'p{}' },
        { name: 'main.js', code: over.js ?? 'a' },
    ],
});

const good = (code = 'built'): BundleResult => ({ ok: true, code, map: null, warnings: [], loops: 0 });

function setup(build: (p: Project) => Promise<BundleResult> = async () => good()) {
    const runs: RunPayload[] = [];
    const port = {
        live: true,
        runs,
        cssUpdates: [] as string[],
        run(payload: RunPayload) {
            runs.push(payload);
            return runs.length;
        },
        updateCss(css: string) {
            port.cssUpdates.push(css);
            return port.live;
        },
    };
    const runner = new Runner(port, build, () => 0);
    return { runner, port };
}

describe('what an edit needs', () => {
    test('the first update builds and starts a page', async () => {
        const { runner, port } = setup();
        const outcome = await runner.update(project());
        expect(outcome).toMatchObject({ kind: 'full', run: 1 });
        expect(port.runs[0]).toEqual({ html: '<p>x</p>', css: 'p{}', code: 'built', network: false });
    });

    test('no change is no work', async () => {
        const { runner, port } = setup();
        await runner.update(project());
        expect(await runner.update(project())).toEqual({ kind: 'none' });
        expect(port.runs).toHaveLength(1);
    });

    test('a stylesheet-only change is swapped in live, without rebuilding', async () => {
        const build = vi.fn(async () => good());
        const { runner, port } = setup(build);
        await runner.update(project());
        expect(await runner.update(project({ css: 'p{color:red}' }))).toEqual({ kind: 'css' });
        expect(port.cssUpdates).toEqual(['p{color:red}']);
        expect(port.runs).toHaveLength(1);
        expect(build).toHaveBeenCalledTimes(1);
    });

    test('several stylesheets are joined in order', async () => {
        const { runner, port } = setup();
        await runner.update({ network: false, files: [{ name: 'index.html', code: '' }, { name: 'a.css', code: 'A' }, { name: 'b.css', code: 'B' }] });
        expect(port.runs[0]!.css).toBe('A\nB');
    });

    test.each([
        ['the page', { html: '<p>changed</p>' }],
        ['the script', { js: 'b' }],
        ['the network setting', { network: true }],
    ])('a change to %s starts a fresh page', async (_label, change) => {
        const { runner, port } = setup();
        await runner.update(project());
        expect(await runner.update(project(change))).toMatchObject({ kind: 'full' });
        expect(port.runs).toHaveLength(2);
    });

    test('if the stylesheet cannot be swapped in (no live page), it falls back to a full run', async () => {
        const { runner, port } = setup();
        await runner.update(project());
        port.live = false;
        expect(await runner.update(project({ css: 'p{color:blue}' }))).toMatchObject({ kind: 'full' });
    });

    test('force starts a fresh page even when nothing changed', async () => {
        const { runner, port } = setup();
        await runner.update(project());
        expect(await runner.update(project(), { force: true })).toMatchObject({ kind: 'full' });
        expect(port.runs).toHaveLength(2);
    });

    test('invalidate forgets what is on screen', async () => {
        const { runner, port } = setup();
        await runner.update(project());
        runner.invalidate();
        expect(await runner.update(project())).toMatchObject({ kind: 'full' });
        expect(port.runs).toHaveLength(2);
    });

    test('reordering the script files is not a change', async () => {
        const { runner } = setup();
        const a = project();
        await runner.update({ ...a, files: [...a.files, { name: 'util.js', code: 'u' }] });
        const reordered = { ...a, files: [a.files[0]!, { name: 'util.js', code: 'u' }, a.files[1]!, a.files[2]!] };
        expect(await runner.update(reordered)).toEqual({ kind: 'none' });
    });
});

describe('when things go wrong', () => {
    test('a failed build changes nothing: the page that works stays, and the next good edit starts fresh', async () => {
        let fail = false;
        const { runner, port } = setup(async () => (fail ? { ok: false, errors: [{ file: 'main.js', line: 1, column: 1, length: 1, message: 'bad', severity: 'error' }] } : good()));
        await runner.update(project());
        fail = true;
        const outcome = await runner.update(project({ js: 'broken' }));
        expect(outcome).toMatchObject({ kind: 'failed', errors: [{ message: 'bad' }] });
        expect(port.runs).toHaveLength(1);
        fail = false;
        expect(await runner.update(project({ js: 'fixed' }))).toMatchObject({ kind: 'full' });
        expect(port.runs).toHaveLength(2);
    });

    test('an edit that fails then returns to the earlier text is still seen as a change from what is running', async () => {
        let fail = false;
        const { runner, port } = setup(async () => (fail ? { ok: false, errors: [] } : good()));
        await runner.update(project());
        fail = true;
        await runner.update(project({ js: 'broken' }));
        fail = false;
        expect(await runner.update(project())).toEqual({ kind: 'none' }); // back to what is already running
        expect(port.runs).toHaveLength(1);
    });

    test('a slow build that has been overtaken by a newer edit is discarded', async () => {
        const gates: (() => void)[] = [];
        const { runner, port } = setup(
            (p) =>
                new Promise((resolve) => {
                    gates.push(() => resolve(good(p.files[2]!.code)));
                }),
        );
        const first = runner.update(project({ js: 'one' }));
        const second = runner.update(project({ js: 'two' }));
        gates[1]!(); // the newer one finishes first
        gates[0]!();
        expect(await second).toMatchObject({ kind: 'full' });
        expect(await first).toEqual({ kind: 'stale' });
        expect(port.runs.map((r) => r.code)).toEqual(['two']);
    });

    test('an error thrown while building is passed up for the caller to report', async () => {
        const { runner } = setup(async () => {
            throw new Error('compiler gone');
        });
        await expect(runner.update(project())).rejects.toThrow('compiler gone');
    });
});

describe('source maps by run', () => {
    test('each run keeps its own, and only the last few are kept', async () => {
        const { runner } = setup();
        for (let i = 0; i < 8; i++) await runner.update(project({ js: String(i) }));
        expect(runner.mapperFor(8)).toBeDefined();
        expect(runner.mapperFor(7)).toBeDefined();
        expect(runner.mapperFor(1)).toBeUndefined();
        expect(runner.mapperFor(999)).toBeUndefined();
    });
});

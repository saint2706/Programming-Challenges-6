/**
 * Numbers for the README: `pnpm run bench`. Real Chrome against the production build: how long the
 * compiler takes to arrive, how fast an edit becomes pixels, and what the loop guard costs.
 */
import { gzipSync } from 'node:zlib';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findChrome, launchChrome } from '../e2e/chrome.ts';
import { openPlayground, previewFrame, runNumber, setCode, startServer, waitForNewRun } from '../e2e/support.ts';
import { GUARD_RUNTIME } from '../src/guardruntime.ts';

const fmt = (n: number, d = 0) => n.toFixed(d);
const pct = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor((p / 100) * xs.length))]!;

if (!findChrome()) throw new Error('No Chrome found. Set CHROME_PATH.');

console.log('## What gets downloaded\n');
const assets = join(fileURLToPath(new URL('..', import.meta.url)), 'dist', 'assets');
console.log('| file | size | gzipped | loaded |\n| --- | ---: | ---: | --- |');
const when = (name: string) => (name.endsWith('.wasm') ? 'on the first run that has a script' : name.startsWith('bundle-') || name.startsWith('browser-') || name.startsWith('_virtual_vendor') ? 'on the first run that has a script' : name.endsWith('.css') ? 'at start' : 'at start');
for (const name of readdirSync(assets).sort()) {
    const path = join(assets, name);
    if (!/\.(js|css|wasm)$/.test(name) || statSync(path).size < 1000) continue;
    const bytes = readFileSync(path);
    console.log(`| ${name.replace(/-[A-Za-z0-9_-]{8}\./, '.')} | ${fmt(bytes.length / 1000)} kB | ${fmt(gzipSync(bytes).length / 1000)} kB | ${when(name)} |`);
}

const server = await startServer();
const browser = await launchChrome();
try {
    console.log('\n## From opening the page to a running preview\n');
    const cold: number[] = [];
    for (let i = 0; i < 5; i++) {
        const started = performance.now();
        const page = await openPlayground(browser, server.url);
        cold.push(performance.now() - started);
        await page.close();
    }
    console.log(`Cold start, including compiling the 14 MB WebAssembly compiler and building the first template: median ${fmt(pct(cold, 50))} ms (min ${fmt(Math.min(...cold))}, max ${fmt(Math.max(...cold))}, 5 runs).`);

    const page = await openPlayground(browser, server.url);

    console.log('\n## An edit, from the keystroke to the new page being live\n');
    const measure = async (name: string, makeCode: (i: number) => string, expectNew: boolean) => {
        const samples: number[] = [];
        for (let i = 0; i < 30; i++) {
            const before = await runNumber(page);
            const started = performance.now();
            await setCode(page, name, makeCode(i));
            if (expectNew) await waitForNewRun(page, before);
            else {
                const frame = await previewFrame(page);
                await frame.waitForFunction((c: string) => getComputedStyle(document.body).backgroundColor === c, { polling: 'raf' }, `rgb(${i + 10}, 20, 30)`);
            }
            samples.push(performance.now() - started);
        }
        return samples.slice(5); // the first few warm things up
    };
    const js = await measure('main.js', (i) => `console.log('run ${i}');`, true);
    const css = await measure('style.css', (i) => `body { background: rgb(${i + 10}, 20, 30); }`, false);
    console.log('| edit | median | p95 | includes |\n| --- | ---: | ---: | --- |');
    console.log(`| script (rebuild, new page) | ${fmt(pct(js, 50))} ms | ${fmt(pct(js, 95))} ms | the 300 ms typing pause, esbuild, loop guarding, a fresh sandbox |`);
    console.log(`| stylesheet (swapped in live) | ${fmt(pct(css, 50))} ms | ${fmt(pct(css, 95))} ms | the 300 ms typing pause and a style swap |`);

    console.log('\n## What the loop guard costs per iteration\n');
    const result = await page.evaluate(
        (runtime: string) => {
            const N = 20_000_000; // small enough that a guarded run finishes inside the two-second allowance
            const bare = () => {
                let s = 0;
                for (let i = 0; i < N; i++) {
                    s += i % 7;
                }
                return s;
            };
            const guarded = () => {
                const guard = new Function(`${runtime}; return __pgLoop;`)() as () => void; // a fresh allowance each time
                let s = 0;
                for (let i = 0; i < N; i++) {
                    guard();
                    s += i % 7;
                }
                return s;
            };
            const best = (f: () => number) =>
                Math.min(
                    ...Array.from({ length: 7 }, () => {
                        const t0 = performance.now();
                        f();
                        return performance.now() - t0;
                    }),
                );
            return { bare: best(bare), guarded: best(guarded), N };
        },
        GUARD_RUNTIME,
    );
    console.log(`${result.N.toLocaleString('en-US')} iterations of \`s += i % 7\`: ${fmt(result.bare)} ms bare, ${fmt(result.guarded)} ms guarded, which is ${fmt(((result.guarded - result.bare) / result.N) * 1e6, 2)} ns of guard per iteration (${fmt(result.guarded / result.bare, 2)}x on a loop that does almost nothing).`);
    await page.close();
} finally {
    await browser.close();
    await server.close();
}

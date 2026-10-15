// Measures what each scroll engine costs in this machine's headless Chrome: bundle sizes, time until
// the engine is live, and main-thread work, long tasks and dropped frames during a scripted scroll.
// Run with `pnpm run bench` (builds dist/ first). Numbers are medians over interleaved runs.
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';
import { launchChrome } from '../e2e/chrome.mjs';
import { startServer } from '../e2e/server.mjs';

const ASSETS = fileURLToPath(new URL('../dist/assets/', import.meta.url));
const REPEATS = 5; // measured runs per engine, interleaved (css, gsap, css, gsap, ...)
const SWEEP_STEPS = 40; // scroll positions across the chart's range
const SETTLE_MS = 120; // wait after each scroll position
const FRAME_BUDGET_MS = 33; // an interval longer than this (two 60 Hz frames) counts as dropped
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const median = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// --- bundle sizes -------------------------------------------------------------------------------
const pick = (re) => readdirSync(ASSETS).find((f) => re.test(f));
const files = { main: pick(/^index-.*\.js$/), css: pick(/^index-.*\.css$/), gsap: pick(/^gsap-engine-.*\.js$/) };
const size = (name) => {
    const raw = readFileSync(ASSETS + name);
    return { raw: raw.length, gzip: gzipSync(raw, { level: 9 }).length };
};
const kb = (bytes) => (bytes / 1000).toFixed(2);
const sizes = {
    main: size(files.main),
    css: size(files.css),
    gsap: size(files.gsap),
};

// --- in-page measurement ------------------------------------------------------------------------
async function runOnce(browser, server, engine) {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'no-preference' }]);
    await page.evaluateOnNewDocument((engineName) => {
        try {
            localStorage.setItem('scroll-story-engine', engineName);
        } catch {
            /* storage blocked */
        }
        // Record the first moment the engine is live: the flag is set, and for GSAP its triggers exist.
        const ready = () => {
            const flagged = document.documentElement.dataset.engine === engineName;
            const live = engineName === 'css' || (globalThis.__storyDebug?.triggerCount() ?? 0) > 0;
            if (flagged && live) globalThis.__readyAt = performance.now();
            else requestAnimationFrame(ready);
        };
        requestAnimationFrame(ready);
    }, engine);
    await page.goto(server.url, { waitUntil: 'load' });
    await page.waitForFunction(() => globalThis.__readyAt !== undefined, { timeout: 20_000, polling: 'raf' });
    const readyMs = await page.evaluate(() => globalThis.__readyAt);

    // Frame intervals and long tasks are recorded only during the scripted scroll.
    await page.evaluate(() => {
        globalThis.__frames = [];
        globalThis.__long = [];
        globalThis.__last = null;
        globalThis.__recording = false;
        new PerformanceObserver((list) => {
            for (const e of list.getEntries()) globalThis.__long.push(e.duration);
        }).observe({ type: 'longtask' });
        const tick = (t) => {
            if (globalThis.__recording && globalThis.__last !== null) globalThis.__frames.push(t - globalThis.__last);
            globalThis.__last = t;
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    });
    await sleep(300);

    const span = await page.evaluate(() => {
        const story = document.querySelector('#chart-story');
        const top = story.getBoundingClientRect().top + window.scrollY;
        return { top, length: story.offsetHeight - window.innerHeight, vh: window.innerHeight };
    });
    const before = await page.metrics();
    await page.evaluate(() => {
        globalThis.__frames = [];
        globalThis.__long = [];
        globalThis.__recording = true;
    });
    for (let i = 0; i <= SWEEP_STEPS; i++) {
        await page.evaluate((y) => window.scrollTo({ top: y, behavior: 'instant' }), span.top + (span.length * i) / SWEEP_STEPS);
        await sleep(SETTLE_MS);
    }
    await sleep(600); // let GSAP's scrub finish catching up before the final sample
    const after = await page.metrics();
    const frames = await page.evaluate(() => {
        globalThis.__recording = false;
        return { intervals: globalThis.__frames, long: globalThis.__long };
    });

    await context.close();
    const delta = (k) => (after[k] - before[k]) * 1000;
    return {
        readyMs,
        scriptMs: delta('ScriptDuration'),
        taskMs: delta('TaskDuration'),
        layoutMs: delta('LayoutDuration'),
        recalcMs: delta('RecalcStyleDuration'),
        longTasks: frames.long.length,
        longTaskMs: frames.long.reduce((a, b) => a + b, 0),
        frames: frames.intervals.length,
        dropped: frames.intervals.filter((d) => d > FRAME_BUDGET_MS).length,
        worstFrameMs: Math.max(0, ...frames.intervals),
    };
}

const browser = await launchChrome();
const server = await startServer();
const results = { css: [], gsap: [] };
try {
    // One discarded warm-up per engine, so the first run's cold caches do not count.
    await runOnce(browser, server, 'css');
    await runOnce(browser, server, 'gsap');
    for (let r = 0; r < REPEATS; r++) {
        for (const engine of ['css', 'gsap']) results[engine].push(await runOnce(browser, server, engine));
        process.stdout.write(`run ${r + 1}/${REPEATS} done\n`);
    }
} finally {
    await browser.close();
    await server.close();
}

const med = (engine, key) => median(results[engine].map((r) => r[key]));
const row = (label, key, digits = 1) =>
    `${label.padEnd(34)} ${med('css', key).toFixed(digits).padStart(10)} ${med('gsap', key).toFixed(digits).padStart(10)}`;

console.log('\nBundle sizes (bytes, gzip level 9)');
console.log(`  main JS             ${sizes.main.raw} raw, ${sizes.main.gzip} gzip  (${files.main})`);
console.log(`  CSS                 ${sizes.css.raw} raw, ${sizes.css.gzip} gzip  (${files.css})`);
console.log(`  GSAP engine chunk   ${sizes.gsap.raw} raw, ${sizes.gsap.gzip} gzip  (${files.gsap})`);
console.log(`  CSS engine ships    ${sizes.main.gzip + sizes.css.gzip} gzip; GSAP engine ships ${sizes.main.gzip + sizes.css.gzip + sizes.gsap.gzip} gzip`);

console.log(`\nIn-page medians over ${REPEATS} interleaved runs (ms unless stated)`);
console.log(`${'metric'.padEnd(34)} ${'css'.padStart(10)} ${'gsap'.padStart(10)}`);
console.log(row('time to engine live', 'readyMs'));
console.log(row('script time, scroll sweep', 'scriptMs'));
console.log(row('task time, scroll sweep', 'taskMs'));
console.log(row('layout time, scroll sweep', 'layoutMs'));
console.log(row('style recalc time, sweep', 'recalcMs'));
console.log(row('long tasks (>50 ms), count', 'longTasks', 0));
console.log(row('long tasks, total duration', 'longTaskMs'));
console.log(row('frame intervals sampled, count', 'frames', 0));
console.log(row('dropped frames (>33 ms), count', 'dropped', 0));
console.log(row('worst frame interval', 'worstFrameMs'));

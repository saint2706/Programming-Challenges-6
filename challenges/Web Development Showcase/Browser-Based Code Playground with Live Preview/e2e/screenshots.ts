/** `pnpm run screenshots`: rebuilds the README images (docs/*.png) from scripted sessions in real Chrome. */
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { launchChrome } from './chrome.ts';
import { openPlayground, previewFrame, runNumber, setCode, startServer, waitForConsole, waitForNewRun } from './support.ts';

const out = fileURLToPath(new URL('../docs/', import.meta.url));
mkdirSync(out, { recursive: true });

const server = await startServer();
const browser = await launchChrome();
const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));

try {
    // 1. TSX with Preact and signals, compiled in the browser (dark).
    const dark = await openPlayground(browser, server.url);
    await dark.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
    let before = await runNumber(dark);
    await dark.select('#template', 'preact');
    await waitForNewRun(dark, before);
    const frame = await previewFrame(dark);
    for (let i = 0; i < 3; i++) await frame.click('button');
    await dark.click('[role=tab][data-name="main.tsx"]');
    await pause(700);
    await dark.screenshot({ path: `${out}playground-tsx-dark.png` });

    // 2. An infinite loop, stopped and explained (light).
    const light = await openPlayground(browser, server.url);
    await light.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    before = await runNumber(light);
    await setCode(light, 'main.js', `console.log('start');\nlet n = 0;\nwhile (true) {\n    n++;\n}\nconsole.log('never printed', n);\n`);
    await waitForConsole(light, (e) => e.some((x) => /Stopped a loop/.test(x.text)), 20_000);
    await light.waitForSelector('.cm-lintRange-error');
    await pause(500);
    await light.screenshot({ path: `${out}playground-loop-light.png` });

    // 3. The phone: the preview pane of the starter project.
    const phone = await openPlayground(browser, server.url, { width: 390, height: 800 });
    await phone.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await phone.click('#panes-nav [data-pane="preview"]');
    await pause(400);
    const pf = await previewFrame(phone);
    await pf.click('#button');
    await pause(400);
    await phone.screenshot({ path: `${out}playground-phone.png` });
    console.log(`wrote 3 images to ${out}`);
} finally {
    await browser.close();
    await server.close();
}

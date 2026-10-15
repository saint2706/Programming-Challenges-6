import { mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import axe from 'axe-core';
import WebSocket from 'ws';
import type { Browser, Page } from 'puppeteer-core';
import { afterAll, beforeAll, describe, expect, test } from 'vitest';
import { findChrome, launchChrome } from './chrome.ts';
import {
    type Harness,
    click,
    drag,
    errorsOf,
    inkIs,
    openTab,
    pickColor,
    pickTool,
    stageOrigin,
    startHarness,
    summary,
    until,
} from './support.ts';

let h: Harness;
let browser: Browser;
let n = 0;
const room = () => `room-${++n}`;
const base = () => h.server.url;

beforeAll(async () => {
    if (!findChrome()) throw new Error('No Chrome or Edge found. Set CHROME_PATH.');
    h = await startHarness();
    browser = await launchChrome();
});

afterAll(async () => {
    await browser?.close();
    await h?.stop();
});

/** Hold Ctrl (or not) around a keyboard action. */
async function chord(page: Page, keys: string[], mods: ('Control' | 'Shift')[] = ['Control']): Promise<void> {
    for (const m of mods) await page.keyboard.down(m);
    for (const k of keys) await page.keyboard.press(k as never);
    for (const m of mods.reverse()) await page.keyboard.up(m);
}

describe('drawing together', () => {
    test('a stroke drawn by one person appears for the other, pixel for pixel on the canvas', async () => {
        const r = room();
        const a = await openTab(browser, base(), r, { name: 'Ada' });
        const b = await openTab(browser, base(), r, { name: 'Grace' });
        await pickTool(a, 'Pen');
        await drag(a, [[200, 200], [400, 200]]);
        await until(b, (wb) => wb.doc.visible().length === 1);
        expect(await inkIs(b, 300, 200, true)).toBe(true);
        expect(await inkIs(b, 300, 400, false)).toBe(false);
        expect(await inkIs(a, 300, 200, true)).toBe(true);
        expect(errorsOf(a)).toEqual([]);
        expect(errorsOf(b)).toEqual([]);
    });

    test('others watch a stroke grow before it is finished', async () => {
        const r = room();
        const a = await openTab(browser, base(), r);
        const b = await openTab(browser, base(), r);
        await pickTool(a, 'Pen');
        const o = await stageOrigin(a);
        await a.mouse.move(o.x + 150, o.y + 150);
        await a.mouse.down();
        await a.mouse.move(o.x + 300, o.y + 260, { steps: 12 });
        await until(b, (wb) => wb.doc.live.size === 1);
        expect((await summary(b)).items).toHaveLength(0); // not committed yet, only previewed
        await a.mouse.up();
        await until(b, (wb) => wb.doc.visible().length === 1 && wb.doc.live.size === 0);
    });

    test('both people see each other in the people list and as cursors', async () => {
        const r = room();
        const a = await openTab(browser, base(), r, { name: 'Ada' });
        const b = await openTab(browser, base(), r, { name: 'Grace' });
        await until(a, (wb) => wb.doc.users.size === 2);
        expect(await a.$eval('#people', (el) => el.textContent)).toContain('Grace');
        expect(await b.$eval('#people', (el) => el.textContent)).toContain('Ada (you)'.replace(' (you)', ''));
        const o = await stageOrigin(b);
        await b.mouse.move(o.x + 500, o.y + 300, { steps: 5 });
        await until(a, (wb) => wb.doc.cursors.size === 1);
        await b.close();
        await until(a, (wb) => wb.doc.users.size === 1);
    });

    test('shapes, a sticky note, editing and moving all sync', async () => {
        const r = room();
        const a = await openTab(browser, base(), r, { name: 'Ada' });
        const b = await openTab(browser, base(), r, { name: 'Grace' });

        await pickTool(a, 'Rectangle');
        await drag(a, [[100, 100], [300, 220]]);
        await pickTool(a, 'Arrow');
        await drag(a, [[320, 160], [420, 160]]);
        await pickTool(a, 'Ellipse');
        await drag(a, [[450, 100], [600, 220]]);
        await pickTool(a, 'Line');
        await drag(a, [[100, 260], [300, 300]]);

        await pickTool(a, 'Sticky note');
        await click(a, 650, 120);
        await a.keyboard.type('Plan');
        await chord(a, ['Enter']);
        await until(b, (wb) => wb.doc.visible().length === 5);
        const seen = await summary(b);
        expect(seen.items.map((i) => i.k ?? i.t).sort()).toEqual(['arrow', 'ellipse', 'line', 'rect', 'text']);
        expect(seen.items.find((i) => i.t === 'text')?.text).toBe('Plan');

        // Edit the note's text by double-clicking it with the select tool.
        await pickTool(a, 'Select and move');
        const dbl = await stageOrigin(a);
        await a.mouse.click(dbl.x + 700, dbl.y + 180, { count: 2 });
        await a.waitForSelector('#editor:not([hidden])');
        await chord(a, ['a']);
        await a.keyboard.type('Plan v2');
        await chord(a, ['Enter']);
        await until(b, (wb) => wb.doc.visible().some((v) => v.item.text === 'Plan v2'));

        // Move it.
        await drag(a, [[700, 180], [760, 230]], 6);
        await until(b, (wb) => wb.doc.visible().some((v) => v.item.text === 'Plan v2' && Math.abs(v.dx - 60) < 2 && Math.abs(v.dy - 50) < 2));
        expect(errorsOf(a)).toEqual([]);
    });

    test('the eraser removes a stroke for everyone', async () => {
        const r = room();
        const a = await openTab(browser, base(), r);
        const b = await openTab(browser, base(), r);
        await pickTool(b, 'Pen');
        await drag(b, [[200, 300], [500, 300]]);
        await until(a, (wb) => wb.doc.visible().length === 1);
        await pickTool(a, 'Eraser');
        await drag(a, [[350, 260], [350, 340]]);
        await until(b, (wb) => wb.doc.visible().length === 0);
        expect(await inkIs(a, 350, 300, false)).toBe(false);
        expect(await inkIs(b, 350, 300, false)).toBe(false);
    });
});

describe('undo and redo are per person', () => {
    test('undo takes back only your own work', async () => {
        const r = room();
        const a = await openTab(browser, base(), r, { name: 'Ada' });
        const b = await openTab(browser, base(), r, { name: 'Grace' });
        await pickTool(a, 'Pen');
        await drag(a, [[100, 150], [300, 150]]);
        await pickTool(b, 'Pen');
        await drag(b, [[100, 350], [300, 350]]);
        await until(a, (wb) => wb.doc.visible().length === 2);
        await until(b, (wb) => wb.doc.visible().length === 2);

        expect(await b.$eval('button[aria-label="Redo"]', (el) => (el as HTMLButtonElement).disabled)).toBe(true);
        await chord(a, ['z']); // Ada undoes
        await until(a, (wb) => wb.doc.visible().length === 1);
        await until(b, (wb) => wb.doc.visible().length === 1);
        expect(await inkIs(b, 200, 350, true)).toBe(true); // Grace's line is still there
        expect(await inkIs(b, 200, 150, false)).toBe(false);

        await chord(a, ['z'], ['Control', 'Shift']); // redo
        await until(b, (wb) => wb.doc.visible().length === 2);

        await chord(b, ['z']); // Grace undoes: only her line goes
        await until(a, (wb) => wb.doc.visible().length === 1);
        expect(await inkIs(a, 200, 150, true)).toBe(true);
        expect(await inkIs(a, 200, 350, false)).toBe(false);
    });
});

describe('an infinite canvas', () => {
    test('wheel pans, ctrl+wheel zooms around the pointer, and the buttons and keys agree', async () => {
        const a = await openTab(browser, base(), room());
        const cam = () => a.evaluate(() => {
            const c = (window as unknown as { __wb: { board: { camera: { x: number; y: number; zoom: number; toWorld(x: number, y: number): [number, number] } } } }).__wb.board.camera;
            return { x: c.x, y: c.y, zoom: c.zoom, at: c.toWorld(300, 200) };
        });
        const o = await stageOrigin(a);
        await a.mouse.move(o.x + 300, o.y + 200);
        const before = await cam();
        await a.mouse.wheel({ deltaY: 200 });
        const panned = await cam();
        expect(panned.y).toBeCloseTo(before.y + 200, 0);
        expect(panned.zoom).toBe(1);

        const w0 = (await cam()).at;
        await a.keyboard.down('Control');
        await a.mouse.wheel({ deltaY: -300 });
        await a.keyboard.up('Control');
        const zoomed = await cam();
        expect(zoomed.zoom).toBeGreaterThan(1.5);
        expect(zoomed.at[0]).toBeCloseTo(w0[0], 0); // the point under the pointer did not move
        expect(zoomed.at[1]).toBeCloseTo(w0[1], 0);

        await a.click('button[aria-label="Reset zoom to 100%"]');
        expect((await cam()).zoom).toBeCloseTo(1, 5);
        await a.click('button[aria-label="Zoom in"]');
        expect(await a.$eval('.zoom', (el) => el.textContent)).toBe('125%');
        await a.keyboard.press('-');
        expect(await a.$eval('.zoom', (el) => el.textContent)).toBe('100%');
    });

    test('ink is stored in world units, so zoom levels agree across people', async () => {
        const r = room();
        const a = await openTab(browser, base(), r);
        const b = await openTab(browser, base(), r);
        await a.click('button[aria-label="Zoom in"]');
        await a.click('button[aria-label="Zoom in"]'); // 156%
        await pickTool(a, 'Line');
        await drag(a, [[200, 200], [200 + 156.25, 200]]); // 156.25 screen px at zoom 1.5625 = 100 world units
        await until(b, (wb) => wb.doc.visible().length === 1);
        const length = await b.evaluate(() => {
            const it = (window as unknown as { __wb: { doc: { visible(): { item: { x1: number; x2: number } }[] } } }).__wb.doc.visible()[0]!.item;
            return Math.abs(it.x2 - it.x1);
        });
        expect(length).toBeGreaterThan(98);
        expect(length).toBeLessThan(102);
    });

    test('space+drag pans and fit frames everything', async () => {
        const a = await openTab(browser, base(), room());
        await pickTool(a, 'Rectangle');
        await drag(a, [[100, 100], [200, 160]]);
        const x0 = await a.evaluate(() => (window as unknown as { __wb: { board: { camera: { x: number } } } }).__wb.board.camera.x);
        await a.keyboard.down('Space');
        await drag(a, [[400, 400], [300, 400]]);
        await a.keyboard.up('Space');
        const x1 = await a.evaluate(() => (window as unknown as { __wb: { board: { camera: { x: number } } } }).__wb.board.camera.x);
        expect(x1).toBeCloseTo(x0 + 100, 0);
        expect((await summary(a)).items).toHaveLength(1); // panning drew nothing

        // far-apart items, then Fit
        await a.keyboard.press('Equal');
        await pickTool(a, 'Rectangle');
        await drag(a, [[900, 500], [1000, 560]]);
        await a.keyboard.press('!');
        const inView = await a.evaluate(() => {
            const wb = (window as unknown as { __wb: { board: { camera: { toScreen(x: number, y: number): [number, number] } }; doc: { visible(): { item: { x1: number; y1: number; x2: number; y2: number } }[] } } }).__wb;
            return wb.doc.visible().every(({ item }) => [wb.board.camera.toScreen(item.x1, item.y1), wb.board.camera.toScreen(item.x2, item.y2)].every(([sx, sy]) => sx >= 0 && sy >= 0 && sx <= 1100 && sy <= 720));
        });
        expect(inView).toBe(true);
    });

    test('a two-finger pinch zooms and does not draw', async () => {
        const a = await openTab(browser, base(), room(), { touch: true });
        await pickTool(a, 'Pen');
        const o = await stageOrigin(a);
        const cdp = await a.createCDPSession();
        const touch = (type: string, points: { x: number; y: number; id: number }[]) =>
            cdp.send('Input.dispatchTouchEvent', { type, touchPoints: points.map((p) => ({ ...p, x: o.x + p.x, y: o.y + p.y })) } as never);
        await touch('touchStart', [{ x: 400, y: 300, id: 1 }]);
        await touch('touchStart', [{ x: 400, y: 300, id: 1 }, { x: 500, y: 300, id: 2 }]);
        for (let i = 1; i <= 8; i++) {
            await touch('touchMove', [{ x: 400 - i * 10, y: 300, id: 1 }, { x: 500 + i * 10, y: 300, id: 2 }]);
        }
        await touch('touchEnd', []);
        const zoom = await a.evaluate(() => (window as unknown as { __wb: { board: { camera: { zoom: number } } } }).__wb.board.camera.zoom);
        expect(zoom).toBeGreaterThan(1.5);
        expect((await summary(a)).items).toHaveLength(0);
    });
});

describe('sharing, saving and recovering', () => {
    test('someone who joins late sees the board, and so does everyone after a server restart', async () => {
        const r = room();
        const a = await openTab(browser, base(), r);
        await pickTool(a, 'Pen');
        await drag(a, [[100, 100], [300, 100]]);
        await pickTool(a, 'Rectangle');
        await drag(a, [[100, 200], [300, 300]]);
        await pickTool(a, 'Ellipse');
        await drag(a, [[400, 200], [500, 300]]);
        const late = await openTab(browser, base(), r);
        await until(late, (wb) => wb.doc.visible().length === 3);
        expect(await inkIs(late, 200, 100, true)).toBe(true);

        await new Promise((res) => setTimeout(res, 300)); // let the debounced save land
        await h.down();
        await h.up();
        const afterRestart = await openTab(browser, base(), r);
        await until(afterRestart, (wb) => wb.doc.visible().length === 3);
        expect(await inkIs(afterRestart, 200, 100, true)).toBe(true);
    });

    test('rooms do not leak into each other', async () => {
        const a = await openTab(browser, base(), room());
        const b = await openTab(browser, base(), room());
        await pickTool(a, 'Pen');
        await drag(a, [[100, 100], [300, 100]]);
        await new Promise((res) => setTimeout(res, 400));
        expect((await summary(b)).items).toHaveLength(0);
    });

    test('work done while the server is down is kept, then sent exactly once when it returns', async () => {
        const r = room();
        const a = await openTab(browser, base(), r, { name: 'Ada' });
        const b = await openTab(browser, base(), r, { name: 'Grace' });
        await pickTool(a, 'Pen');
        await drag(a, [[100, 100], [300, 100]]);
        await until(b, (wb) => wb.doc.visible().length === 1);

        await h.down();
        await until(a, (wb) => wb.conn.status === 'connecting');
        await drag(a, [[100, 200], [300, 200]]); // drawn offline
        await drag(a, [[100, 300], [300, 300]]);
        const offline = await summary(a);
        expect(offline.items).toHaveLength(3); // still on screen
        expect(offline.pending).toBe(2);
        expect(await a.$eval('#status', (el) => el.textContent)).toContain('will sync');

        await h.up();
        // Backoff can be up to 8 s between attempts, and a loaded CI runner is slower still.
        await until(a, (wb) => wb.conn.status === 'online' && wb.doc.pendingCount === 0, 30_000);
        await until(b, (wb) => wb.conn.status === 'online' && wb.doc.visible().length === 3, 30_000);
        await new Promise((res) => setTimeout(res, 300));
        expect((await summary(a)).items).toHaveLength(3); // not duplicated by the resend
        expect((await summary(b)).items).toHaveLength(3);
    }, 90_000);

    test('PNG and SVG exports contain the board and agree with each other', async () => {
        const r = room();
        const a = await openTab(browser, base(), r);
        await pickColor(a, '#d1242f');
        await pickTool(a, 'Pen');
        await drag(a, [[150, 150], [250, 200], [350, 150]], 6);
        await pickColor(a, '#0969da');
        await pickTool(a, 'Rectangle');
        await drag(a, [[160, 260], [340, 360]]);
        await pickTool(a, 'Ellipse');
        await drag(a, [[400, 240], [560, 360]]);

        const downloads = mkdtempSync(join(tmpdir(), 'wb-dl-'));
        try {
            const cdp = await a.createCDPSession();
            await cdp.send('Page.setDownloadBehavior', { behavior: 'allow', downloadPath: downloads } as never);
            const waitFor = async (suffix: string) => {
                for (let i = 0; i < 100; i++) {
                    const f = readdirSync(downloads).find((x) => x.endsWith(suffix));
                    if (f) return join(downloads, f);
                    await new Promise((res) => setTimeout(res, 100));
                }
                throw new Error(`no ${suffix} download`);
            };
            await a.click('button[aria-label="Export as PNG"]');
            const png = readFileSync(await waitFor('.png'));
            await a.click('button[aria-label="Export as SVG"]');
            const svg = readFileSync(await waitFor('.svg'), 'utf8');

            expect([...png.subarray(0, 8)]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
            const width = png.readUInt32BE(16);
            const height = png.readUInt32BE(20);
            // Board is about 410x230 world units plus 24 padding each side, drawn at 2x.
            expect(width).toBeGreaterThan(800);
            expect(width).toBeLessThan(1100);
            expect(height).toBeGreaterThan(400);
            expect(svg.startsWith('<svg xmlns="http://www.w3.org/2000/svg"')).toBe(true);

            // Draw both into one canvas and compare: they come from different code paths.
            const diff = await a.evaluate(
                async (pngB64: string, svgText: string) => {
                    const load = (src: string) =>
                        new Promise<HTMLImageElement>((resolve, reject) => {
                            const img = new Image();
                            img.onload = () => resolve(img);
                            img.onerror = () => reject(new Error('image failed to load'));
                            img.src = src;
                        });
                    const parsed = new DOMParser().parseFromString(svgText, 'image/svg+xml');
                    if (parsed.querySelector('parsererror')) throw new Error('SVG is not well formed');
                    const [pngImg, svgImg] = await Promise.all([
                        load(`data:image/png;base64,${pngB64}`),
                        load(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`),
                    ]);
                    const W = 600;
                    const H = Math.round((W * pngImg.height) / pngImg.width);
                    const grab = (img: HTMLImageElement) => {
                        const c = document.createElement('canvas');
                        c.width = W;
                        c.height = H;
                        const ctx = c.getContext('2d')!;
                        ctx.fillStyle = '#fff';
                        ctx.fillRect(0, 0, W, H);
                        ctx.drawImage(img, 0, 0, W, H);
                        return ctx.getImageData(0, 0, W, H).data;
                    };
                    const p = grab(pngImg);
                    const s = grab(svgImg);
                    let ink = 0;
                    let differing = 0;
                    for (let i = 0; i < p.length; i += 4) {
                        const isInk = p[i]! < 200 || s[i]! < 200;
                        if (isInk) ink++;
                        if (Math.abs(p[i]! - s[i]!) + Math.abs(p[i + 1]! - s[i + 1]!) + Math.abs(p[i + 2]! - s[i + 2]!) > 150) differing++;
                    }
                    return { ink, differing, total: p.length / 4 };
                },
                png.toString('base64'),
                svg,
            );
            expect(diff.ink).toBeGreaterThan(1000);
            expect(diff.differing / diff.ink).toBeLessThan(0.1); // under 10% of inked pixels disagree
        } finally {
            rmSync(downloads, { recursive: true, force: true });
        }
    });

    test('exporting an empty board says so instead of downloading nothing', async () => {
        const a = await openTab(browser, base(), room());
        await a.click('button[aria-label="Export as PNG"]');
        await a.waitForSelector('.toast');
        expect(await a.$eval('.toast', (el) => el.textContent)).toContain('nothing');
    });
});

describe('hostile content', () => {
    test('names and notes are always text, never markup', async () => {
        const r = room();
        const evil = await openTab(browser, base(), r, { name: '<img src=x onerror=p()>' });
        await evil.evaluate(() => ((window as unknown as { p: () => number }).p = () => ((window as unknown as { __pwned: number }).__pwned = 1)));
        const victim = await openTab(browser, base(), r, { name: 'Victim' });
        await victim.evaluate(() => ((window as unknown as { p: () => number }).p = () => ((window as unknown as { __pwned: number }).__pwned = 1)));
        await until(victim, (wb) => wb.doc.users.size === 2);

        await pickTool(evil, 'Text');
        await click(evil, 300, 200);
        await evil.keyboard.type('<script>window.__pwned=2</script><img src=x onerror=p()>');
        await chord(evil, ['Enter']);
        await until(victim, (wb) => wb.doc.visible().length === 1);
        await new Promise((res) => setTimeout(res, 800)); // the screen-reader summary refreshes after a short delay

        const probe = await victim.evaluate(() => ({
            pwned: (window as unknown as { __pwned?: number }).__pwned,
            images: document.querySelectorAll('#people img, #notes img, #toasts img').length,
            scripts: document.querySelectorAll('#notes script, #people script').length,
            people: document.getElementById('people')!.textContent,
            note: document.querySelector('#notes li')?.textContent,
        }));
        expect(probe.pwned).toBeUndefined();
        expect(probe.images).toBe(0);
        expect(probe.scripts).toBe(0);
        expect(probe.people).toContain('<img src=x onerror=p()>');
        expect(probe.note).toBe('<script>window.__pwned=2</script><img src=x onerror=p()>');
        expect(errorsOf(victim).filter((e) => !/favicon/.test(e))).toEqual([]);
    });
});

describe('a hostile client', () => {
    test('a 2,000-letter sticky note posted straight to the socket does not freeze anyone, and can be erased', async () => {
        const r = room();
        // Not a browser: a script speaking the protocol, sending what the page itself never would.
        const attacker = new WebSocket(`${base().replace('http', 'ws')}/ws/${r}`);
        await new Promise<void>((resolve) => attacker.once('open', () => resolve()));
        attacker.send(JSON.stringify({ t: 'hello', v: 1, id: 'attacker-secret', name: 'x', last: null }));
        attacker.send(
            JSON.stringify({
                t: 'op',
                opId: 'evil1',
                op: { k: 'add', item: { t: 'text', id: 'evil', color: '#000000', size: 20, x: 0, y: 0, text: 'A'.repeat(2000), sticky: true } },
            }),
        );
        await new Promise((res) => setTimeout(res, 300));

        const victim = await openTab(browser, base(), r, { name: 'Victim' });
        await until(victim, (wb) => wb.doc.visible().length === 1);
        const cost = await victim.evaluate(() => {
            const wb = (window as unknown as { __wb: { board: { paint(): void; baseDirty: boolean } } }).__wb;
            const t0 = performance.now();
            for (let i = 0; i < 20; i++) {
                wb.board.baseDirty = true;
                wb.board.paint();
            }
            return (performance.now() - t0) / 20;
        });
        expect(cost).toBeLessThan(50); // milliseconds per repaint; before the fix this took seconds

        // The page still answers, so it can erase the note.
        await pickTool(victim, 'Eraser');
        const o = await stageOrigin(victim);
        await victim.mouse.move(o.x + 1100 / 2 + 50, o.y + 720 / 2 + 40);
        await victim.mouse.down();
        await victim.mouse.move(o.x + 1100 / 2 + 80, o.y + 720 / 2 + 60, { steps: 6 });
        await victim.mouse.up();
        await until(victim, (wb) => wb.doc.visible().length === 0);
        attacker.close();
    });
});

describe('keyboard and assistive technology', () => {
    test('shortcuts switch tools, and a note can be made without a mouse', async () => {
        const a = await openTab(browser, base(), room());
        await a.focus('#top');
        const pressed = () => a.$eval('button[aria-pressed="true"]', (el) => el.getAttribute('aria-label'));
        await a.keyboard.press('e');
        expect(await pressed()).toBe('Eraser');
        await a.keyboard.press('r');
        expect(await pressed()).toBe('Rectangle');
        await a.keyboard.press('n');
        expect(await pressed()).toBe('Sticky note');

        await a.keyboard.press('Enter'); // opens a note at the middle of the screen
        await a.waitForSelector('#editor:not([hidden])');
        await a.keyboard.type('From the keyboard');
        await chord(a, ['Enter']);
        await until(a, (wb) => wb.doc.visible().some((v) => v.item.text === 'From the keyboard'));

        await a.keyboard.press('Enter'); // another one, then Escape throws it away
        await a.waitForSelector('#editor:not([hidden])');
        await a.keyboard.type('never saved');
        await a.keyboard.press('Escape');
        await a.waitForSelector('#editor', { hidden: true });
        expect((await summary(a)).items).toHaveLength(1);

        // Typing in the name box must not trigger tool shortcuts.
        await a.click('#name');
        await a.keyboard.press('e');
        expect(await pressed()).toBe('Sticky note');
    });

    test('arrow keys pan the board and the canvas describes its contents', async () => {
        const a = await openTab(browser, base(), room());
        await a.focus('#top');
        const x0 = await a.evaluate(() => (window as unknown as { __wb: { board: { camera: { x: number } } } }).__wb.board.camera.x);
        await a.keyboard.press('ArrowRight');
        const x1 = await a.evaluate(() => (window as unknown as { __wb: { board: { camera: { x: number } } } }).__wb.board.camera.x);
        expect(x1).toBeGreaterThan(x0);
        await pickTool(a, 'Pen');
        await drag(a, [[100, 100], [200, 100]]);
        await until(a, () => document.getElementById('top')!.getAttribute('aria-label')!.includes('1 strokes'));
    });

    test('axe finds no accessibility violations on a busy board, desktop or phone', async () => {
        const r = room();
        const a = await openTab(browser, base(), r, { name: 'Ada' });
        await openTab(browser, base(), r, { name: 'Grace' });
        await pickTool(a, 'Pen');
        await drag(a, [[100, 100], [300, 140]]);
        await pickTool(a, 'Sticky note');
        await click(a, 400, 200);
        await a.keyboard.type('Remember the milk');
        await chord(a, ['Enter']);
        await until(a, (wb) => wb.doc.users.size === 2 && wb.doc.visible().length === 2);
        await new Promise((res) => setTimeout(res, 700));

        const phone = await openTab(browser, base(), r, { name: 'Linus', width: 360, height: 740 });
        await until(phone, (wb) => wb.doc.visible().length === 2);

        for (const page of [a, phone]) {
            await page.evaluate(axe.source);
            const violations = await page.evaluate(async () => {
                const result = await (window as unknown as { axe: { run(): Promise<{ violations: { id: string; help: string; nodes: { html: string }[] }[] }> } }).axe.run();
                return result.violations.map((v) => `${v.id}: ${v.help} (${v.nodes.map((nd) => nd.html.slice(0, 80)).join(' | ')})`);
            });
            expect(violations).toEqual([]);
        }
    });
});

describe('small screens', () => {
    test('a 360px phone has no sideways page scroll and every tool is reachable', async () => {
        const a = await openTab(browser, base(), room(), { width: 360, height: 740, touch: true });
        const probe = await a.evaluate(() => ({
            scrollW: document.documentElement.scrollWidth,
            headerH: document.querySelector('header')!.getBoundingClientRect().height,
            tools: document.querySelectorAll('.toolbar button.tool').length,
            toolbarScrolls: document.querySelector('.toolbar')!.scrollWidth > document.querySelector('.toolbar')!.clientWidth,
        }));
        expect(probe.scrollW).toBeLessThanOrEqual(360);
        expect(probe.headerH).toBeLessThan(150);
        expect(probe.tools).toBe(13); // 11 tools + undo + redo
        expect(probe.toolbarScrolls).toBe(true); // it scrolls instead of overflowing the page
        // A finger can draw.
        const o = await stageOrigin(a);
        await a.touchscreen.tap(o.x + 100, o.y + 300);
        await until(a, (wb) => wb.doc.visible().length === 1);
    });
});

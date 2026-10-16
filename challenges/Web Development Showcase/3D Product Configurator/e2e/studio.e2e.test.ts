import axe from 'axe-core';
import type { Browser, HTTPRequest, Page } from 'puppeteer-core';
import { afterAll, afterEach, beforeAll, describe, expect, test } from 'vitest';
import { cameraAbove, defaults, screenFromDiagonal } from '../src/headcoupled/calibration.ts';
import { CAMERA, faceLandmarks } from '../tests/helpers.ts';
import { findChrome, launchChrome } from './chrome.ts';
import { after, centroidOfHue, closeContexts, countHue, differ, newContext, openStudio, pixelAt, read, stagePixels, startServer, widestRunOf } from './support.ts';

let browser: Browser;
let cameraBrowser: Browser;
let server: Awaited<ReturnType<typeof startServer>>;
const base = () => server.url;

beforeAll(async () => {
    if (!findChrome()) throw new Error('No Chrome or Edge found. Set CHROME_PATH.');
    server = await startServer();
    browser = await launchChrome();
    // A second browser with a fake webcam (a test pattern, no face), for the camera tests.
    cameraBrowser = await launchChrome(['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream']);
});

afterEach(closeContexts);

afterAll(async () => {
    await browser?.close();
    await cameraBrowser?.close();
    await server?.close();
});

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const click = (page: Page, selector: string) => page.click(selector);

const mode = async (page: Page, name: 'Design' | 'Window preview') => {
    await page.evaluate((n: string) => [...document.querySelectorAll<HTMLElement>('.modes [role=radio]')].find((b) => b.textContent === n)!.click(), name);
    await sleep(300);
};
const choosePart = (page: Page, id: string) => click(page, `.part[data-id=${id}]`);
const swatch = (page: Page, name: string) => click(page, `.swatch[aria-label="${name}"]`);

const PATHS = [
    { name: 'WebGPU where there is one', query: '', backends: ['WebGPU', 'WebGL 2'] },
    { name: 'WebGL 2 forced', query: '&renderer=webgl', backends: ['WebGL 2'] },
];

describe.each(PATHS)('the design view ($name)', ({ query, backends }) => {
    test('loads, draws the shoe, says which way it is drawing, and logs nothing wrong', async () => {
        const { page, errors } = await openStudio(browser, base(), { query });
        expect(backends).toContain(await read(page, 'studio.backend'));
        expect(await read(page, 'studio.error')).toBe('');
        expect(await page.title()).toBe('Sneaker Configurator');
        const p = await stagePixels(page);
        expect(countHue(p, 205)).toBeGreaterThan(8000); // the blue knit upper
        expect(errors).toEqual([]);
    });

    test('every visible part can be coloured on its own, which shows the model really is split into parts', async () => {
        const { page } = await openStudio(browser, base(), { query });
        const colours: Record<string, [string, number]> = {
            upper: ['#ff0000', 0],
            overlays: ['#00ff00', 120],
            laces: ['#ffff00', 60],
            accents: ['#ff00ff', 300],
            lining: ['#00ffff', 180],
            midsole: ['#0000ff', 240],
            outsole: ['#8000ff', 270],
        };
        await after(page, () =>
            page.evaluate((c: Record<string, [string, number]>) => {
                const s = (window as unknown as { __app: { studio: { setColor(p: string, h: string): void } } }).__app.studio;
                for (const [part, [hex]] of Object.entries(c)) s.setColor(part, hex);
            }, colours),
        );
        const p = await stagePixels(page);
        for (const part of ['upper', 'overlays', 'laces', 'accents', 'lining', 'midsole']) {
            const [, hue] = colours[part]!;
            expect(countHue(p, hue, 20), part).toBeGreaterThan(120);
        }
        // The blue of the original upper is gone: the upper really took the new colour everywhere.
        expect(countHue(p, 205, 12)).toBeLessThan(200);
    });

    test('the outsole is a part too (it faces the floor, so it is seen from below the floor: the page does not allow that, so it is only checked to take the colour)', async () => {
        const { page } = await openStudio(browser, base(), { query });
        await choosePart(page, 'outsole');
        await swatch(page, 'Ember');
        expect(await read(page, "studio.design.colors.outsole")).toBe('#e8502f');
    });

    test('a colour is picked from the swatches, a preset or the colour box, and the part list shows it', async () => {
        const { page } = await openStudio(browser, base(), { query });
        const blue = await stagePixels(page);
        await choosePart(page, 'upper');
        await after(page, () => swatch(page, 'Ember'));
        const ember = await stagePixels(page);
        expect(countHue(ember, 12, 18)).toBeGreaterThan(8000);
        expect(countHue(ember, 205, 18)).toBeLessThan(countHue(blue, 205, 18) / 2); // what blue is left is the overlays and eyelets
        expect(await page.$eval('.part[data-id=upper] i', (el) => (el as HTMLElement).style.background)).toMatch(/rgb\(232, 80, 47\)|#e8502f/i);

        await after(page, () => page.evaluate(() => [...document.querySelectorAll<HTMLElement>('.chip')].find((b) => /Beach/.test(b.textContent ?? ''))!.click()));
        expect(await read(page, 'studio.design.colors.upper')).toBe('#ba9696');
        expect(await page.$eval('.chip[aria-pressed=true]', (el) => el.textContent?.trim())).toBe('Beach');

        await after(page, () =>
            page.$eval('.custom input', (el) => {
                const input = el as HTMLInputElement;
                input.value = '#00aa44';
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }),
        );
        expect(await read(page, 'studio.design.colors.upper')).toBe('#00aa44');
    });

    test('a finish changes how light plays on the part, not its colour: patent leather shines where matte does not', async () => {
        const { page } = await openStudio(browser, base(), { query });
        await choosePart(page, 'upper');
        await after(page, () => swatch(page, 'Ink'));
        const matte = await stagePixels(page);
        await after(page, () => page.evaluate(() => [...document.querySelectorAll<HTMLElement>('.finishes [role=radio]')].find((b) => b.textContent === 'Patent')!.click()));
        const patent = await stagePixels(page);
        const bright = (p: typeof matte) => {
            let n = 0;
            for (let i = 0; i < p.data.length; i += 3) if (p.data[i]! > 150 && p.data[i + 1]! > 150 && p.data[i + 2]! > 150 && i / 3 % p.width > 120 && i / 3 % p.width < p.width - 120) n++;
            return n;
        };
        expect(differ(matte, patent)).toBeGreaterThan(0.01);
        expect(bright(patent)).toBeGreaterThan(bright(matte) + 200);
    });
});

describe('the design view: moving around', () => {
    test('dragging turns the shoe, the wheel zooms, the arrow keys do the same without a mouse', async () => {
        const { page } = await openStudio(browser, base());
        const stage = (await (await page.$('.stage'))!.boundingBox())!;
        const cx = stage.x + stage.width / 2;
        const cy = stage.y + stage.height / 2;
        const before = await stagePixels(page);

        await page.mouse.move(cx, cy);
        await page.mouse.down();
        await page.mouse.move(cx + 220, cy, { steps: 12 });
        await page.mouse.up();
        await sleep(900);
        const dragged = await stagePixels(page);
        expect(differ(before, dragged)).toBeGreaterThan(0.05);
        expect(countHue(dragged, 205)).toBeGreaterThan(3000); // still there

        await page.mouse.wheel({ deltaY: -400 });
        await sleep(900);
        const zoomed = await stagePixels(page);
        expect(countHue(zoomed, 205)).toBeGreaterThan(countHue(dragged, 205) * 1.15);

        await page.focus('.stage');
        const keyed0 = await stagePixels(page);
        for (let i = 0; i < 4; i++) await page.keyboard.press('ArrowRight');
        await sleep(300);
        const keyed1 = await stagePixels(page);
        expect(differ(keyed0, keyed1)).toBeGreaterThan(0.03);
        for (let i = 0; i < 8; i++) await page.keyboard.press('-');
        await sleep(300);
        const out = await stagePixels(page);
        expect(countHue(out, 205)).toBeLessThan(countHue(keyed1, 205));
    });

    test('the view is bounded: zooming never goes through the shoe or far away, and the camera never goes below the floor', async () => {
        const { page } = await openStudio(browser, base());
        await page.focus('.stage');
        for (let i = 0; i < 60; i++) await page.keyboard.press('+');
        for (let i = 0; i < 40; i++) await page.keyboard.press('ArrowDown');
        await sleep(300);
        const near = await stagePixels(page);
        expect(countHue(near, 205)).toBeGreaterThan(5000);
        for (let i = 0; i < 80; i++) await page.keyboard.press('-');
        for (let i = 0; i < 40; i++) await page.keyboard.press('ArrowUp');
        await sleep(300);
        const far = await stagePixels(page);
        expect(countHue(far, 205)).toBeGreaterThan(500); // never lost
    });

    test('the design is remembered, a corrupt memory is ignored, and Reset goes back', async () => {
        const { page } = await openStudio(browser, base());
        await choosePart(page, 'laces');
        await swatch(page, 'Volt');
        await sleep(200);
        await page.reload();
        await page.waitForFunction(() => (window as unknown as { __app: { studio: { ready: boolean } } }).__app?.studio.ready, { timeout: 60_000 });
        expect(await read(page, 'studio.design.colors.laces')).toBe('#c6ff1a');
        await click(page, '.footer .btn');
        expect(await read(page, 'studio.design.colors.laces')).toBe('#2e2e2e');

        const bad = await openStudio(browser, base(), { storage: { 'configurator:design:v1': '{"colors":{"upper":"url(javascript:alert(1))"}}' } });
        expect(await read(bad.page, 'studio.design.colors.upper')).toBe('#136690');
        expect(bad.errors).toEqual([]);
    });
});

describe('the window preview', () => {
    test('the picture follows the eye: the room shifts, the shoe in front shifts less than the wall behind it, and the glass stays put', async () => {
        const { page } = await openStudio(browser, base(), { query: '&mode=window' });
        // A red upper: nothing else in the room is red, so the shoe's position can be read off the picture.
        await page.evaluate(() => (window as unknown as { __app: { studio: { setColor(p: string, h: string): void } } }).__app.studio.setColor('upper', '#ff2a00'));
        const shots: Record<string, Awaited<ReturnType<typeof stagePixels>>> = {};
        for (const [name, x] of [['left', -0.1], ['mid', 0], ['right', 0.1]] as const) {
            await page.evaluate((eye) => ((window as unknown as { __app: { studio: { eyeOverride: unknown } } }).__app.studio.eyeOverride = eye), { x, y: 0, z: 0.55 });
            await sleep(500);
            shots[name] = await stagePixels(page);
        }
        expect(await read(page, 'studio.seen.source')).toBe('override');
        // Looking through a window from the left, what is behind it shifts to the left, and the further back, the more.
        // The back wall is the big flat panel above the shoe: its colour from the middle picture, then where that colour is in the others.
        const row = Math.round(shots.mid!.height * 0.3);
        const like = pixelAt(shots.mid!, Math.round(shots.mid!.width / 2), row);
        const centreOf = (p: typeof shots.mid) => {
            const run = widestRunOf(p!, row, like);
            expect(run.right - run.left, 'the back wall is found').toBeGreaterThan(80);
            return (run.left + run.right) / 2;
        };
        const wallFollows = centreOf(shots.right) - centreOf(shots.left); // positive: the wall goes the way the eye goes
        const shoeFollows = centroidOfHue(shots.right!, 10, 20)!.x - centroidOfHue(shots.left!, 10, 20)!.x;
        expect(wallFollows).toBeGreaterThan(60);
        expect(shoeFollows).toBeGreaterThan(10); // the shoe goes the same way ...
        expect(shoeFollows).toBeLessThan(wallFollows * 0.8); // ... but less, because it is nearer the glass: that is the depth
        // The glass does not move: the window's frame is the picture's frame from every eye, so every corner stays room, never page background.
        for (const p of [shots.left!, shots.mid!, shots.right!]) {
            for (const [x, y] of [[3, 3], [p.width - 4, 3], [3, p.height - 4], [p.width - 4, p.height - 4]]) {
                const i = (y! * p.width + x!) * 3;
                const [, s] = [0, Math.max(p.data[i]!, p.data[i + 1]!, p.data[i + 2]!) - Math.min(p.data[i]!, p.data[i + 1]!, p.data[i + 2]!)];
                expect(s).toBeLessThan(60); // a grey room wall, not the blue shoe or an accent
                expect(p.data[i]!).toBeGreaterThan(40); // and not the clear colour of an empty canvas
            }
        }
    });

    test('without a camera the pointer is the head', async () => {
        const { page } = await openStudio(browser, base(), { query: '&mode=window' });
        const stage = (await (await page.$('.stage'))!.boundingBox())!;
        await page.mouse.move(stage.x + 30, stage.y + stage.height / 2);
        await sleep(900);
        expect(await read(page, 'studio.seen.source')).toBe('pointer');
        const leftEye = await read<number>(page, 'studio.seen.eye.x');
        await page.mouse.move(stage.x + stage.width - 30, stage.y + stage.height / 2);
        await sleep(900);
        const rightEye = await read<number>(page, 'studio.seen.eye.x');
        expect(leftEye).toBeLessThan(-0.05);
        expect(rightEye).toBeGreaterThan(0.05);
    });

    test('the window is as big as the canvas physically is, so a bigger screen size means a bigger room; the shoe is life size when it fits', async () => {
        const { page } = await openStudio(browser, base(), { query: '&mode=window' });
        const small = await read<number>(page, 'studio.seen.width');
        await page.select('select', '27');
        await sleep(400);
        const big = await read<number>(page, 'studio.seen.width');
        expect(big / small).toBeGreaterThan(1.5);
        expect(await read<number>(page, 'studio.shoeScale')).toBeCloseTo(1, 1);
        expect(await page.$eval('.panel', (el) => el.textContent)).toMatch(/life size/);
    });

    test('landmarks of a head at a known place put the eye there, left is left and near is near; a lost face drifts back to the middle', async () => {
        const { page } = await openStudio(browser, base(), { query: '&mode=window' });
        const s = await page.evaluate(() => (window as unknown as { __app: { head: { settings: unknown } } }).__app.head.settings) as ReturnType<typeof defaults>;
        const above = cameraAbove(screenFromDiagonal(s.diagonal, s.aspect));
        const eye = { x: 0.12, y: 0.03, z: 0.5 };
        const lm = faceLandmarks({ x: -eye.x, y: above - eye.y, z: eye.z }, 0.063, CAMERA);
        await page.evaluate((l, size) => (window as unknown as { __app: { head: { inject(a: unknown, s: unknown): void } } }).__app.head.inject(l, size), lm, { width: CAMERA.width, height: CAMERA.height });
        await sleep(400);
        expect(await read(page, 'head.status')).toBe('tracking');
        expect(await read<number>(page, 'head.eye.x')).toBeCloseTo(eye.x, 3);
        expect(await read<number>(page, 'head.eye.z')).toBeCloseTo(eye.z, 3);
        expect(await read(page, 'studio.seen.source')).toBe('camera');
        await page.evaluate(() => (window as unknown as { __app: { head: { inject(a: unknown): void } } }).__app.head.inject(null));
        expect(await read(page, 'head.status')).toBe('no-face');
        expect(await page.$eval('.panel', (el) => el.textContent)).toMatch(/Looking for a face/);
    });

    test('the numbers can be shown, and say how the picture is being drawn', async () => {
        const { page } = await openStudio(browser, base(), { query: '&mode=window' });
        await page.evaluate(() => [...document.querySelectorAll<HTMLElement>('.panel .btn')].find((b) => b.textContent === 'Show numbers')!.click());
        await sleep(600);
        const text = await page.$eval('.hud', (el) => el.textContent ?? '');
        expect(text).toMatch(/WebGPU|WebGL 2/);
        expect(text).toMatch(/Eye \(x, y, z\)/);
        expect(text).toMatch(/\d+ fps/);
    });

    test('going between the two views many times does not leak: the renderer holds the same geometry and textures at the end as after the first round', async () => {
        const { page } = await openStudio(browser, base());
        const cycle = async () => {
            await mode(page, 'Window preview');
            await sleep(500);
            await mode(page, 'Design');
            await sleep(500);
        };
        await cycle();
        const first = await read<{ geometries: number; textures: number }>(page, 'studio.memory()');
        for (let i = 0; i < 4; i++) await cycle();
        const last = await read<{ geometries: number; textures: number }>(page, 'studio.memory()');
        expect(last.geometries).toBeLessThanOrEqual(first.geometries);
        expect(last.textures).toBeLessThanOrEqual(first.textures);
    });
});

describe('the camera', () => {
    test('with a webcam: asks, shows what it sees with a camera-on bar, runs the face model, and stops when told', async () => {
        const { page, errors } = await openStudio(cameraBrowser, base(), { query: '&mode=window' });
        await click(page, '.panel .btn.primary'); // Use my camera
        await page.waitForFunction(() => ['no-face', 'tracking', 'error'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 90_000 });
        expect(await read(page, 'head.note')).not.toMatch(/could not start/);
        expect(['GPU', 'CPU']).toContain(await read(page, 'head.delegate'));
        await page.waitForFunction(() => (window as unknown as { __app: { head: { fps: number } } }).__app.head.fps > 3, { timeout: 30_000 });
        expect(await page.$eval('.inset .what', (el) => el.textContent)).toMatch(/Camera on/);
        expect(await page.$('.inset video')).not.toBeNull();
        expect(await page.evaluate(() => (window as unknown as { __app: { head: { stream: MediaStream | null } } }).__app.head.stream?.getVideoTracks().some((t) => t.readyState === 'live'))).toBe(true);

        await click(page, '.inset .stop');
        await sleep(300);
        expect(await read(page, 'head.status')).toBe('off');
        expect(await page.$('.inset')).toBeNull();
        expect(await read(page, 'head.stream')).toBeNull();
        expect(errors.filter((e) => !/GPU|gpu|WebGL|webgl|GL |ANGLE|swiftshader|face_landmarker|XNNPACK|TensorFlow Lite/i.test(e))).toEqual([]);
    });

    test('leaving the window preview, or hiding the tab, turns the camera off, and it stays off until it is asked for again', async () => {
        const { page } = await openStudio(cameraBrowser, base(), { query: '&mode=window' });
        await click(page, '.panel .btn.primary');
        await page.waitForFunction(() => ['no-face', 'tracking'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 90_000 });

        await page.evaluate(() => {
            Object.defineProperty(document, 'hidden', { configurable: true, get: () => true });
            document.dispatchEvent(new Event('visibilitychange'));
        });
        await sleep(300);
        expect(await read(page, 'head.status')).toBe('off');
        expect(await read(page, 'head.stream')).toBeNull();
        expect(await read(page, 'head.note')).toMatch(/background/);
        await page.evaluate(() => {
            Object.defineProperty(document, 'hidden', { configurable: true, get: () => false });
            document.dispatchEvent(new Event('visibilitychange'));
        });
        await sleep(800);
        expect(await read(page, 'head.status')).toBe('off'); // coming back does not turn it on
        expect(await read(page, 'head.stream')).toBeNull();
        await click(page, '.panel .btn.primary');
        await page.waitForFunction(() => ['no-face', 'tracking'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 30_000 });

        await mode(page, 'Design');
        await sleep(300);
        expect(await read(page, 'head.status')).toBe('off');
        expect(await read(page, 'head.stream')).toBeNull();
    });

    test('turning the camera off and on while the browser is still asking never leaves a camera running that nobody sees', async () => {
        const context = await newContext(cameraBrowser);
        const page = await context.newPage();
        await page.setViewport({ width: 1280, height: 800 });
        await page.evaluateOnNewDocument(() => {
            // Permission that takes a while to be given, as a person reading the prompt does.
            const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
            (window as unknown as { __streams: MediaStream[] }).__streams = [];
            navigator.mediaDevices.getUserMedia = async (constraints) => {
                await new Promise((r) => setTimeout(r, 700));
                const stream = await real(constraints);
                (window as unknown as { __streams: MediaStream[] }).__streams.push(stream);
                return stream;
            };
        });
        await page.goto(`${base()}/?e2e=1&mode=window`);
        await page.waitForFunction(() => (window as unknown as { __app?: { studio: { ready: boolean } } }).__app?.studio.ready, { timeout: 60_000 });
        for (const action of ['start', 'stop', 'start', 'stop', 'start']) {
            await page.evaluate((a: string) => {
                const head = (window as unknown as { __app: { head: { start(): void; stop(): void } } }).__app.head;
                if (a === 'start') void head.start();
                else head.stop();
            }, action);
            await sleep(60);
        }
        await page.waitForFunction(() => ['no-face', 'tracking'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 90_000 });
        await sleep(1500);
        const live = await page.evaluate(() => (window as unknown as { __streams: MediaStream[] }).__streams.filter((s) => s.getVideoTracks().some((t) => t.readyState === 'live')).length);
        expect(live).toBe(1); // the one in use; every other the browser handed over was let go
        expect(await page.evaluate(() => (window as unknown as { __app: { head: { stream: MediaStream | null } } }).__app.head.stream?.getVideoTracks()[0]?.readyState)).toBe('live');
        // And stopping now really stops everything.
        await page.evaluate(() => (window as unknown as { __app: { head: { stop(): void } } }).__app.head.stop());
        const after = await page.evaluate(() => (window as unknown as { __streams: MediaStream[] }).__streams.filter((s) => s.getVideoTracks().some((t) => t.readyState === 'live')).length);
        expect(after).toBe(0);
    });

    test('a camera that is unplugged, or whose sharing is ended in the browser, is noticed: the bar goes and the page says so', async () => {
        const { page } = await openStudio(cameraBrowser, base(), { query: '&mode=window' });
        await click(page, '.panel .btn.primary');
        await page.waitForFunction(() => ['no-face', 'tracking'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 90_000 });
        await page.evaluate(() => (window as unknown as { __app: { head: { stream: MediaStream } } }).__app.head.stream.getVideoTracks()[0]!.dispatchEvent(new Event('ended')));
        await sleep(200);
        expect(await read(page, 'head.status')).toBe('off');
        expect(await page.$('.inset')).toBeNull();
        expect(await read(page, 'head.note')).toMatch(/camera stopped/);
    });

    test('a blocked camera is explained, and the pointer carries on', async () => {
        const context = await newContext(browser);
        const page = await context.newPage();
        await page.setViewport({ width: 1280, height: 800 });
        await page.evaluateOnNewDocument(() => {
            navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('no', 'NotAllowedError'));
        });
        await page.goto(`${base()}/?e2e=1&mode=window`);
        await page.waitForFunction(() => (window as unknown as { __app?: { studio: { ready: boolean } } }).__app?.studio.ready, { timeout: 60_000 });
        await click(page, '.panel .btn.primary');
        await page.waitForFunction(() => (window as unknown as { __app: { head: { status: string } } }).__app.head.status === 'denied');
        expect(await page.$eval('.panel [role=status]', (el) => el.textContent)).toMatch(/blocked/);
        expect(await page.$('.inset')).toBeNull();
        expect(await read(page, 'studio.seen.source')).toBe('pointer');
    });
});

describe('what the page does with the network and the browser', () => {
    test('everything comes from this site: the model, the lighting, the face model and its runtime; nothing from a CDN, and no policy violation', async () => {
        const context = await newContext(cameraBrowser);
        const page = await context.newPage();
        await page.setViewport({ width: 1280, height: 800 });
        const hosts = new Set<string>();
        await page.setRequestInterception(true);
        page.on('request', (r: HTTPRequest) => {
            const url = new URL(r.url());
            if (/^https?:$/.test(url.protocol)) hosts.add(url.host);
            void r.continue();
        });
        await page.evaluateOnNewDocument(() => {
            (window as unknown as { __violations: string[] }).__violations = [];
            document.addEventListener('securitypolicyviolation', (e) => (window as unknown as { __violations: string[] }).__violations.push(`${e.violatedDirective} ${e.blockedURI}`));
        });
        await page.goto(`${base()}/?e2e=1&mode=window`);
        await page.waitForFunction(() => (window as unknown as { __app?: { studio: { ready: boolean } } }).__app?.studio.ready, { timeout: 60_000 });
        await click(page, '.panel .btn.primary');
        await page.waitForFunction(() => ['no-face', 'tracking'].includes((window as unknown as { __app: { head: { status: string } } }).__app.head.status), { timeout: 90_000 });
        await page.evaluate(() => (window as unknown as { __app: { head: { stop(): void } } }).__app.head.stop());
        expect([...hosts]).toEqual([new URL(base()).host]);
        expect(await page.evaluate(() => (window as unknown as { __violations: string[] }).__violations)).toEqual([]);
    });

    test('its content security policy allows only its own origin, and no inline or evaluated script', async () => {
        const { page } = await openStudio(browser, base());
        const csp = await page.$eval('meta[http-equiv="Content-Security-Policy"]', (el) => el.getAttribute('content') ?? '');
        expect(csp).toMatch(/default-src 'none'/);
        expect(csp).toMatch(/script-src 'self' 'wasm-unsafe-eval'/);
        expect(csp).not.toMatch(/unsafe-inline|unsafe-eval(?!')/);
        expect(csp).toMatch(/connect-src 'self'/);
    });
});

describe('everyone can use it', () => {
    const scan = async (page: Page) => {
        await page.evaluate(axe.source);
        return page.evaluate(async () => {
            const result = await (window as unknown as { axe: { run(): Promise<{ violations: { id: string; impact: string; nodes: { html: string }[] }[] }> } }).axe.run();
            return result.violations.map((v) => `${v.id} (${v.impact}): ${v.nodes.map((n) => n.html.slice(0, 100)).join(' | ')}`);
        });
    };

    test.each(['light', 'dark'] as const)('axe finds nothing in either view in %s mode', async (scheme) => {
        const { page } = await openStudio(browser, base());
        await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
        expect(await scan(page)).toEqual([]);
        await mode(page, 'Window preview');
        await click(page, '.panel .btn[aria-pressed]'); // Show numbers
        expect(await scan(page)).toEqual([]);
    });

    test('every control can be reached and used from the keyboard: parts and finishes move with the arrows', async () => {
        const { page } = await openStudio(browser, base());
        await page.focus('.part[aria-checked=true]');
        await page.keyboard.press('ArrowDown');
        expect(await read(page, 'studio.selected')).toBe('overlays');
        await page.keyboard.press('ArrowUp');
        await page.keyboard.press('ArrowUp');
        expect(await read(page, 'studio.selected')).toBe('outsole'); // wraps round
        await page.focus('.finishes [aria-checked=true]');
        await page.keyboard.press('ArrowRight');
        expect(await read(page, 'studio.design.finishes.outsole')).toBe('satin');
        expect(await page.evaluate(() => document.activeElement?.textContent)).toBe('Satin');
    });

    test('a phone gets the picture and the controls one above the other, with no sideways scroll, in both views', async () => {
        const { page } = await openStudio(browser, base(), { width: 390, height: 800 });
        for (const view of ['Design', 'Window preview'] as const) {
            await mode(page, view);
            const [scrollWidth, clientWidth] = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
            expect(scrollWidth, view).toBeLessThanOrEqual(clientWidth);
            const stage = (await (await page.$('.stage'))!.boundingBox())!;
            expect(stage.width).toBeGreaterThan(300);
            expect(stage.height).toBeGreaterThan(250);
            const panel = (await (await page.$('.panel'))!.boundingBox())!;
            expect(panel.y).toBeGreaterThanOrEqual(stage.y + stage.height - 1);
        }
    });

    test('with reduced motion nothing animates in the interface', async () => {
        const { page } = await openStudio(browser, base());
        await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }]);
        const durations = await page.evaluate(() => [...document.querySelectorAll('button')].map((b) => getComputedStyle(b).transitionDuration));
        expect(durations.every((d) => d === '0s')).toBe(true);
    });
});

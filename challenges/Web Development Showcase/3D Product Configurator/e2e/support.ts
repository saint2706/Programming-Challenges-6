import { fileURLToPath } from 'node:url';
import sharp from 'sharp';
import type { Browser, BrowserContext, Page } from 'puppeteer-core';
import { preview } from 'vite';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

/** Serve the built app (`dist/`) on a free local port. Run `vite build` first. */
export async function startServer(): Promise<{ url: string; close(): Promise<void> }> {
    const server = await preview({ root: ROOT, logLevel: 'silent', preview: { host: '127.0.0.1', port: 4290, strictPort: false } });
    const url = server.resolvedUrls!.local[0]!.replace(/\/$/, '');
    return { url, close: () => server.close() };
}

export interface Hook {
    studio: {
        ready: boolean;
        error: string;
        backend: string;
        mode: 'design' | 'window';
        frames: number;
        fps: number;
        shoeScale: number;
        selected: string;
        hud: boolean;
        design: { colors: Record<string, string>; finishes: Record<string, string> };
        seen: { eye: { x: number; y: number; z: number }; width: number; height: number; source: string };
        eyeOverride: { x: number; y: number; z: number } | null;
        setColor(part: string, hex: string): void;
    };
    head: {
        status: string;
        note: string;
        delegate: string;
        fps: number;
        settings: { diagonal: number; aspect: number; hfov: number; depth: number };
        eye: { x: number; y: number; z: number };
        inject(landmarks: unknown[] | null, size?: { width: number; height: number }): void;
        choose(change: Record<string, number>): void;
    };
}

export interface Tab {
    page: Page;
    errors: string[];
}

const contexts: BrowserContext[] = [];

/** What a browser without a GPU says about itself: the renderer's own notice that it is using WebGL 2, Chrome's note that there is no WebGPU adapter, and software GL's performance remarks. */
const BENIGN = /WebGPU is not available, running under WebGL2|No available adapters|GL Driver Message/;

/** A browser context that is closed by `closeContexts`: every open page holds a graphics device, and a long run of tests runs out of them. */
export async function newContext(browser: Browser): Promise<BrowserContext> {
    const context = await browser.createBrowserContext();
    contexts.push(context);
    return context;
}

export async function closeContexts(): Promise<void> {
    await Promise.all(contexts.splice(0).map((c) => c.close().catch(() => {})));
}

/** Open the configurator in its own browser context. Waits until the shoe has been drawn. */
export async function openStudio(browser: Browser, base: string, options: { query?: string; width?: number; height?: number; waitReady?: boolean; storage?: Record<string, string> } = {}): Promise<Tab> {
    const context = await newContext(browser);
    const page = await context.newPage();
    await page.setViewport({ width: options.width ?? 1280, height: options.height ?? 800, deviceScaleFactor: 1 });
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    // Where there is no GPU the renderer says so, once, and carries on in WebGL 2: that is the designed fallback, not a fault.
    page.on('console', (m) => (m.type() === 'error' || m.type() === 'warn') && !BENIGN.test(m.text()) && errors.push(`${m.type()}: ${m.text()}`));
    if (options.storage) {
        await page.evaluateOnNewDocument((entries: Record<string, string>) => {
            for (const [k, v] of Object.entries(entries)) if (!localStorage.getItem(k)) localStorage.setItem(k, v);
        }, options.storage);
    }
    await page.goto(`${base}/?e2e=1${options.query ?? ''}`);
    await page.waitForFunction(() => (window as unknown as { __app?: unknown }).__app !== undefined);
    if (options.waitReady !== false) await page.waitForFunction(() => (window as unknown as { __app: Hook }).__app.studio.ready, { timeout: 90_000 });
    return { page, errors };
}

export const read = <T = unknown>(page: Page, path: string): Promise<T> => page.evaluate(`window.__app.${path}`) as Promise<T>;

export interface Pixels {
    data: Uint8Array;
    width: number;
    height: number;
}

/** The stage (the 3D view and what is drawn over it) as pixels. */
export async function stagePixels(page: Page): Promise<Pixels> {
    const stage = await page.$('.stage');
    const png = await stage!.screenshot();
    const { data, info } = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height };
}

/** Run something that changes the picture and wait until the renderer has drawn it (the design view draws one frame for each change). */
export async function after(page: Page, action: () => Promise<unknown>): Promise<void> {
    const start = await read<number>(page, 'studio.frames');
    await action();
    await page.waitForFunction((n: number) => (window as unknown as { __app: Hook }).__app.studio.frames > n, { timeout: 20_000 }, start);
}

/** [hue 0-360, saturation 0-1, lightness 0-1] of one pixel. */
export function hsl(r: number, g: number, b: number): [number, number, number] {
    const R = r / 255;
    const G = g / 255;
    const B = b / 255;
    const max = Math.max(R, G, B);
    const min = Math.min(R, G, B);
    const l = (max + min) / 2;
    if (max === min) return [0, 0, l];
    const d = max - min;
    const sat = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    const h = max === R ? (G - B) / d + (G < B ? 6 : 0) : max === G ? (B - R) / d + 2 : (R - G) / d + 4;
    return [h * 60, sat, l];
}

/** How many pixels are of this colour: a hue within `tolerance` degrees, clearly coloured, neither black nor white. */
export function countHue({ data }: Pixels, hue: number, tolerance = 22): number {
    let n = 0;
    for (let i = 0; i < data.length; i += 3) {
        const [h, s, l] = hsl(data[i]!, data[i + 1]!, data[i + 2]!);
        const d = Math.min(Math.abs(h - hue), 360 - Math.abs(h - hue));
        if (s > 0.4 && l > 0.12 && l < 0.92 && d <= tolerance) n++;
    }
    return n;
}

/** The fraction of pixels that differ between two pictures of the same size by more than `by` in any channel. */
export function differ(a: Pixels, b: Pixels, by = 12): number {
    let n = 0;
    for (let i = 0; i < a.data.length; i += 3) if (Math.abs(a.data[i]! - b.data[i]!) > by || Math.abs(a.data[i + 1]! - b.data[i + 1]!) > by || Math.abs(a.data[i + 2]! - b.data[i + 2]!) > by) n++;
    return n / (a.data.length / 3);
}

/** Centre (x, y) of the pixels of one hue, or null. */
export function centroidOfHue(p: Pixels, hue: number, tolerance = 22): { x: number; y: number; n: number } | null {
    let n = 0;
    let sx = 0;
    let sy = 0;
    for (let y = 0; y < p.height; y++) {
        for (let x = 0; x < p.width; x++) {
            const i = (y * p.width + x) * 3;
            const [h, s, l] = hsl(p.data[i]!, p.data[i + 1]!, p.data[i + 2]!);
            const d = Math.min(Math.abs(h - hue), 360 - Math.abs(h - hue));
            if (s > 0.4 && l > 0.12 && l < 0.92 && d <= tolerance) {
                n++;
                sx += x;
                sy += y;
            }
        }
    }
    return n ? { x: sx / n, y: sy / n, n } : null;
}

/** The left and right ends of the run of pixels near the colour at column `from` along row `y`, bridging gaps of up to `gap` pixels (a grid line). */
export function runAround(p: Pixels, y: number, from: number, tolerance = 14, gap = 0): { left: number; right: number } {
    const at = (x: number) => [p.data[(y * p.width + x) * 3]!, p.data[(y * p.width + x) * 3 + 1]!, p.data[(y * p.width + x) * 3 + 2]!];
    const like = at(from);
    const near = (x: number) => at(x).every((v, k) => Math.abs(v - like[k]!) <= tolerance);
    const extend = (step: -1 | 1) => {
        let end = from;
        let misses = 0;
        for (let x = from + step; x >= 0 && x < p.width && misses <= gap; x += step) {
            if (near(x)) {
                end = x;
                misses = 0;
            } else misses++;
        }
        return end;
    };
    return { left: extend(-1), right: extend(1) };
}

/** The pixel at (x, y). */
export function pixelAt(p: Pixels, x: number, y: number): [number, number, number] {
    const i = (y * p.width + x) * 3;
    return [p.data[i]!, p.data[i + 1]!, p.data[i + 2]!];
}

/** The widest run of pixels near the colour `like` anywhere along row `y`, bridging gaps of up to `gap` pixels (a grid line). */
export function widestRunOf(p: Pixels, y: number, like: number[], tolerance = 16, gap = 8): { left: number; right: number } {
    const near = (x: number) => pixelAt(p, x, y).every((v, k) => Math.abs(v - like[k]!) <= tolerance);
    let best = { left: 0, right: -1 };
    let start = -1;
    let last = -1;
    let misses = 0;
    const close = () => {
        if (start >= 0 && last - start > best.right - best.left) best = { left: start, right: last };
        start = -1;
        misses = 0;
    };
    for (let x = 0; x < p.width; x++) {
        if (near(x)) {
            if (start < 0) start = x;
            last = x;
            misses = 0;
        } else if (start >= 0 && ++misses > gap) close();
    }
    close();
    return best;
}

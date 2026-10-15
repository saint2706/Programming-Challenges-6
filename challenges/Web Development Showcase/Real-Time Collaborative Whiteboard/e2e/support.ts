import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Browser, Page } from 'puppeteer-core';
import { type RunningServer, startServer } from '../server/app.ts';

const DIST = fileURLToPath(new URL('../dist', import.meta.url));

export interface Harness {
    server: RunningServer;
    dataDir: string;
    /** Stop the server but keep its port and data, as in a crash. */
    down(): Promise<void>;
    /** Start it again on the same port and data directory. */
    up(): Promise<void>;
    stop(): Promise<void>;
}

/** Serve the built app on a fixed free port so a restart keeps the same URL. */
export async function startHarness(): Promise<Harness> {
    const dataDir = mkdtempSync(join(tmpdir(), 'wb-e2e-'));
    const options = { dataDir, staticDir: DIST, rooms: { saveDelayMs: 50 } };
    let server = await startServer({ ...options, port: 0 });
    const port = server.port;
    return {
        get server() {
            return server;
        },
        dataDir,
        async down() {
            await server.close();
        },
        async up() {
            server = await startServer({ ...options, port });
        },
        async stop() {
            await server.close().catch(() => {});
            rmSync(dataDir, { recursive: true, force: true });
        },
    };
}

const pageErrors = new WeakMap<Page, string[]>();

/** Uncaught errors and console errors seen by a tab so far. */
export const errorsOf = (page: Page): string[] => pageErrors.get(page) ?? [];

/** Open the board as a separate person: its own browser context, so its own storage and identity. */
export async function openTab(browser: Browser, base: string, room: string, options: { name?: string; width?: number; height?: number; query?: string; touch?: boolean } = {}): Promise<Page> {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ width: options.width ?? 1100, height: options.height ?? 720, deviceScaleFactor: 1, hasTouch: options.touch ?? false });
    const errors: string[] = [];
    pageErrors.set(page, errors);
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    if (options.name) {
        await page.evaluateOnNewDocument((name: string) => localStorage.setItem('wb:name', name), options.name);
    }
    await page.goto(`${base}/b/${room}?e2e=1${options.query ?? ''}`);
    await page.waitForFunction(() => (window as unknown as { __wb?: { conn: { status: string } } }).__wb?.conn.status === 'online');
    return page;
}

export async function pickTool(page: Page, label: string): Promise<void> {
    await page.click(`button[aria-label="${label}"]`);
}

export async function pickColor(page: Page, hex: string): Promise<void> {
    await page.$eval(`input[name="color"][value="${hex}"]`, (el) => (el as HTMLInputElement).click());
}

export async function stageOrigin(page: Page): Promise<{ x: number; y: number; w: number; h: number }> {
    return page.$eval('#stage', (el) => {
        const r = el.getBoundingClientRect();
        return { x: r.left, y: r.top, w: r.width, h: r.height };
    });
}

/** Draw with the real mouse: points are in stage coordinates (CSS pixels from the stage's top-left). */
export async function drag(page: Page, points: [number, number][], steps = 4): Promise<void> {
    const o = await stageOrigin(page);
    const [first, ...rest] = points;
    if (!first) return;
    await page.mouse.move(o.x + first[0], o.y + first[1]);
    await page.mouse.down();
    for (const [x, y] of rest) await page.mouse.move(o.x + x, o.y + y, { steps });
    await page.mouse.up();
}

export async function click(page: Page, x: number, y: number): Promise<void> {
    const o = await stageOrigin(page);
    await page.mouse.click(o.x + x, o.y + y);
}

type Hook = {
    board: { camera: { x: number; y: number; zoom: number } };
    doc: {
        visible(): { item: { id: string; t: string; author: string; color: string; text?: string; size: number; k?: string; pts?: number[] }; dx: number; dy: number }[];
        pendingCount: number;
        seq: number;
        users: Map<string, { name: string }>;
        live: Map<string, unknown>;
        cursors: Map<string, unknown>;
        boardVersion: number;
        canUndo: boolean;
        canRedo: boolean;
    };
    conn: { status: string; dropForTest(): void };
};

export function summary(page: Page) {
    return page.evaluate(() => {
        const wb = (window as unknown as { __wb: Hook }).__wb;
        return {
            items: wb.doc.visible().map((v) => ({ id: v.item.id, t: v.item.t, color: v.item.color, k: v.item.k, text: v.item.text, dx: v.dx, dy: v.dy })),
            pending: wb.doc.pendingCount,
            seq: wb.doc.seq,
            users: [...wb.doc.users.values()].map((u) => u.name),
            canUndo: wb.doc.canUndo,
            canRedo: wb.doc.canRedo,
            status: wb.conn.status,
        };
    });
}

/** Wait until `fn` (run in the page) returns truthy. */
export function until<T>(page: Page, fn: (wb: Hook) => T, timeout = 10_000): Promise<unknown> {
    return page.waitForFunction(`(${fn.toString()})(window.__wb)`, { timeout });
}

/** The colour of one pixel of the committed-items canvas, at stage coordinates. */
export function pixel(page: Page, x: number, y: number): Promise<[number, number, number, number]> {
    return page.evaluate(
        (px, py) => {
            const c = document.getElementById('base') as HTMLCanvasElement;
            const dpr = c.width / c.clientWidth;
            const d = c.getContext('2d')!.getImageData(Math.round(px * dpr), Math.round(py * dpr), 1, 1).data;
            return [d[0]!, d[1]!, d[2]!, d[3]!] as [number, number, number, number];
        },
        x,
        y,
    );
}

/** Is there any non-paper ink in a small box around the point? Robust to antialiasing and line width. */
export function inkNear(page: Page, x: number, y: number, radius = 4): Promise<boolean> {
    return page.evaluate(
        (px, py, r) => {
            const c = document.getElementById('base') as HTMLCanvasElement;
            const dpr = c.width / c.clientWidth;
            const size = Math.round(r * 2 * dpr) + 1;
            const d = c.getContext('2d')!.getImageData(Math.round((px - r) * dpr), Math.round((py - r) * dpr), size, size).data;
            for (let i = 0; i < d.length; i += 4) {
                // The paper is white and the grid dots are light grey; ink is clearly darker or coloured.
                if (d[i]! < 150 || d[i + 1]! < 150 || d[i + 2]! < 150) return true;
            }
            return false;
        },
        x,
        y,
        radius,
    );
}

/**
 * Poll until the ink state near a point matches `want` (the canvas repaints on the next animation
 * frame after the data changes, so a one-shot read right after a sync would race it). Returns the
 * final state, so a mismatch fails the caller's assertion with a clear true/false.
 */
export async function inkIs(page: Page, x: number, y: number, want: boolean, timeout = 3000): Promise<boolean> {
    const deadline = Date.now() + timeout;
    let state = await inkNear(page, x, y);
    while (state !== want && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 40));
        state = await inkNear(page, x, y);
    }
    return state;
}

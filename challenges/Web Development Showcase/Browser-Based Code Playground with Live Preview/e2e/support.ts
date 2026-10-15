import { fileURLToPath } from 'node:url';
import type { Browser, Frame, Page } from 'puppeteer-core';
import { preview } from 'vite';

const ROOT = fileURLToPath(new URL('..', import.meta.url));

/** Serve the built app (`dist/`) on a free local port. Run `vite build` first. */
export async function startServer(): Promise<{ url: string; close(): Promise<void> }> {
    const server = await preview({ root: ROOT, logLevel: 'silent', preview: { host: '127.0.0.1', port: 4190, strictPort: false } });
    const url = server.resolvedUrls!.local[0]!.replace(/\/$/, '');
    return { url, close: () => server.close() };
}

export interface PlaygroundHook {
    project: { files: { name: string; code: string }[]; network: boolean };
    active: string;
    fromLink: boolean;
    editor: {
        view: { state: { doc: { length: number; toString(): string } }; dispatch(spec: unknown): void };
        active: string;
        text(name: string): string | undefined;
    };
    previewHandle: { live: boolean; currentRun: number };
    run(force: boolean): Promise<void>;
}

export interface Tab {
    page: Page;
    errors: string[];
}

const tabErrors = new WeakMap<Page, string[]>();
export const errorsOf = (page: Page): string[] => tabErrors.get(page) ?? [];

/**
 * Open the playground as a separate person (its own browser context, so its own storage). Waits until
 * it has started, and until the first preview is live unless `waitForPreview` is false.
 */
export async function openPlayground(browser: Browser, base: string, options: { hash?: string; width?: number; height?: number; waitForPreview?: boolean; storage?: Record<string, string>; blockStorage?: boolean } = {}): Promise<Page> {
    const context = await browser.createBrowserContext();
    const page = await context.newPage();
    await page.setViewport({ width: options.width ?? 1280, height: options.height ?? 800, deviceScaleFactor: 1 });
    const errors: string[] = [];
    tabErrors.set(page, errors);
    page.on('pageerror', (e) => errors.push(String(e)));
    page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
    if (options.blockStorage) {
        await page.evaluateOnNewDocument(() => {
            if (window !== window.top) return; // the sandboxed preview frame has its own rules
            Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('blocked', 'SecurityError'); } });
        });
    }
    if (options.storage) {
        await page.evaluateOnNewDocument((entries: Record<string, string>) => {
            if (window !== window.top) return; // the sandboxed preview frame cannot (and must not) touch storage
            for (const [k, v] of Object.entries(entries)) if (!localStorage.getItem(k)) localStorage.setItem(k, v);
        }, options.storage);
    }
    await page.goto(`${base}/?e2e=1${options.hash ?? ''}`);
    await page.waitForFunction(() => (window as unknown as { __pg?: unknown }).__pg !== undefined);
    if (options.waitForPreview !== false) await page.waitForFunction(() => (window as unknown as { __pg: PlaygroundHook }).__pg.previewHandle.live, { timeout: 60_000 });
    return page;
}

/** The preview's current page as a Puppeteer frame (opaque origin, so only CDP can look inside). */
export async function previewFrame(page: Page): Promise<Frame> {
    const handle = await page.waitForSelector('#preview-host iframe:not(.pending)', { timeout: 20_000 });
    const frame = await handle!.contentFrame();
    if (!frame) throw new Error('preview frame has no content');
    return frame;
}

/** Replace a file's text the way an edit does: through the editor, so every hook fires. */
export async function setCode(page: Page, name: string, code: string): Promise<void> {
    await page.evaluate(
        (n: string, c: string) => {
            const pg = (window as unknown as { __pg: PlaygroundHook }).__pg;
            const tab = document.querySelector<HTMLElement>(`[role=tab][data-name="${CSS.escape(n)}"]`);
            if (pg.editor.active !== n) tab?.click();
            const doc = pg.editor.view.state.doc;
            pg.editor.view.dispatch({ changes: { from: 0, to: doc.length, insert: c } });
        },
        name,
        code,
    );
}

export function consoleEntries(page: Page): Promise<{ level: string; text: string; where: string }[]> {
    return page.$$eval('#console-log .entry', (rows) =>
        rows.map((r) => ({
            level: (r as HTMLElement).dataset.level ?? '',
            text: r.querySelector('.msg')?.textContent ?? '',
            where: r.querySelector('.where')?.textContent ?? '',
        })),
    );
}

export async function waitForConsole(page: Page, predicate: (entries: { level: string; text: string; where: string }[]) => boolean, timeout = 10_000): Promise<{ level: string; text: string; where: string }[]> {
    const deadline = Date.now() + timeout;
    for (;;) {
        const entries = await consoleEntries(page);
        if (predicate(entries)) return entries;
        if (Date.now() > deadline) throw new Error(`console never matched; has: ${JSON.stringify(entries)}`);
        await new Promise((r) => setTimeout(r, 50));
    }
}

/** A new run has started and is live. */
export function runNumber(page: Page): Promise<number> {
    return page.evaluate(() => (window as unknown as { __pg: PlaygroundHook }).__pg.previewHandle.currentRun);
}

export async function waitForNewRun(page: Page, after: number, timeout = 20_000): Promise<number> {
    await page.waitForFunction((n: number) => {
        const p = (window as unknown as { __pg: PlaygroundHook }).__pg.previewHandle;
        return p.live && p.currentRun > n;
    }, { timeout }, after);
    return runNumber(page);
}

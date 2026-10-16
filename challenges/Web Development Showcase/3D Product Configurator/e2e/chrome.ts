import { existsSync } from 'node:fs';
import puppeteer, { type Browser } from 'puppeteer-core';

/** Where a real Chrome or Edge usually lives. `CHROME_PATH` wins when set. */
const CANDIDATES = [
    process.env.CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
];

export function findChrome(): string | null {
    return CANDIDATES.find((path) => path && existsSync(path)) ?? null;
}

/** Launch the system browser (no download). `--font-render-hinting=none` keeps text metrics stable. */
export function launchChrome(extraArgs: string[] = []): Promise<Browser> {
    const executablePath = findChrome();
    if (!executablePath) throw new Error('No Chrome or Edge found. Set CHROME_PATH to a browser executable.');
    return puppeteer.launch({
        executablePath,
        headless: true,
        args: [
            '--font-render-hinting=none',
            '--disable-lcd-text',
            '--no-first-run',
            '--no-default-browser-check',
            // Without a GPU (CI) WebGL runs in software, which Chrome now has to be told to allow.
            '--enable-unsafe-swiftshader',
            // A CI runner has no GPU: say so, rather than let Chrome look for one (WebGPU on a software adapter is very slow).
            ...(process.env.CI ? ['--disable-gpu'] : []),
            ...(process.env.E2E_CHROME_ARGS?.split(' ').filter(Boolean) ?? []),
            ...extraArgs,
        ],
    });
}

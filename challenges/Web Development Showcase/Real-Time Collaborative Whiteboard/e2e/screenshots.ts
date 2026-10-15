/** `pnpm run screenshots`: rebuilds the README images (docs/*.png) from a scripted two-person session. */
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Page } from 'puppeteer-core';
import { launchChrome } from './chrome.ts';
import { click, drag, openTab, pickColor, pickTool, startHarness, until } from './support.ts';

const out = fileURLToPath(new URL('../docs/', import.meta.url));
mkdirSync(out, { recursive: true });

const h = await startHarness();
const browser = await launchChrome();

async function note(page: Page, tool: 'Sticky note' | 'Text', color: string | null, x: number, y: number, text: string) {
    await pickTool(page, tool);
    if (color) await pickColor(page, color);
    await click(page, x, y);
    await page.keyboard.type(text);
    await page.keyboard.down('Control');
    await page.keyboard.press('Enter');
    await page.keyboard.up('Control');
}

try {
    const room = 'sprint-retro';
    const ada = await openTab(browser, h.server.url, room, { name: 'Ada' });
    await ada.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    const grace = await openTab(browser, h.server.url, room, { name: 'Grace' });

    await pickTool(ada, 'Highlighter');
    await drag(ada, [[84, 112], [330, 116]], 8);
    // Title and column headings (the text size buttons are radios, so set via the board).
    await ada.evaluate(() => (window as unknown as { __wb: { board: { setSize(n: number): void } } }).__wb.board.setSize(2));
    await note(ada, 'Text', null, 80, 60, 'Sprint retro');
    await ada.evaluate(() => (window as unknown as { __wb: { board: { setSize(n: number): void } } }).__wb.board.setSize(0));
    await note(ada, 'Text', null, 84, 150, 'Went well');
    await note(ada, 'Text', null, 424, 150, 'Could be better');
    await note(ada, 'Text', null, 764, 150, 'Ideas');
    await ada.evaluate(() => (window as unknown as { __wb: { board: { setSize(n: number): void } } }).__wb.board.setSize(1));
    await note(ada, 'Sticky note', '#c9f0d2', 80, 190, 'Shipped on Fridays without drama');
    await note(ada, 'Sticky note', '#fff3a3', 80, 410, 'Pairing sessions');
    await note(ada, 'Sticky note', '#ffd0d6', 420, 190, 'Flaky CI');
    await note(ada, 'Sticky note', '#ffd0d6', 420, 410, 'Too many meetings');
    await note(ada, 'Sticky note', '#c7e3ff', 760, 190, 'Try a demo day');

    await pickTool(ada, 'Pen');
    await pickColor(ada, '#d1242f');
    await drag(ada, [[404, 200], [408, 180], [440, 176], [660, 178], [690, 190], [694, 300], [690, 390], [660, 398], [440, 396], [410, 392], [402, 300], [404, 204]], 5);
    await pickTool(ada, 'Arrow');
    await pickColor(ada, '#0969da');
    await drag(ada, [[690, 270], [750, 270]]);

    await pickTool(grace, 'Pen');
    await pickColor(grace, '#1a7f37');
    await drag(grace, [[780, 480], [820, 440], [860, 500], [900, 440], [940, 500], [980, 450]], 6);
    await grace.mouse.move(1010, 540, { steps: 8 });
    await until(ada, (wb) => wb.doc.visible().length >= 13 && wb.doc.cursors.size === 1);
    await new Promise((r) => setTimeout(r, 900));
    await ada.screenshot({ path: `${out}board-light.png` });

    const dark = await openTab(browser, h.server.url, room, { name: 'Linus' });
    await dark.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
    await until(dark, (wb) => wb.doc.visible().length >= 13);
    await new Promise((r) => setTimeout(r, 900));
    await dark.screenshot({ path: `${out}board-dark.png` });

    const phone = await openTab(browser, h.server.url, room, { name: 'Margaret', width: 360, height: 740, touch: true });
    await phone.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }]);
    await until(phone, (wb) => wb.doc.visible().length >= 13);
    await phone.evaluate(() => (window as unknown as { __wb: { board: { fitContent(): void } } }).__wb.board.fitContent());
    await new Promise((r) => setTimeout(r, 900));
    await phone.screenshot({ path: `${out}board-phone-360.png` });
    console.log(`wrote 3 images to ${out}`);
} finally {
    await browser.close();
    await h.stop();
}

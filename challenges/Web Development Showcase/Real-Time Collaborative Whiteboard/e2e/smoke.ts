import { launchChrome } from './chrome.ts';
import { click, drag, openTab, pickColor, pickTool, startHarness } from './support.ts';

const out = process.argv[2]!;
const h = await startHarness();
const browser = await launchChrome();
try {
    const a = await openTab(browser, h.server.url, 'smoke', { name: 'Ada' });
    const b = await openTab(browser, h.server.url, 'smoke', { name: 'Grace' });
    await pickTool(a, 'Pen');
    await drag(a, [[200, 150], [260, 120], [330, 160], [400, 130], [470, 170]], 6);
    await pickColor(a, '#d1242f');
    await pickTool(a, 'Rectangle');
    await drag(a, [[220, 260], [420, 380]]);
    await pickColor(a, '#0969da');
    await pickTool(a, 'Ellipse');
    await drag(a, [[500, 250], [680, 360]]);
    await pickTool(a, 'Arrow');
    await drag(a, [[430, 320], [490, 300]]);
    await pickTool(a, 'Sticky note');
    await click(a, 600, 120);
    await a.keyboard.type('Hello from Ada');
    await a.keyboard.down('Control'); await a.keyboard.press('Enter'); await a.keyboard.up('Control');
    await pickTool(b, 'Highlighter');
    await drag(b, [[220, 450], [520, 450]]);
    await b.mouse.move(700, 500);
    await new Promise((r) => setTimeout(r, 1200));
    const m = await openTab(browser, h.server.url, 'smoke', { name: 'Linus', width: 360, height: 740 });
    await new Promise((r) => setTimeout(r, 500));
    await m.screenshot({ path: `${out}/m.png` });
    await a.screenshot({ path: `${out}/a.png` });
    await b.screenshot({ path: `${out}/b.png` });
    console.log('done');
} finally {
    await browser.close();
    await h.stop();
}

import { launchChrome } from './chrome.ts';
import { openPlayground, startServer } from './support.ts';

const out = process.argv[2]!;
const server = await startServer();
const browser = await launchChrome();
try {
    const page = await openPlayground(browser, server.url);
    await new Promise((r) => setTimeout(r, 800));
    await page.screenshot({ path: `${out}/pg.png` });
    const phone = await openPlayground(browser, server.url, { width: 390, height: 800 });
    await new Promise((r) => setTimeout(r, 500));
    await phone.screenshot({ path: `${out}/pg-phone.png` });
    console.log('ok');
} finally {
    await browser.close();
    await server.close();
}

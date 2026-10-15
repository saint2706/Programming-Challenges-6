/**
 * Numbers for the README: `pnpm run bench`. Three parts: the room's CPU cost per op, the round trip
 * over a real loopback WebSocket, and drawing speed in a real Chrome. Results depend on the machine.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import WebSocket from 'ws';
import { findChrome, launchChrome } from '../e2e/chrome.ts';
import { openTab } from '../e2e/support.ts';
import { startServer } from '../server/app.ts';
import { Room } from '../server/room.ts';
import type { ClientMsg, Item, ServerMsg } from '../shared/types.ts';

const pct = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor((p / 100) * xs.length))]!;
const fmt = (n: number, d = 2) => n.toFixed(d);

function strokeOf(id: string, points: number): Item {
    const pts: number[] = [];
    for (let i = 0; i < points; i++) pts.push(Math.round((i * 3 + Math.sin(i / 5) * 40) * 10) / 10, Math.round(Math.cos(i / 7) * 60 * 10) / 10);
    return { t: 'stroke', id, author: 'x', color: '#1f2328', size: 4, pts };
}

console.log('## Message sizes (JSON text frames)\n');
console.log('| message | bytes |\n| --- | ---: |');
for (const points of [20, 100, 400]) {
    const msg: ClientMsg = { t: 'op', opId: 'a'.repeat(32), op: { k: 'add', item: strokeOf('b'.repeat(32), points) } };
    console.log(`| ${points}-point stroke | ${JSON.stringify(msg).length} |`);
}
console.log(`| cursor move | ${JSON.stringify({ t: 'cur', x: 1234.5, y: -678.9 }).length} |`);

console.log('\n## Room CPU: one 100-point stroke committed and broadcast\n');
console.log('| people in room | stringify per recipient | stringify once (shipped) | ops/s (shipped) |\n| ---: | ---: | ---: | ---: |');
for (const people of [2, 10, 50]) {
    const run = (shared: boolean) => {
        const clock = { t: 0 };
        const room = new Room('bench', { now: () => (clock.t += 100) });
        let bytes = 0;
        const conns = Array.from({ length: people }, () => ({
            send: (msg: ServerMsg, json?: string) => void (bytes += (shared ? (json ?? JSON.stringify(msg)) : JSON.stringify(msg)).length),
            close: () => {},
        }));
        conns.forEach((c, i) => room.join(c, { t: 'hello', v: 1, id: `u${i}`, name: `u${i}`, last: null }));
        const ops = 4000;
        const prepared = Array.from({ length: ops }, (_, i): ClientMsg => ({ t: 'op', opId: `o${i}`, op: { k: 'add', item: strokeOf(`s${i}`, 100) } }));
        const start = performance.now();
        for (const m of prepared) room.receive('u0', conns[0]!, m);
        const ms = performance.now() - start;
        return { perOp: (ms / ops) * 1000, opsPerSecond: ops / (ms / 1000), bytes };
    };
    // "per recipient" makes the shared string unusable by always ignoring it.
    const a = run(false);
    const b = run(true);
    console.log(`| ${people} | ${fmt(a.perOp, 1)} µs/op | ${fmt(b.perOp, 1)} µs/op | ${Math.round(b.opsPerSecond).toLocaleString('en-US')} |`);
}

console.log('\n## Loopback WebSocket round trip (sender → server → another client)\n');
{
    const dir = mkdtempSync(join(tmpdir(), 'wb-bench-'));
    const server = await startServer({ dataDir: dir, port: 0 });
    const open = (id: string) =>
        new Promise<WebSocket>((resolve) => {
            const ws = new WebSocket(`${server.url.replace('http', 'ws')}/ws/bench`);
            ws.on('open', () => {
                ws.send(JSON.stringify({ t: 'hello', v: 1, id, name: id, last: null } satisfies ClientMsg));
                ws.once('message', () => resolve(ws)); // the welcome
            });
        });
    const a = await open('a');
    const b = await open('b');
    const samples: number[] = [];
    let pending: ((t: number) => void) | null = null;
    b.on('message', (data) => {
        if ((JSON.parse(String(data)) as ServerMsg).t === 'op') pending?.(performance.now());
    });
    for (let i = 0; i < 600; i++) {
        const t0 = performance.now();
        const arrived = new Promise<number>((r) => (pending = r));
        a.send(JSON.stringify({ t: 'op', opId: `o${i}`, op: { k: 'add', item: strokeOf(`s${i}`, 100) } } satisfies ClientMsg));
        samples.push((await arrived) - t0);
        if (i % 14 === 13) await new Promise((r) => setTimeout(r, 1000)); // stay under the per-connection rate limit
    }
    const warm = samples.slice(50);
    console.log(`${warm.length} samples, 100-point stroke: median ${fmt(pct(warm, 50))} ms, p95 ${fmt(pct(warm, 95))} ms, p99 ${fmt(pct(warm, 99))} ms`);
    a.close();
    b.close();
    await server.close();
    rmSync(dir, { recursive: true, force: true });
}

console.log('\n## Drawing speed in Chrome: panning across a board of N strokes (80 points each)\n');
if (!findChrome()) {
    console.log('(no Chrome found, skipped)');
} else {
    const dir = mkdtempSync(join(tmpdir(), 'wb-bench-'));
    const server = await startServer({ dataDir: dir, port: 0, staticDir: fileURLToPath(new URL('../dist', import.meta.url)) });
    const browser = await launchChrome();
    try {
        console.log('| strokes | in view at 100% | paint CPU (median) | frame interval median / p95 |\n| ---: | ---: | ---: | ---: |');
        for (const count of [1000, 5000, 20_000]) {
            const page = await openTab(browser, server.url, `bench-${count}`);
            const result = await page.evaluate(async (n: number) => {
                const wb = (window as unknown as { __wb: { board: any; doc: any } }).__wb;
                const side = Math.ceil(Math.sqrt(n));
                for (let i = 0; i < n; i++) {
                    const ox = (i % side) * 120;
                    const oy = Math.floor(i / side) * 120;
                    const pts: number[] = [];
                    for (let k = 0; k < 80; k++) pts.push(ox + k * 1.2, oy + Math.sin(k / 6) * 30 + 50);
                    wb.doc.confirmed.items.set(`s${i}`, { item: { t: 'stroke', id: `s${i}`, author: 'x', color: '#1f2328', size: 4, pts }, deleted: false, dx: 0, dy: 0 });
                }
                wb.doc.boardVersion++;
                wb.board.camera.x = side * 60;
                wb.board.camera.y = side * 60;
                wb.board.camera.zoom = 1;
                const view = wb.board.camera.view(1100, 668);
                let inView = 0;
                wb.doc.forEachVisible(({ item }: { item: { pts: number[] } }) => {
                    if (item.pts[0]! >= view.minX - 100 && item.pts[0]! <= view.maxX && item.pts[1]! >= view.minY - 100 && item.pts[1]! <= view.maxY) inView++;
                });
                const paintTimes: number[] = [];
                const intervals: number[] = [];
                await new Promise<void>((resolve) => {
                    let frames = 0;
                    let last = performance.now();
                    const step = (now: number) => {
                        intervals.push(now - last);
                        last = now;
                        wb.board.camera.x += 4;
                        wb.board.baseDirty = true;
                        const t0 = performance.now();
                        wb.board.paint();
                        paintTimes.push(performance.now() - t0);
                        if (++frames < 150) requestAnimationFrame(step);
                        else resolve();
                    };
                    requestAnimationFrame(step);
                });
                const med = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]!;
                const p95 = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length * 0.95)]!;
                return { inView, paint: med(paintTimes.slice(10)), mi: med(intervals.slice(10)), pi: p95(intervals.slice(10)) };
            }, count);
            console.log(`| ${count.toLocaleString('en-US')} | ${result.inView.toLocaleString('en-US')} | ${fmt(result.paint)} ms | ${fmt(result.mi, 1)} / ${fmt(result.pi, 1)} ms |`);
            await page.close();
        }
    } finally {
        await browser.close();
        await server.close();
        rmSync(dir, { recursive: true, force: true });
    }
}

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, test } from 'vitest';
import WebSocket from 'ws';
import { publicId, startServer, type RunningServer } from '../server/app.ts';
import type { ClientMsg, ServerMsg } from '../shared/types.ts';
import { stroke } from './helpers.ts';

let dir: string;
let server: RunningServer;
const sockets: WebSocket[] = [];

beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), 'wb-int-'));
    mkdirSync(join(dir, 'site', 'assets'), { recursive: true });
    writeFileSync(join(dir, 'site', 'index.html'), '<!doctype html><title>board</title>');
    writeFileSync(join(dir, 'site', 'assets', 'app.js'), 'console.log(1)');
    server = await startServer({
        dataDir: join(dir, 'rooms'),
        staticDir: join(dir, 'site'),
        helloTimeoutMs: 150,
        rooms: { saveDelayMs: 20 },
    });
});
afterEach(async () => {
    for (const s of sockets.splice(0)) s.terminate();
    await server.close();
    rmSync(dir, { recursive: true, force: true });
});

const wsUrl = (room: string, s = server) => `${s.url.replace('http', 'ws')}/ws/${room}`;

/** A client that records every message and lets a test wait for one. */
class Client {
    readonly got: ServerMsg[] = [];
    readonly ws: WebSocket;
    closed: Promise<{ code: number; reason: string }>;
    private waiters: (() => void)[] = [];

    constructor(url: string, options?: WebSocket.ClientOptions) {
        this.ws = new WebSocket(url, options);
        sockets.push(this.ws);
        this.ws.on('message', (data) => {
            this.got.push(JSON.parse(String(data)));
            for (const w of this.waiters.splice(0)) w();
        });
        this.closed = new Promise((resolve) =>
            this.ws.on('close', (code, reason) => resolve({ code, reason: String(reason) })),
        );
    }

    opened(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.ws.once('open', () => resolve());
            this.ws.once('error', reject);
        });
    }

    send(msg: ClientMsg | string | Buffer): void {
        this.ws.send(typeof msg === 'object' && !Buffer.isBuffer(msg) ? JSON.stringify(msg) : msg);
    }

    async next<T extends ServerMsg['t']>(t: T): Promise<Extract<ServerMsg, { t: T }>> {
        for (;;) {
            const found = this.got.find((m) => m.t === t);
            if (found) {
                this.got.splice(this.got.indexOf(found), 1);
                return found as Extract<ServerMsg, { t: T }>;
            }
            await new Promise<void>((resolve) => this.waiters.push(resolve));
        }
    }
}

async function enter(room: string, id: string, last: number | null = null, s = server) {
    const c = new Client(wsUrl(room, s));
    await c.opened();
    c.send({ t: 'hello', v: 1, id, name: id, last });
    const welcome = await c.next('welcome');
    return { c, welcome };
}

const add = (id: string): ClientMsg => ({ t: 'op', opId: `op-${id}`, op: { k: 'add', item: stroke(id) } });

describe('collaboration over a real socket', () => {
    test('what one client draws, the other receives, and the author is the sender', async () => {
        const a = await enter('demo', 'alice');
        const b = await enter('demo', 'bob');
        a.c.send(add('s1'));
        const seen = await b.c.next('op');
        expect(seen).toMatchObject({ seq: 1, by: publicId('alice') });
        expect(seen.op.k === 'add' && seen.op.item.id).toBe('s1');
    });

    test('rooms are isolated from each other', async () => {
        const a = await enter('one', 'alice');
        const other = await enter('two', 'bob');
        a.c.send(add('s1'));
        await a.c.next('op');
        const late = await enter('two', 'carol');
        expect(late.welcome.items).toEqual([]);
        expect(other.welcome.items).toEqual([]);
    });

    test('the board survives a server restart', async () => {
        const a = await enter('keep', 'alice');
        a.c.send(add('s1'));
        await a.c.next('op');
        a.c.ws.close();
        await a.c.closed;
        await server.close();

        server = await startServer({ dataDir: join(dir, 'rooms'), helloTimeoutMs: 150 });
        const back = await enter('keep', 'alice');
        expect(back.welcome.mode).toBe('snapshot');
        expect(back.welcome.items?.map((s) => s.item.id)).toEqual(['s1']);
        expect(back.welcome.seq).toBe(1);
    });

    test('a client that reconnects with its last seq gets just what it missed', async () => {
        const a = await enter('catch', 'alice');
        const b = await enter('catch', 'bob');
        a.c.send(add('s1'));
        await b.c.next('op');
        b.c.ws.close();
        await b.c.closed;
        a.c.send(add('s2'));
        await a.c.next('op');
        const back = await enter('catch', 'bob', 1);
        expect(back.welcome.mode).toBe('catchup');
        expect(back.welcome.ops?.map((e) => e.seq)).toEqual([2]);
    });
});

describe('hostile clients', () => {
    test('a page on another origin cannot open a socket', async () => {
        const c = new Client(wsUrl('demo'), { headers: { Origin: 'http://evil.example' } });
        await expect(c.opened()).rejects.toThrow(/403/);
    });

    test('the same origin and no origin at all (non-browser clients) are fine', async () => {
        const same = new Client(wsUrl('demo'), { headers: { Origin: server.url } });
        await same.opened();
        const none = new Client(wsUrl('demo'));
        await none.opened();
    });

    test('an explicitly allowed origin is let in', async () => {
        const s2 = await startServer({
            dataDir: join(dir, 'rooms2'),
            allowedOrigins: ['https://board.example'],
        });
        try {
            const c = new Client(wsUrl('demo', s2), { headers: { Origin: 'https://board.example' } });
            await c.opened();
        } finally {
            await s2.close();
        }
    });

    test.each(['..%2Fetc%2Fpasswd', 'a%20b', 'x'.repeat(41)])('room id %j is refused', async (room) => {
        const c = new Client(wsUrl(room));
        await expect(c.opened()).rejects.toThrow(/400|404/);
    });

    test('a frame over the size limit closes the connection', async () => {
        const { c } = await enter('demo', 'alice');
        c.send(`{"t":"ping","pad":"${'x'.repeat(300_000)}"}`);
        expect((await c.closed).code).toBe(1009);
    });

    test('binary frames are refused', async () => {
        const { c } = await enter('demo', 'alice');
        c.send(Buffer.from([1, 2, 3]));
        expect((await c.closed).code).toBe(1003);
    });

    test('talking before saying hello gets you dropped', async () => {
        const c = new Client(wsUrl('demo'));
        await c.opened();
        c.send(add('s1'));
        expect((await c.closed).code).toBe(1008);
    });

    test('silence after connecting gets you dropped', async () => {
        const c = new Client(wsUrl('demo'));
        await c.opened();
        expect((await c.closed).reason).toMatch(/hello timeout/);
    });

    test('a stream of garbage gets you dropped after a few strikes', async () => {
        const { c } = await enter('demo', 'alice');
        for (let i = 0; i < 25; i++) c.send('not json');
        expect((await c.closed).code).toBe(1008);
    });

    test('one bad message does not hurt a well-behaved connection', async () => {
        const { c } = await enter('demo', 'alice');
        c.send('not json');
        c.send({ t: 'ping' });
        await c.next('pong');
    });
});

describe('identity', () => {
    test('what everyone else sees is a hash: it cannot be used to take someone over', async () => {
        const victim = await enter('id', 'victim-secret-token');
        const publicOfVictim = victim.welcome.you.id;
        expect(publicOfVictim).toBe(publicId('victim-secret-token'));
        expect(publicOfVictim).not.toContain('victim-secret-token');

        // An attacker who learned the public id sends it as their own id.
        const attacker = await enter('id', publicOfVictim);
        expect(attacker.welcome.you.id).not.toBe(publicOfVictim); // a different person
        victim.c.send({ t: 'ping' });
        await victim.c.next('pong'); // and the victim was not kicked
        expect(attacker.welcome.users.map((u) => u.id)).toContain(publicOfVictim);
    });

    test('the same secret from a new connection replaces the old one (a tab reconnecting)', async () => {
        const first = await enter('id2', 'tab-secret');
        const second = await enter('id2', 'tab-secret');
        const closed = await first.c.closed;
        expect(closed.code).toBe(4001);
        expect(second.welcome.you.id).toBe(first.welcome.you.id);
    });
});

describe('who the server talks to', () => {
    const get = (path: string, host: string) =>
        new Promise<number>((resolve, reject) => {
            const req = request({ host: '127.0.0.1', port: server.port, path, headers: { Host: host } }, (res) => {
                res.resume();
                resolve(res.statusCode ?? 0);
            });
            req.on('error', reject);
            req.end();
        });

    test('a request carrying a foreign Host (DNS rebinding) is refused, localhost names are fine', async () => {
        expect(await get('/healthz', 'evil.example:8787')).toBe(421);
        expect(await get('/', 'rebind.attacker.test')).toBe(421);
        expect(await get('/healthz', 'localhost:8787')).toBe(200);
        expect(await get('/healthz', '127.0.0.1')).toBe(200);
        expect(await get('/healthz', '[::1]:8787')).toBe(200);
    });

    test('a WebSocket handshake whose Host and Origin agree on a foreign name is refused too', async () => {
        const c = new Client(wsUrl('demo'), { headers: { Host: 'evil.example:8787', Origin: 'http://evil.example:8787' } });
        await expect(c.opened()).rejects.toThrow(/421/);
    });

    test('allowedHosts opens the server to a real domain', async () => {
        const s2 = await startServer({ dataDir: join(dir, 'rooms3'), allowedHosts: ['board.example'] });
        try {
            const status = await new Promise<number>((resolve, reject) => {
                const req = request({ host: '127.0.0.1', port: s2.port, path: '/healthz', headers: { Host: 'board.example' } }, (res) => {
                    res.resume();
                    resolve(res.statusCode ?? 0);
                });
                req.on('error', reject);
                req.end();
            });
            expect(status).toBe(200);
        } finally {
            await s2.close();
        }
    });
});

describe('static files', () => {
    test('serves the app with a strict content security policy', async () => {
        const res = await fetch(`${server.url}/`);
        expect(res.status).toBe(200);
        expect(await res.text()).toContain('<title>board</title>');
        const csp = res.headers.get('content-security-policy') ?? '';
        expect(csp).toContain("default-src 'self'");
        expect(csp).toContain("object-src 'none'");
        expect(csp).not.toContain('unsafe-inline');
        expect(res.headers.get('x-content-type-options')).toBe('nosniff');
    });

    test('board links serve the app, assets keep their type, nothing escapes the folder', async () => {
        const board = await fetch(`${server.url}/b/some-room`);
        expect(await board.text()).toContain('<title>board</title>');
        const js = await fetch(`${server.url}/assets/app.js`);
        expect(js.headers.get('content-type')).toMatch(/javascript/);
        const escape = await fetch(`${server.url}/..%2f..%2fpackage.json`);
        expect(escape.status).toBe(404);
        expect((await fetch(`${server.url}/healthz`)).status).toBe(200);
    });
});

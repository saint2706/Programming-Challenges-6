import { serve, upgradeWebSocket } from '@hono/node-server';
import { createHash } from 'node:crypto';
import type { Server } from 'node:http';
import { serveStatic } from '@hono/node-server/serve-static';
import { Hono } from 'hono';
import { secureHeaders } from 'hono/secure-headers';
import { WebSocketServer, type WebSocket as RawSocket } from 'ws';
import { MAX_FRAME_BYTES, ROOM_ID } from '../shared/limits.ts';
import type { ServerMsg } from '../shared/types.ts';
import type { Conn, Room } from './room.ts';
import { Rooms, type RoomsOptions } from './rooms.ts';
import { parseClientMsg } from './schema.ts';
import { RoomStore } from './store.ts';

export interface ServerOptions {
    dataDir: string;
    /** The built client (`dist/`). Omit to run the API only (tests, Vite dev). */
    staticDir?: string;
    port?: number;
    hostname?: string;
    /** Extra origins allowed to open a WebSocket (same-origin always is). */
    allowedOrigins?: string[];
    /**
     * Host names the server answers to. Defaults to localhost names when it listens on loopback, which
     * stops a DNS-rebinding page from reaching it; set it when serving a real domain behind a proxy.
     */
    allowedHosts?: string[];
    rooms?: RoomsOptions;
    /** How long a new connection may stay silent before it is dropped. */
    helloTimeoutMs?: number;
}

export interface RunningServer {
    port: number;
    url: string;
    rooms: Rooms;
    close(): Promise<void>;
}

const HEARTBEAT_MS = 30_000;
/** A client that cannot keep up gets dropped instead of growing the server's memory. */
const MAX_BUFFERED_BYTES = 8 * 1024 * 1024;
const MAX_STRIKES = 20;

export function startServer(options: ServerOptions): Promise<RunningServer> {
    const rooms = new Rooms(new RoomStore(options.dataDir), options.rooms);
    const wss = new WebSocketServer({ noServer: true, maxPayload: MAX_FRAME_BYTES });
    const app = new Hono();
    const allowed = new Set(options.allowedOrigins ?? []);
    const bind = options.hostname ?? '127.0.0.1';
    const hosts =
        options.allowedHosts ?? (['127.0.0.1', 'localhost', '::1'].includes(bind) ? ['localhost', '127.0.0.1', '[::1]'] : null);
    const allowedHosts = hosts ? new Set(hosts.map((h) => h.toLowerCase())) : null;
    app.use(async (c, next) => {
        const host = (c.req.header('host') ?? '').toLowerCase().replace(/:\d+$/, '');
        if (allowedHosts && !allowedHosts.has(host)) return c.text('unexpected host', 421);
        await next();
    });

    const secure = secureHeaders({
        contentSecurityPolicy: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'"],
            styleSrc: ["'self'"],
            imgSrc: ["'self'", 'data:', 'blob:'],
            connectSrc: ["'self'"],
            objectSrc: ["'none'"],
            baseUri: ["'none'"],
            frameAncestors: ["'none'"],
            formAction: ["'self'"],
        },
        referrerPolicy: 'no-referrer',
    });
    // The upgrade response must keep its headers untouched, so the WebSocket route skips this.
    app.use(async (c, next) => (c.req.path.startsWith('/ws/') ? next() : secure(c, next)));

    app.get('/healthz', (c) => c.text('ok'));

    app.get(
        '/ws/:room',
        async (c, next) => {
            if (!ROOM_ID.test(c.req.param('room'))) return c.text('bad room id', 400);
            // Browsers always send Origin on a WebSocket handshake. Refusing foreign ones stops another
            // site's page from driving this server through a visitor's browser (cross-site hijacking).
            const origin = c.req.header('origin');
            if (origin && !allowed.has(origin) && !sameHost(origin, c.req.header('host'))) {
                return c.text('forbidden origin', 403);
            }
            await next();
        },
        upgradeWebSocket((c) => {
            const roomId = c.req.param('room') ?? '';
            let room: Room | undefined;
            let joinedAs: string | null = null;
            let strikes = 0;
            let helloTimer: NodeJS.Timeout | undefined;
            let conn: Conn | undefined;
            return {
                onOpen(_event, ws) {
                    // Opened only after the upgrade succeeded, so stray HTTP hits cannot pin rooms in memory.
                    const opened = rooms.open(roomId);
                    if ('error' in opened) return ws.close(1013, opened.error);
                    room = opened.room;
                    const raw = ws.raw as unknown as RawSocket;
                    conn = {
                        send(msg: ServerMsg, json?: string) {
                            if (ws.readyState !== 1) return;
                            if (raw.bufferedAmount > MAX_BUFFERED_BYTES) return ws.close(1013, 'too slow');
                            ws.send(json ?? JSON.stringify(msg));
                        },
                        close: (code, reason) => ws.close(code, reason),
                    };
                    helloTimer = setTimeout(() => ws.close(1008, 'hello timeout'), options.helloTimeoutMs ?? 5000);
                },
                onMessage(event, ws) {
                    if (!room || !conn) return;
                    if (typeof event.data !== 'string') return ws.close(1003, 'text frames only');
                    const msg = parseClientMsg(event.data);
                    if (!msg) {
                        if (++strikes > MAX_STRIKES || joinedAs === null) ws.close(1008, 'invalid message');
                        return;
                    }
                    if (joinedAs === null) {
                        if (msg.t !== 'hello') return ws.close(1008, 'hello first');
                        clearTimeout(helloTimer);
                        // The id the client sends is its secret. Everyone else only ever sees a hash of it, so
                        // knowing a collaborator's public id does not let you replace their connection.
                        joinedAs = publicId(msg.id);
                        return room.join(conn, { ...msg, id: joinedAs });
                    }
                    room.receive(joinedAs, conn, msg);
                },
                onClose() {
                    clearTimeout(helloTimer);
                    if (room && joinedAs !== null && conn) room.leave(joinedAs, conn);
                    if (room) rooms.release(roomId);
                },
                onError() {
                    // `ws` reports the close separately; nothing to add.
                },
            };
        }),
    );

    if (options.staticDir) {
        app.use(serveStatic({ root: options.staticDir }));
        // Board links look like /b/<room>; the client reads the room from the path.
        app.get('/b/*', serveStatic({ root: options.staticDir, path: 'index.html' }));
    }

    // Drop connections that stopped answering protocol pings (half-open TCP, a closed laptop lid).
    const alive = new WeakSet<RawSocket>();
    wss.on('connection', (ws: RawSocket) => {
        alive.add(ws);
        ws.on('pong', () => alive.add(ws));
    });
    const heartbeat = setInterval(() => {
        for (const ws of wss.clients) {
            if (!alive.has(ws)) {
                ws.terminate();
                continue;
            }
            alive.delete(ws);
            ws.ping();
        }
    }, HEARTBEAT_MS);
    heartbeat.unref();

    return new Promise((resolve) => {
        const server = serve(
            {
                fetch: app.fetch,
                port: options.port ?? 0,
                hostname: options.hostname ?? '127.0.0.1',
                websocket: { server: wss },
            },
            (info) => {
                resolve({
                    port: info.port,
                    url: `http://${info.address === '::1' ? '[::1]' : info.address}:${info.port}`,
                    rooms,
                    async close() {
                        clearInterval(heartbeat);
                        for (const ws of wss.clients) ws.close(1001, 'server shutting down');
                        await rooms.flushAll();
                        rooms.dispose();
                        wss.close();
                        (server as Server).closeAllConnections();
                        await new Promise<void>((done) => server.close(() => done()));
                    },
                });
            },
        );
    });
}

/** What other people see as someone's id: a one-way hash of the secret that tab sent in `hello`. */
export function publicId(secret: string): string {
    return createHash('sha256').update(`whiteboard-public-id:${secret}`).digest('hex').slice(0, 32);
}

function sameHost(origin: string, host: string | undefined): boolean {
    try {
        return host !== undefined && new URL(origin).host === host;
    } catch {
        return false;
    }
}

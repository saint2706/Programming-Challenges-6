// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import type { ServerMsg } from '../shared/types.ts';
import { Connection, type Status } from '../src/connection.ts';

/** A WebSocket the test drives by hand. */
class FakeSocket {
    static all: FakeSocket[] = [];
    readyState = 0;
    sent: string[] = [];
    closeArgs: [number, string] | null = null;
    private listeners = new Map<string, ((e: never) => void)[]>();
    readonly url: string;
    constructor(url: string) {
        this.url = url;
        FakeSocket.all.push(this);
    }
    addEventListener(type: string, fn: (e: never) => void) {
        this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
    }
    send(data: string) {
        this.sent.push(data);
    }
    close(code = 1000, reason = '') {
        this.closeArgs = [code, reason];
        this.fire('close', { code, reason });
    }
    open() {
        this.readyState = 1;
        this.fire('open', {});
    }
    message(msg: ServerMsg | string) {
        this.fire('message', { data: typeof msg === 'string' ? msg : JSON.stringify(msg) });
    }
    drop(code = 1006) {
        this.readyState = 3;
        this.fire('close', { code, reason: '' });
    }
    fire(type: string, event: object) {
        for (const fn of this.listeners.get(type) ?? []) fn(event as never);
    }
}

const welcome: ServerMsg = { t: 'welcome', you: { id: 'a', name: 'a', color: '#000000' }, seq: 0, mode: 'snapshot', items: [], users: [], canUndo: false, canRedo: false };

function make(random = () => 0.5) {
    const statuses: [Status, string | undefined][] = [];
    const messages: ServerMsg[] = [];
    let opens = 0;
    const conn = new Connection({
        url: 'ws://test/ws/room',
        WebSocketImpl: FakeSocket as unknown as typeof WebSocket,
        random,
        onOpen: () => opens++,
        onMessage: (m) => messages.push(m),
        onStatus: (s, d) => statuses.push([s, d]),
    });
    return { conn, statuses, messages, opens: () => opens };
}

const last = () => FakeSocket.all.at(-1)!;

beforeEach(() => {
    FakeSocket.all = [];
    vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe('connecting', () => {
    test('opens, delivers parsed messages, and ignores junk frames', () => {
        const { conn, messages, opens } = make();
        conn.start();
        last().open();
        expect(opens()).toBe(1);
        last().message(welcome);
        last().message('not json');
        expect(messages).toEqual([welcome]);
        conn.stop();
    });

    test('send works only while open', () => {
        const { conn } = make();
        conn.start();
        expect(conn.send({ t: 'ping' })).toBe(false);
        last().open();
        expect(conn.send({ t: 'ping' })).toBe(true);
        expect(last().sent).toEqual(['{"t":"ping"}']);
        conn.stop();
    });

    test('the status becomes online only once the server has said welcome', () => {
        const { conn, statuses } = make();
        conn.start();
        last().open();
        expect(conn.status).toBe('connecting');
        conn.welcomed();
        expect(conn.status).toBe('online');
        expect(statuses.at(-1)).toEqual(['online', undefined]);
        conn.stop();
    });
});

describe('reconnecting', () => {
    test('backs off with growing, jittered, capped delays', () => {
        const { conn } = make(() => 0.5); // jitter factor 1.0
        conn.start();
        const delays: number[] = [];
        for (let i = 0; i < 9; i++) {
            const before = FakeSocket.all.length;
            last().drop();
            // Find the delay by advancing until a new socket appears.
            let waited = 0;
            while (FakeSocket.all.length === before && waited < 20_000) {
                vi.advanceTimersByTime(50);
                waited += 50;
            }
            delays.push(waited);
        }
        expect(delays[0]).toBeGreaterThanOrEqual(450);
        expect(delays[0]).toBeLessThanOrEqual(550);
        for (let i = 1; i < 5; i++) expect(delays[i]!).toBeGreaterThan(delays[i - 1]!);
        expect(Math.max(...delays)).toBeLessThanOrEqual(8100); // never above the cap
        expect(delays.at(-1)).toBeGreaterThanOrEqual(7900);
        conn.stop();
    });

    test('jitter spreads the retries of many clients, within a quarter either side', () => {
        for (const [random, low, high] of [[0, 370, 380], [0.999, 620, 630]] as const) {
            FakeSocket.all = [];
            const { conn } = make(() => random);
            conn.start();
            last().drop();
            let waited = 0;
            while (FakeSocket.all.length === 1) {
                vi.advanceTimersByTime(10);
                waited += 10;
            }
            expect(waited).toBeGreaterThanOrEqual(low);
            expect(waited).toBeLessThanOrEqual(high);
            conn.stop();
        }
    });

    test('a welcome resets the backoff', () => {
        const { conn } = make(() => 0.5);
        conn.start();
        for (let i = 0; i < 4; i++) {
            last().drop();
            vi.advanceTimersByTime(10_000);
        }
        last().open();
        conn.welcomed();
        const before = FakeSocket.all.length;
        last().drop();
        vi.advanceTimersByTime(600);
        expect(FakeSocket.all.length).toBe(before + 1); // back to the short first delay
        conn.stop();
    });

    test('coming back online reconnects at once instead of waiting out the delay', () => {
        const { conn } = make();
        conn.start();
        for (let i = 0; i < 3; i++) {
            last().drop();
            vi.advanceTimersByTime(10_000);
        }
        last().drop(); // now waiting for a long retry
        const before = FakeSocket.all.length;
        window.dispatchEvent(new Event('online'));
        expect(FakeSocket.all.length).toBe(before + 1);
        conn.stop();
    });

    test.each([
        [4001, /another tab/],
        [1008, /rejected/],
        [1009, /rejected/],
    ])('close code %i is final: no retry, and the reason is reported', (code, reason) => {
        const { conn, statuses } = make();
        conn.start();
        last().open();
        last().drop(code);
        vi.advanceTimersByTime(60_000);
        expect(FakeSocket.all).toHaveLength(1);
        expect(conn.status).toBe('closed');
        expect(statuses.at(-1)![1]).toMatch(reason);
    });

    test('stop cancels a pending retry', () => {
        const { conn } = make();
        conn.start();
        last().drop();
        conn.stop();
        vi.advanceTimersByTime(60_000);
        expect(FakeSocket.all).toHaveLength(1);
    });
});

describe('heartbeat', () => {
    test('pings every 15 seconds and drops a link that never answers', () => {
        const { conn } = make();
        conn.start();
        last().open();
        conn.welcomed();
        const socket = last();
        vi.advanceTimersByTime(15_000);
        expect(socket.sent).toContain('{"t":"ping"}');
        expect(socket.closeArgs).toBeNull();
        vi.advanceTimersByTime(10_000);
        expect(socket.closeArgs?.[0]).toBe(4002); // no answer, so it reconnects
        conn.stop();
    });

    test('any message counts as proof of life', () => {
        const { conn } = make();
        conn.start();
        last().open();
        conn.welcomed();
        const socket = last();
        vi.advanceTimersByTime(15_000);
        socket.message({ t: 'pong' });
        vi.advanceTimersByTime(10_000);
        expect(socket.closeArgs).toBeNull();
        conn.stop();
    });
});

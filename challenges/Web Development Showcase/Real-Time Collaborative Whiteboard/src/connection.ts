import type { ClientMsg, ServerMsg } from '../shared/types.ts';

export type Status =
    /** Trying to connect, or waiting to retry. */
    | 'connecting'
    /** Got the server's welcome: edits are flowing. */
    | 'online'
    /** Gave up on its own (see `fatal`). */
    | 'closed';

export interface ConnectionOptions {
    url: string;
    onMessage: (msg: ServerMsg) => void;
    /** Called after the socket opens, before anything else is sent. */
    onOpen: () => void;
    onStatus: (status: Status, detail?: string) => void;
    /** Test hooks. */
    WebSocketImpl?: typeof WebSocket;
    random?: () => number;
}

const MIN_DELAY = 500;
const MAX_DELAY = 8000;
const PING_EVERY = 15_000;
const PONG_WITHIN = 10_000;

/** Close codes the client must not retry: it was replaced, or the server rejected its behaviour. */
const FATAL: Record<number, string> = {
    4001: 'This board was opened in another tab.',
    1008: 'The server rejected a message from this tab.',
    1003: 'The server rejected a message from this tab.',
    1009: 'The server rejected a message from this tab.',
};

/**
 * A WebSocket that reconnects on its own with jittered exponential backoff, notices a connection that
 * silently died (laptop lid, Wi-Fi switch) through a heartbeat, and comes back at once when the
 * browser reports it is online again.
 */
export class Connection {
    status: Status = 'connecting';
    private ws: WebSocket | null = null;
    private delay = MIN_DELAY;
    private retry: ReturnType<typeof setTimeout> | undefined;
    private ping: ReturnType<typeof setTimeout> | undefined;
    private pong: ReturnType<typeof setTimeout> | undefined;
    private stopped = false;
    private readonly options: ConnectionOptions;
    private readonly onNetwork = () => this.reconnectNow();

    constructor(options: ConnectionOptions) {
        this.options = options;
    }

    start(): void {
        addEventListener('online', this.onNetwork);
        document.addEventListener('visibilitychange', this.onNetwork);
        this.open();
    }

    stop(): void {
        this.stopped = true;
        removeEventListener('online', this.onNetwork);
        document.removeEventListener('visibilitychange', this.onNetwork);
        this.clearTimers();
        this.ws?.close(1000, 'bye');
    }

    /** Drop the connection on purpose to test the recovery path. */
    dropForTest(): void {
        this.ws?.close(4000, 'test drop');
    }

    send(msg: ClientMsg): boolean {
        if (this.ws?.readyState !== 1) return false;
        this.ws.send(JSON.stringify(msg));
        return true;
    }

    /** The server answered our hello: the connection is good, so reset the backoff. */
    welcomed(): void {
        this.delay = MIN_DELAY;
        this.set('online');
        this.heartbeat();
    }

    private open(): void {
        if (this.stopped) return;
        clearTimeout(this.retry);
        const Impl = this.options.WebSocketImpl ?? WebSocket;
        this.set('connecting');
        const ws = new Impl(this.options.url);
        this.ws = ws;
        ws.addEventListener('open', () => {
            if (this.ws === ws) this.options.onOpen();
        });
        ws.addEventListener('message', (event) => {
            if (this.ws !== ws || typeof event.data !== 'string') return;
            clearTimeout(this.pong); // any traffic proves the link is alive
            let msg: ServerMsg;
            try {
                msg = JSON.parse(event.data);
            } catch {
                return;
            }
            this.options.onMessage(msg);
        });
        ws.addEventListener('close', (event) => {
            if (this.ws !== ws) return;
            this.ws = null;
            this.clearTimers();
            const fatal = FATAL[event.code];
            if (fatal || this.stopped) return this.set('closed', fatal);
            this.set('connecting');
            const jitter = 0.75 + (this.options.random ?? Math.random)() * 0.5;
            this.retry = setTimeout(() => this.open(), this.delay * jitter);
            this.delay = Math.min(MAX_DELAY, this.delay * 1.7);
        });
        // `error` is always followed by `close`, which handles the retry.
        ws.addEventListener('error', () => {});
    }

    private heartbeat(): void {
        clearTimeout(this.ping);
        this.ping = setTimeout(() => {
            if (this.send({ t: 'ping' })) {
                this.pong = setTimeout(() => this.ws?.close(4002, 'no answer'), PONG_WITHIN);
            }
            this.heartbeat();
        }, PING_EVERY);
    }

    private reconnectNow(): void {
        if (this.stopped || this.status === 'closed' || document.visibilityState === 'hidden') return;
        if (this.ws) {
            // Looks connected, but after sleep or a network change it may be dead: ask, and give up soon.
            if (this.send({ t: 'ping' })) {
                clearTimeout(this.pong);
                this.pong = setTimeout(() => this.ws?.close(4002, 'no answer'), PONG_WITHIN);
            }
            return;
        }
        this.delay = MIN_DELAY;
        this.open();
    }

    private clearTimers(): void {
        clearTimeout(this.ping);
        clearTimeout(this.pong);
        clearTimeout(this.retry);
    }

    private set(status: Status, detail?: string): void {
        if (this.status === status && detail === undefined) return;
        this.status = status;
        this.options.onStatus(status, detail);
    }
}

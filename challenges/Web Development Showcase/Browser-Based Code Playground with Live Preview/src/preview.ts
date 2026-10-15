import { type SandboxEvent, parseSandboxEvent } from './messages.ts';

export interface RunPayload {
    html: string;
    css: string;
    code: string;
    network: boolean;
}

interface Frame {
    el: HTMLIFrameElement;
    run: number;
    network: boolean;
    /** Set once the sandbox page has said it is ready and has been sent its run. */
    started: boolean;
    payload: RunPayload;
    /** How many times the frame has loaded. The first is the sandbox page; later ones may be it rewriting itself, or a navigation. */
    loads: number;
    /** The sandbox page has said it is ready. */
    ready: boolean;
    /** Given to the sandbox page with its run and never again, so only that page can say it is still there. */
    token: string;
    /** The question most recently asked of the page after a load, and the clock for its answer. */
    asked: number;
    deadline: ReturnType<typeof setTimeout> | null;
    /** Messages received from this frame, which a hostile page could make as many of as it likes. */
    messages: number;
}

/** More than this from one run is a flood, not a program; the rest is dropped. Honest runs stop at 1,000 console lines. */
const MAX_MESSAGES_PER_RUN = 5000;

/** How long a page has to say it is still the sandbox page after a load. It answers in a millisecond or two. */
const PRESENCE_MS = 1500;

/**
 * The preview: a sandboxed iframe that is replaced for every run, so one run can never leak state
 * into the next. The sandbox is `allow-scripts allow-forms` only: no `allow-same-origin`, so the page has an opaque
 * origin and can reach nothing of the playground (storage, cookies, DOM, parent) except by postMessage.
 */
export class Preview {
    private readonly host: HTMLElement;
    private readonly onEvent: (event: SandboxEvent) => void;
    private readonly sandboxUrl: (network: boolean) => string;
    private current: Frame | null = null;
    private next: Frame | null = null;
    private runs = 0;
    private scroll = { x: 0, y: 0 };
    private readonly listener = (event: MessageEvent) => this.receive(event);

    constructor(host: HTMLElement, onEvent: (event: SandboxEvent) => void, sandboxUrl: (network: boolean) => string) {
        this.host = host;
        this.onEvent = onEvent;
        this.sandboxUrl = sandboxUrl;
        window.addEventListener('message', this.listener);
    }

    /** Is there a live page that a stylesheet could be swapped into? */
    get live(): boolean {
        return this.current?.started === true && this.next === null;
    }

    get currentRun(): number {
        return this.current?.run ?? 0;
    }

    /** Called when a frame is taken down because its page tried to leave. Says why, in words for the console. */
    onBlocked: (message: string) => void = () => {};

    /** Start a run in a fresh iframe. Returns the run's number. The old page stays until the new one is up. */
    run(payload: RunPayload): number {
        this.discard(this.next);
        const run = ++this.runs;
        const el = document.createElement('iframe');
        // `allow-forms` only lets a form's submit event fire (without it submission is refused before the event,
        // which would silently break every form demo). The sandbox page's `form-action 'none'` stops any navigation.
        el.setAttribute('sandbox', 'allow-scripts allow-forms');
        el.setAttribute('title', 'Preview of your page');
        el.setAttribute('referrerpolicy', 'no-referrer');
        el.className = 'frame pending';
        el.src = this.sandboxUrl(payload.network);
        const frame: Frame = { el, run, network: payload.network, started: false, payload, loads: 0, ready: false, token: crypto.randomUUID(), asked: 0, deadline: null, messages: 0 };
        el.addEventListener('load', () => this.loaded(frame));
        this.next = frame;
        this.host.append(el);
        return run;
    }

    /** Swap the stylesheet in the running page without reloading it. False when there is nothing to swap into. */
    updateCss(css: string): boolean {
        const frame = this.current;
        if (!frame?.started || this.next) return false;
        frame.el.contentWindow?.postMessage({ t: 'css', css }, '*');
        frame.payload = { ...frame.payload, css };
        return true;
    }

    /** Empty the preview (the page is removed, which stops everything it was running). */
    stop(): void {
        this.discard(this.next);
        this.discard(this.current);
        this.next = null;
        this.current = null;
    }

    dispose(): void {
        window.removeEventListener('message', this.listener);
        this.stop();
    }

    private receive(event: MessageEvent): void {
        // Only our own iframes count. A message from any other window, or from the playground itself, is not ours.
        const frame = [this.next, this.current].find((f) => f && f.el.contentWindow === event.source) ?? null;
        if (!frame) return;
        const parsed = parseSandboxEvent(event.data);
        if (!parsed) return;

        if (parsed.t === 'ready') {
            if (frame !== this.next || frame.started) return;
            frame.ready = true;
            this.tryStart(frame);
            return;
        }
        if (parsed.t === 'pong') {
            if (parsed.token === frame.token && parsed.n === frame.asked && frame.deadline) {
                clearTimeout(frame.deadline);
                frame.deadline = null;
            }
            return;
        }
        if (frame !== this.current || ('run' in parsed && parsed.run !== frame.run)) return; // a stale run
        if (++frame.messages > MAX_MESSAGES_PER_RUN) {
            if (frame.messages === MAX_MESSAGES_PER_RUN + 1) this.onEvent({ t: 'log', run: frame.run, level: 'system', group: 0, text: 'This page is sending messages far faster than a program should, so the rest are ignored.', stack: '' });
            return;
        }
        if (parsed.t === 'scroll') this.scroll = { x: parsed.x, y: parsed.y };
        this.onEvent(parsed);
    }

    /**
     * A frame's page may navigate itself and nothing can stop it (no sandbox flag or policy covers it,
     * and the Navigation API is off for opaque origins), so one request to another site, carrying
     * whatever the page likes, can leave before this notices. This takes the page down so that it cannot
     * go on to run anything there. The sandbox page loads once before it is given its run, so a
     * second load then means it left. After the run the page rewrites itself, which loads again, so
     * every later load is answered by asking the page to prove it is still the sandbox page: it holds
     * a token that only its own script was given. A page that has navigated away cannot answer, and
     * the frame is removed.
     */
    private loaded(frame: Frame): void {
        frame.loads++;
        if (frame.loads === 1) return this.tryStart(frame);
        if (frame !== this.next && frame !== this.current) return;
        if (!frame.started) return this.block(frame);
        if (frame.deadline) clearTimeout(frame.deadline);
        frame.asked++;
        frame.el.contentWindow?.postMessage({ t: 'ping', n: frame.asked }, '*');
        frame.deadline = setTimeout(() => this.block(frame), PRESENCE_MS);
    }

    private block(frame: Frame): void {
        if (frame !== this.next && frame !== this.current) return;
        this.discard(frame);
        if (this.next === frame) this.next = null;
        if (this.current === frame) this.current = null;
        this.onBlocked('The preview tried to navigate away, so it was stopped. A page in the preview cannot leave it.');
    }

    /** Start the run once the sandbox has said it is ready and its document has loaded (in either order). */
    private tryStart(frame: Frame): void {
        if (frame !== this.next || frame.started || !frame.ready || frame.loads < 1) return;
        frame.started = true;
        frame.el.contentWindow?.postMessage({ t: 'run', run: frame.run, token: frame.token, html: frame.payload.html, css: frame.payload.css, code: frame.payload.code, scroll: this.scroll }, '*');
        // The new page is running: show it and drop the old one.
        frame.el.classList.remove('pending');
        const old = this.current;
        this.current = frame;
        this.next = null;
        this.discard(old);
    }

    private discard(frame: Frame | null): void {
        if (frame?.deadline) clearTimeout(frame.deadline);
        frame?.el.remove();
    }
}

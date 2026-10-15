// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import type { SandboxEvent } from '../src/messages.ts';
import { Preview, type RunPayload } from '../src/preview.ts';

const payload = (over: Partial<RunPayload> = {}): RunPayload => ({ html: '<p>x</p>', css: 'p{}', code: 'a', network: false, ...over });

let host: HTMLElement;
let events: SandboxEvent[];
let preview: Preview;

/** A message as the browser delivers it: `source` is the window that posted it. */
function post(source: Window | null, data: unknown) {
    window.dispatchEvent(new MessageEvent('message', { data, source }));
}

const frames = () => [...host.querySelectorAll('iframe')];

/** The sandbox page has loaded and said it is ready (a real browser does the first on its own). */
function boot(frame: HTMLIFrameElement) {
    frame.dispatchEvent(new Event('load'));
    post(frame.contentWindow, { t: 'ready' });
}
const spyOnFrame = (frame: HTMLIFrameElement) => vi.spyOn(frame.contentWindow!, 'postMessage');

beforeEach(() => {
    document.body.innerHTML = '<div id="host"></div>';
    host = document.getElementById('host')!;
    events = [];
    preview = new Preview(host, (e) => events.push(e), (network) => (network ? 'sandbox-net.html' : 'sandbox.html'));
});

afterEach(() => {
    preview.dispose();
    vi.useRealTimers();
});

describe('the iframe', () => {
    test('is sandboxed with scripts and form events allowed and nothing else: no same-origin, so it has no origin of its own to share', () => {
        preview.run(payload());
        const [frame] = frames();
        expect(frame!.getAttribute('sandbox')).toBe('allow-scripts allow-forms');
        const tokens = frame!.getAttribute('sandbox')!.split(/\s+/); // (jsdom has no DOMTokenList for this)
        expect(tokens).not.toContain('allow-same-origin');
        expect(tokens).not.toContain('allow-top-navigation');
        expect(tokens).not.toContain('allow-popups');
        expect(tokens).toContain('allow-forms'); // so a form's submit event fires; the page's form-action 'none' stops it going anywhere
        expect(tokens).not.toContain('allow-modals');
        expect(frame!.getAttribute('referrerpolicy')).toBe('no-referrer');
        expect(frame!.title).toMatch(/Preview/);
    });

    test('the page it loads depends on whether the network is allowed', () => {
        preview.run(payload({ network: false }));
        expect(frames()[0]!.getAttribute('src')).toBe('sandbox.html');
        preview.run(payload({ network: true }));
        expect(frames().at(-1)!.getAttribute('src')).toBe('sandbox-net.html');
    });
});

describe('starting a run', () => {
    test('the sandbox is sent its page only after it says it is ready, and only once', () => {
        const run = preview.run(payload());
        const frame = frames()[0]!;
        const send = spyOnFrame(frame);
        expect(send).not.toHaveBeenCalled();
        boot(frame);
        expect(send).toHaveBeenCalledTimes(1);
        expect(send.mock.calls[0]![0]).toMatchObject({ t: 'run', run, html: '<p>x</p>', css: 'p{}', code: 'a', scroll: { x: 0, y: 0 } });
        post(frame.contentWindow, { t: 'ready' }); // a second ready (a forged one, say) does nothing
        expect(send).toHaveBeenCalledTimes(1);
    });

    test('the old page stays until the new one is up, then goes', () => {
        preview.run(payload());
        const first = frames()[0]!;
        boot(first);
        expect(preview.live).toBe(true);

        preview.run(payload({ code: 'b' }));
        const [old, fresh] = frames();
        expect(old).toBe(first);
        expect(fresh!.classList.contains('pending')).toBe(true); // hidden while it starts
        expect(preview.live).toBe(false);
        boot(fresh!);
        expect(frames()).toEqual([fresh]);
        expect(fresh!.classList.contains('pending')).toBe(false);
        expect(preview.live).toBe(true);
    });

    test('starting another run before the last one came up drops the unfinished one', () => {
        preview.run(payload());
        preview.run(payload());
        expect(frames()).toHaveLength(1);
    });

    test('run numbers count up', () => {
        const a = preview.run(payload());
        boot(frames()[0]!);
        const b = preview.run(payload());
        expect(b).toBe(a + 1);
        expect(preview.currentRun).toBe(a);
        boot(frames().at(-1)!);
        expect(preview.currentRun).toBe(b);
    });
});

describe('when a run starts', () => {
    test('only once the page has both loaded and said ready, whichever comes first', () => {
        for (const order of ['ready first', 'load first'] as const) {
            preview.run(payload());
            const frame = frames().at(-1)!;
            const send = spyOnFrame(frame);
            if (order === 'ready first') post(frame.contentWindow, { t: 'ready' });
            else frame.dispatchEvent(new Event('load'));
            expect(send, order).not.toHaveBeenCalled();
            if (order === 'ready first') frame.dispatchEvent(new Event('load'));
            else post(frame.contentWindow, { t: 'ready' });
            expect(send, order).toHaveBeenCalledTimes(1);
        }
    });
});

describe('a page that navigates away', () => {
    /** Boot a run and return the frame, the token its page was given, and a spy on what it is sent. */
    function started() {
        const blocked: string[] = [];
        preview.onBlocked = (m) => blocked.push(m);
        preview.run(payload());
        const frame = frames()[0]!;
        const send = spyOnFrame(frame);
        boot(frame);
        const token = (send.mock.calls[0]![0] as { token: string }).token;
        return { frame, send, token, blocked };
    }
    const pings = (send: ReturnType<typeof spyOnFrame>) => send.mock.calls.map((c) => c[0] as { t: string; n: number }).filter((m) => m.t === 'ping');

    test('after its run a page loads again as it rewrites itself, so it is asked to prove it is still the sandbox page, and stays if it can', () => {
        vi.useFakeTimers();
        const { frame, send, token, blocked } = started();
        frame.dispatchEvent(new Event('load'));
        expect(pings(send)).toHaveLength(1);
        post(frame.contentWindow, { t: 'pong', n: pings(send)[0]!.n, token });
        vi.advanceTimersByTime(10_000);
        expect(frames()).toHaveLength(1);
        expect(preview.live).toBe(true);
        expect(blocked).toEqual([]);
    });

    test('a page that does not answer has left: the frame is removed and the reason given', () => {
        vi.useFakeTimers();
        const { frame, blocked } = started();
        frame.dispatchEvent(new Event('load')); // the page went somewhere else
        expect(frames()).toHaveLength(1); // it is given a moment to answer
        vi.advanceTimersByTime(2000);
        expect(frames()).toHaveLength(0);
        expect(preview.live).toBe(false);
        expect(blocked).toHaveLength(1);
        expect(blocked[0]).toMatch(/navigate away/);
    });

    test('an answer with the wrong token, or to an old question, does not count: any page can post, only the sandbox page has the token', () => {
        vi.useFakeTimers();
        const { frame, send, token } = started();
        frame.dispatchEvent(new Event('load'));
        const n = pings(send)[0]!.n;
        post(frame.contentWindow, { t: 'pong', n, token: 'guess' });
        post(frame.contentWindow, { t: 'pong', n: n + 1, token });
        post(frame.contentWindow, { t: 'pong', n });
        vi.advanceTimersByTime(2000);
        expect(frames()).toHaveLength(0);
    });

    test('a frame that loads twice before its run was sent is removed at once, and nothing it says afterwards is heard', () => {
        const blocked: string[] = [];
        preview.onBlocked = (m) => blocked.push(m);
        const run = preview.run(payload());
        const frame = frames()[0]!;
        frame.dispatchEvent(new Event('load'));
        frame.dispatchEvent(new Event('load'));
        expect(frames()).toHaveLength(0);
        post(frame.contentWindow, { t: 'log', run, level: 'log', group: 0, text: 'ghost' });
        expect(events).toEqual([]);
        expect(blocked).toHaveLength(1);
    });

    test('the token is different for every run', () => {
        const first = started().token;
        preview.run(payload());
        const fresh = frames().at(-1)!;
        const send = spyOnFrame(fresh);
        boot(fresh);
        expect((send.mock.calls[0]![0] as { token: string }).token).not.toBe(first);
    });
});

describe('a flood of messages', () => {
    test('is cut off after a few thousand from one run, with a single notice', () => {
        const run = preview.run(payload());
        const frame = frames()[0]!;
        boot(frame);
        for (let i = 0; i < 20_000; i++) post(frame.contentWindow, { t: 'log', run, level: 'log', group: 0, text: String(i) });
        const logs = events.filter((e) => e.t === 'log');
        expect(logs.length).toBeLessThan(5100);
        expect(logs.filter((e) => e.t === 'log' && e.level === 'system')).toHaveLength(1);
        // A new run has its own allowance.
        preview.run(payload());
        const next = frames().at(-1)!;
        boot(next);
        events.length = 0;
        post(next.contentWindow, { t: 'log', run: run + 1, level: 'log', group: 0, text: 'fresh' });
        expect(events).toHaveLength(1);
    });
});

describe('who it believes', () => {
    function started() {
        const run = preview.run(payload());
        const frame = frames()[0]!;
        boot(frame);
        return { run, frame };
    }

    test('events from its own current frame are passed on', () => {
        const { run, frame } = started();
        post(frame.contentWindow, { t: 'log', run, level: 'log', group: 0, text: 'hi' });
        expect(events).toEqual([{ t: 'log', run, level: 'log', group: 0, text: 'hi', stack: '' }]);
    });

    test('a message from the playground\'s own window, from another frame, or from nowhere is ignored', () => {
        const { run } = started();
        const stranger = document.createElement('iframe');
        document.body.append(stranger);
        for (const source of [window, stranger.contentWindow, null]) post(source, { t: 'log', run, level: 'log', group: 0, text: 'spoof' });
        expect(events).toEqual([]);
    });

    test('a stale run\'s events are dropped, and a frame that is not yet current cannot speak (other than ready)', () => {
        const { run, frame } = started();
        post(frame.contentWindow, { t: 'log', run: run + 5, level: 'log', group: 0, text: 'stale' });
        preview.run(payload());
        const pending = frames().at(-1)!;
        post(pending.contentWindow, { t: 'log', run: run + 1, level: 'log', group: 0, text: 'too early' });
        expect(events).toEqual([]);
    });

    test('an old frame that has been replaced is no longer heard', () => {
        const { run, frame } = started();
        preview.run(payload());
        boot(frames().at(-1)!);
        post(frame.contentWindow, { t: 'log', run, level: 'log', group: 0, text: 'ghost' });
        expect(events).toEqual([]);
    });

    test('junk from the right frame is ignored too', () => {
        const { frame } = started();
        for (const data of [null, 'x', 5, { t: 'unknown' }, { t: 'log', level: 'nope' }]) post(frame.contentWindow, data);
        expect(events).toEqual([]);
    });
});

describe('scroll position and styles', () => {
    test('the scroll position reported by the page is given to the next run, so a reload does not jump to the top', () => {
        const run = preview.run(payload());
        boot(frames()[0]!);
        post(frames()[0]!.contentWindow, { t: 'scroll', run, x: 0, y: 420 });
        preview.run(payload());
        const next = frames().at(-1)!;
        const send = spyOnFrame(next);
        boot(next);
        expect(send.mock.calls[0]![0]).toMatchObject({ scroll: { x: 0, y: 420 } });
    });

    test('a stylesheet is swapped into the live page, but only when there is one', () => {
        expect(preview.updateCss('a{}')).toBe(false); // nothing running
        preview.run(payload());
        expect(preview.updateCss('a{}')).toBe(false); // still starting
        const frame = frames()[0]!;
        boot(frame);
        const send = spyOnFrame(frame);
        expect(preview.updateCss('b{}')).toBe(true);
        expect(send).toHaveBeenCalledWith({ t: 'css', css: 'b{}' }, '*');
        preview.run(payload()); // a new page is coming: do not touch the old one
        expect(preview.updateCss('c{}')).toBe(false);
    });
});

test('stop empties the preview, and dispose stops listening', () => {
    preview.run(payload());
    boot(frames()[0]!);
    preview.stop();
    expect(frames()).toHaveLength(0);
    expect(preview.live).toBe(false);
    const run = preview.run(payload());
    const frame = frames()[0]!;
    preview.dispose();
    boot(frame);
    expect(run).toBeGreaterThan(0);
    expect(frames()).toHaveLength(0);
});

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { JSDOM } from 'jsdom';
import { beforeEach, describe, expect, test } from 'vitest';

const read = (name: string) => readFileSync(join(process.cwd(), 'public', name), 'utf8');
const SOURCES = [read('sandbox-format.js'), read('sandbox.js')];

type Post = Record<string, unknown>;

/**
 * A fresh page with the sandbox's own scripts loaded, as the browser would: `window.parent` is the
 * playground, which here is a recorder. The clock and the timers are ours, so tests control time.
 */
function sandbox(origin = 'null') {
    const dom = new JSDOM('<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="default-src \'none\'"></head><body></body></html>', {
        runScripts: 'outside-only',
        url: 'https://sandbox.test/sandbox.html',
    });
    const win = dom.window as unknown as Window & typeof globalThis & Record<string, any>;
    const posted: Post[] = [];
    let clock = 1000;
    const timers: (() => void)[] = [];
    Object.defineProperty(win, 'origin', { value: origin, configurable: true }); // "null" is what a sandboxed page reports
    Object.defineProperty(win, 'postMessage', { value: (message: Post) => void posted.push(message), configurable: true });
    win.performance.now = () => clock;
    let nextTimer = 1;
    win.setTimeout = ((fn: () => void) => (timers.push(fn), nextTimer++)) as never; // a real id, as a browser returns
    for (const source of SOURCES) win.eval(source);
    win.URL.createObjectURL = (() => 'blob:run') as never;
    win.URL.revokeObjectURL = (() => {}) as never;
    return {
        win,
        posted,
        logs: () => posted.filter((m) => m.t === 'log'),
        advance: (ms: number) => void (clock += ms),
        turn: () => timers.splice(0).forEach((fn) => fn()), // the event loop turns
        fromParent: (data: unknown) => win.dispatchEvent(new win.MessageEvent('message', { data, source: win })),
    };
}

let s: ReturnType<typeof sandbox>;
beforeEach(() => {
    s = sandbox();
});

test('it announces itself as ready', () => {
    expect(s.posted[0]).toEqual({ t: 'ready' });
});

describe('it refuses to run unless it really is sandboxed', () => {
    test('opened on the playground\'s own origin (top level, or in an iframe without the sandbox attribute) it does nothing', () => {
        const t = sandbox('https://playground.example');
        expect(t.posted).toEqual([]); // not even "ready"
        t.fromParent({ t: 'run', run: 1, html: '<p id="x">x</p>', css: '', code: 'steal()' });
        t.fromParent({ t: 'css', css: 'body{}' });
        expect(t.win.document.getElementById('x')).toBeNull();
        expect(t.win.document.getElementById('pg-css')).toBeNull();
        // And the console is left alone, so nothing is reported either.
        t.win.console.log('hello');
        expect(t.posted).toEqual([]);
    });

    test('it says why', () => {
        const t = sandbox('https://playground.example');
        t.win.document.dispatchEvent(new t.win.Event('DOMContentLoaded'));
        expect(t.win.document.body.textContent).toMatch(/only runs inside the playground/);
    });
});

describe('console capture', () => {
    test('log, info, debug, warn and error arrive with their level and text', () => {
        const { console: c } = s.win;
        c.log('a', 1);
        c.info('b');
        c.debug('c');
        c.warn('d');
        c.error('e', { x: 1 });
        expect(s.logs().map((m) => [m.level, m.text])).toEqual([
            ['log', 'a 1'],
            ['info', 'b'],
            ['debug', 'c'],
            ['warn', 'd'],
            ['error', 'e { x: 1 }'],
        ]);
    });

    test('warnings and errors carry a stack; plain logs do not (it costs time)', () => {
        s.win.console.error('boom');
        s.win.console.log('plain');
        const [error, plain] = s.logs();
        expect(typeof error!.stack).toBe('string');
        expect(String(error!.stack)).not.toMatch(/sandbox\.js/);
        expect(plain!.stack).toBeUndefined();
    });

    test('assert only speaks when the condition is false', () => {
        s.win.console.assert(true, 'quiet');
        s.win.console.assert(false, 'loud', 2);
        expect(s.logs().map((m) => m.text)).toEqual(['Assertion failed: loud 2']);
    });

    test('groups indent, and clear resets', () => {
        const c = s.win.console;
        c.group('outer');
        c.log('inside');
        c.group();
        c.log('deeper');
        c.groupEnd();
        c.groupEnd();
        c.groupEnd(); // extra ends never go negative
        c.log('out');
        expect(s.logs().map((m) => [m.text, m.group])).toEqual([['outer', 0], ['inside', 1], ['deeper', 2], ['out', 0]]);
        c.clear();
        expect(s.posted.at(-1)).toMatchObject({ t: 'clear' });
    });

    test('count and time', () => {
        const c = s.win.console;
        c.count();
        c.count('x');
        c.count();
        c.time('t');
        s.advance(5);
        c.timeEnd('t');
        c.timeEnd('missing');
        expect(s.logs().map((m) => m.text)).toEqual(['default: 1', 'x: 1', 'default: 2', 't: 5.00 ms', 'Timer "missing" does not exist']);
    });

    test('dialogs are explained instead of failing silently, and return safe values', () => {
        expect(s.win.confirm('sure?')).toBe(false);
        expect(s.win.prompt('name?')).toBeNull();
        s.win.alert('hi');
        expect(s.logs()).toHaveLength(3);
        expect(s.logs().every((m) => m.level === 'warn' && /not available/.test(String(m.text)))).toBe(true);
    });

    test('after 1,000 messages it says so once and drops the rest, and clearing gives a fresh allowance', () => {
        for (let i = 0; i < 5000; i++) s.win.console.log(i);
        expect(s.logs()).toHaveLength(1001);
        expect(s.logs()[1000]).toMatchObject({ level: 'system' });
        s.win.console.clear();
        s.win.console.log('again');
        expect(s.logs().at(-1)!.text).toBe('again');
    });

    test('logging something hostile never throws into the person\'s code', () => {
        const hostile = new Proxy({}, { ownKeys() { throw new Error('no'); } });
        expect(() => s.win.console.log(hostile, { get x() { throw new Error('no'); } })).not.toThrow();
    });
});

describe('reporting problems from the page', () => {
    test('an uncaught error', () => {
        const error = new s.win.RangeError('too far');
        s.win.dispatchEvent(new s.win.ErrorEvent('error', { error, message: 'ignored when error is set', filename: 'blob:x', lineno: 7, colno: 3 }));
        expect(s.posted.at(-1)).toMatchObject({ t: 'error', name: 'RangeError', message: 'too far', line: 7, column: 3 });
    });

    test('an error with no Error object (a script error)', () => {
        s.win.dispatchEvent(new s.win.ErrorEvent('error', { message: 'Script error.' }));
        expect(s.posted.at(-1)).toMatchObject({ t: 'error', message: 'Script error.', name: 'Error' });
    });

    test('an unhandled promise rejection', () => {
        const event = new s.win.Event('unhandledrejection') as Event & { reason: unknown };
        event.reason = new s.win.Error('nope');
        s.win.dispatchEvent(event);
        expect(s.posted.at(-1)).toMatchObject({ t: 'error', name: 'Error' });
        expect(String(s.posted.at(-1)!.message)).toMatch(/Unhandled promise rejection/);
    });

    test('a content security policy violation', () => {
        const event = new s.win.Event('securitypolicyviolation');
        Object.assign(event, { effectiveDirective: 'script-src-elem', blockedURI: 'inline', sample: 'alert(1)' });
        s.win.dispatchEvent(event);
        expect(s.posted.at(-1)).toMatchObject({ t: 'csp', directive: 'script-src-elem', blocked: 'inline' });
    });

    test('scrolling is reported, but not on every pixel', () => {
        for (let i = 0; i < 3; i++) s.win.dispatchEvent(new s.win.Event('scroll'));
        s.turn();
        expect(s.posted.filter((m) => m.t === 'scroll')).toHaveLength(1);
    });
});

describe('who it listens to', () => {
    test('messages from any window but its parent are ignored', () => {
        const stranger = s.win.document.createElement('iframe');
        s.win.document.body.append(stranger);
        s.win.dispatchEvent(new s.win.MessageEvent('message', { data: { t: 'css', css: 'body{color:red}' }, source: stranger.contentWindow }));
        expect(s.win.document.getElementById('pg-css')).toBeNull();
        s.fromParent({ t: 'css', css: 'body{color:red}' });
        expect(s.win.document.getElementById('pg-css')?.textContent).toBe('body{color:red}');
    });

    test('malformed messages do nothing', () => {
        for (const data of [null, undefined, 5, 'run', [], { t: 'run' }, { t: 'run', html: 1, css: '', code: '' }, { t: 'css', css: 5 }, { t: 'bogus' }]) {
            expect(() => s.fromParent(data)).not.toThrow();
        }
        expect(s.win.document.body.innerHTML).toBe('');
    });

    test('oversized pages are refused', () => {
        s.fromParent({ t: 'run', run: 1, html: 'x'.repeat(3_000_001), css: '', code: '' });
        expect(s.win.document.body.textContent).toBe('');
    });
});

describe('starting a run', () => {
    test('the page is built as a tree: our policy first, then the person\'s head and body, their stylesheet and the script last', () => {
        s.fromParent({ t: 'run', run: 4, html: '<!doctype html><html lang="fr"><head><title>T</title></head><body class="b"><h1>Hi</h1></body></html>', css: 'h1{color:teal}', code: 'console.log(1)' });
        const d = s.win.document;
        expect(d.documentElement.lang).toBe('fr');
        expect(d.body.className).toBe('b');
        expect(d.head.firstElementChild?.getAttribute('http-equiv')).toBe('Content-Security-Policy');
        expect(d.head.firstElementChild?.getAttribute('content')).toBe("default-src 'none'");
        expect(d.title).toBe('T');
        expect(d.getElementById('pg-css')?.textContent).toBe('h1{color:teal}');
        expect(d.querySelector('h1')?.textContent).toBe('Hi');
        const script = d.body.lastElementChild;
        expect(script?.tagName).toBe('SCRIPT');
        expect(script?.getAttribute('type')).toBe('module');
        expect(script?.getAttribute('src')).toBe('blob:run');
    });

    test('their HTML cannot break out of its place: a stray closing tag lands inside the tree', () => {
        s.fromParent({ t: 'run', run: 1, html: '</body></html><p>after</p>', css: '', code: 'y' });
        expect(s.win.document.querySelector('p')?.textContent).toBe('after');
        expect(s.win.document.head.firstElementChild?.getAttribute('http-equiv')).toBe('Content-Security-Policy');
        expect(s.win.document.body.lastElementChild?.getAttribute('src')).toBe('blob:run');
    });

    test('with no code there is no script, and later logs carry the run number', () => {
        s.fromParent({ t: 'run', run: 9, html: '<p>x</p>', css: '', code: '' });
        expect(s.win.document.body.querySelector('script[type=module]')).toBeNull();
        s.win.console.log('hello');
        expect(s.logs().at(-1)).toMatchObject({ run: 9, text: 'hello' });
    });

    test('the previous run\'s blob URL is released', () => {
        const revoked: string[] = [];
        s.win.URL.revokeObjectURL = ((u: string) => void revoked.push(u)) as never;
        s.fromParent({ t: 'run', run: 1, html: '<p>1</p>', css: '', code: 'a' });
        s.fromParent({ t: 'run', run: 2, html: '<p>2</p>', css: '', code: 'b' });
        expect(revoked).toContain('blob:run');
    });

    test('a stack that mentions the blob URL is shown as bundle.js', () => {
        s.fromParent({ t: 'run', run: 1, html: '<p>x</p>', css: '', code: 'a' });
        const error = new s.win.Error('x');
        error.stack = 'Error: x\n    at f (blob:run:3:9)';
        s.win.dispatchEvent(new s.win.ErrorEvent('error', { error }));
        expect(String(s.posted.at(-1)!.stack)).toContain('bundle.js:3:9');
        expect(String(s.posted.at(-1)!.stack)).not.toContain('blob:run');
    });

    test('the window\'s listeners are put back after the page is replaced, so errors in the new page still reach the playground', () => {
        s.fromParent({ t: 'run', run: 3, html: '<p>x</p>', css: '', code: '' });
        s.win.dispatchEvent(new s.win.ErrorEvent('error', { error: new s.win.Error('later') }));
        expect(s.posted.at(-1)).toMatchObject({ t: 'error', run: 3, message: 'later' });
        s.fromParent({ t: 'css', css: 'p{}' }); // and messages from the playground still arrive
        expect(s.win.document.getElementById('pg-css')?.textContent).toBe('p{}');
    });
});

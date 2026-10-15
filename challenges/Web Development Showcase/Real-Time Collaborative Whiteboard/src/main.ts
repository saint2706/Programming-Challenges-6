import { ROOM_ID } from '../shared/limits.ts';
import './style.css';
import { Connection } from './connection.ts';
import { Doc } from './doc.ts';
import { toPng, toSvg } from './export.ts';
import { h, mountUi } from './ui.ts';
import { Whiteboard } from './whiteboard.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** 32 hex characters from the browser's CSPRNG. (`randomUUID` needs a secure context; this does not.) */
function uuid(): string {
    return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
}

function newRoomId(): string {
    return Array.from(crypto.getRandomValues(new Uint8Array(10)), (b) => (b % 36).toString(36)).join('');
}

/** Storage can be blocked or full; the app has to work without it. */
function stored(storage: 'session' | 'local', key: string, make: () => string): string {
    try {
        const s = storage === 'session' ? sessionStorage : localStorage;
        const existing = s.getItem(key);
        if (existing) return existing;
        const value = make();
        s.setItem(key, value);
        return value;
    } catch {
        return make();
    }
}

function remember(key: string, value: string): void {
    try {
        localStorage.setItem(key, value);
    } catch {
        // not saved, still works
    }
}

const ADJECTIVES = ['Quick', 'Calm', 'Brave', 'Bright', 'Gentle', 'Lucky', 'Witty', 'Swift', 'Sunny', 'Keen', 'Merry', 'Noble'];
const ANIMALS = ['Otter', 'Falcon', 'Lynx', 'Heron', 'Panda', 'Gecko', 'Newt', 'Robin', 'Koala', 'Fox', 'Wren', 'Seal'];

function funName(seed: string): string {
    const n = parseInt(seed.slice(0, 6), 16) || 0;
    return `${ADJECTIVES[n % ADJECTIVES.length]} ${ANIMALS[Math.floor(n / ADJECTIVES.length) % ANIMALS.length]}`;
}

function roomFromLocation(): string | null {
    if (location.pathname === '/') {
        const id = newRoomId();
        history.replaceState(null, '', `/b/${id}${location.search}`);
        return id;
    }
    const match = /^\/b\/([^/]+)\/?$/.exec(location.pathname);
    return match && ROOM_ID.test(match[1]!) ? match[1]! : null;
}

function download(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = h('a', { href: url, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

const NACK_TEXT: Record<string, string> = {
    rate: 'You are drawing faster than the server allows, so that change was dropped.',
    full: 'This board is full. Erase something to make room.',
};

function start(): void {
    const room = roomFromLocation();
    if (!room) {
        document.body.replaceChildren(
            h('main', { class: 'noscript' }, h('h1', {}, 'That board link is not valid'), h('p', {}, h('a', { href: '/' }, 'Start a new board'))),
        );
        return;
    }
    document.title = `Whiteboard ${room} · Collaborative Whiteboard`;

    const clientId = stored('session', 'wb:id', uuid);
    const nameInput = $<HTMLInputElement>('name');
    nameInput.value = stored('local', 'wb:name', () => funName(clientId));
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/${room}`;

    const doc = new Doc((msg) => void conn.send(msg), uuid, clientId);
    const conn: Connection = new Connection({
        url: wsUrl,
        onOpen: () => conn.send(doc.hello(nameInput.value.trim() || funName(clientId))),
        onMessage: (msg) => {
            if (msg.t === 'welcome') conn.welcomed();
            doc.handle(msg);
        },
        onStatus: (status, detail) => {
            ui.setDetail(status === 'closed' ? detail : undefined);
            if (status === 'online') ui.announce('Connected');
            else if (status === 'connecting' && wasOnline) ui.announce('Connection lost, reconnecting');
            wasOnline = status === 'online';
        },
    });
    let wasOnline = false;

    const board = new Whiteboard(
        { stage: $('stage'), base: $<HTMLCanvasElement>('base'), top: $<HTMLCanvasElement>('top'), editor: $<HTMLTextAreaElement>('editor') },
        doc,
        (msg) => void conn.send(msg),
        uuid,
    );

    const ui = mountUi(
        {
            toolbar: $('toolbar'),
            status: $('status'),
            people: $('people'),
            toasts: $('toasts'),
            announcer: $('announcer'),
            notes: $('notes'),
            nameInput,
            top: $<HTMLCanvasElement>('top'),
            viewbar: $('viewbar'),
            actions: $('actions'),
        },
        board,
        doc,
        conn,
        {
            rename(name) {
                remember('wb:name', name);
                conn.send({ t: 'name', name });
            },
            async exportPng() {
                const blob = await toPng(doc.visible(), board.measure);
                if (blob) download(blob, `whiteboard-${room}.png`);
                else ui.toast('There is nothing on the board to export yet.');
            },
            exportSvg() {
                const svg = toSvg(doc.visible(), board.measure);
                if (svg) download(new Blob([svg], { type: 'image/svg+xml' }), `whiteboard-${room}.svg`);
                else ui.toast('There is nothing on the board to export yet.');
            },
            async copyLink() {
                const link = `${location.origin}/b/${room}`;
                try {
                    await navigator.clipboard.writeText(link);
                    ui.toast('Link copied. Anyone who opens it joins this board.');
                } catch {
                    ui.toast(`Copy this link: ${link}`);
                }
            },
        },
    );

    const redraw = doc.onChange;
    doc.onChange = () => {
        redraw();
        ui.sync();
    };
    board.onState = ui.sync;
    doc.onNack = (reason) => {
        const text = NACK_TEXT[reason];
        if (text) ui.toast(text);
    };

    // Unsent changes would be lost on close, so ask first (only while there are any).
    addEventListener('beforeunload', (e) => {
        if (doc.pendingCount > 0) e.preventDefault();
    });

    // A hook for the browser tests: it exposes the live objects, not a way to bypass the server.
    if (new URLSearchParams(location.search).has('e2e')) {
        Object.assign(window, { __wb: { board, doc, conn } });
    }

    ui.sync();
    conn.start();
}

start();

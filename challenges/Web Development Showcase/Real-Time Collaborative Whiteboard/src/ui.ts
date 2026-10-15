import { MAX_NAME } from '../shared/limits.ts';
import type { Connection } from './connection.ts';
import type { Doc } from './doc.ts';
import { icon } from './icons.ts';
import { COLORS, SIZE_NAMES, STICKY_COLORS, TOOLS, type ToolId } from './tools.ts';
import type { Whiteboard } from './whiteboard.ts';

type Attrs = Record<string, string | boolean | undefined>;

/** A tiny element builder. Text goes in with `textContent`, never as HTML, so names and notes cannot inject markup. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (v === undefined || v === false) continue;
        node.setAttribute(k, v === true ? '' : v);
    }
    node.append(...children);
    return node;
}

const parser = new DOMParser();

/** Icons are parsed as SVG documents (our own constants), so no HTML string ever reaches the page. */
function svgIcon(name: Parameters<typeof icon>[0]): HTMLElement {
    const svg = parser.parseFromString(icon(name), 'image/svg+xml').documentElement;
    return h('span', { class: 'icon' }, document.importNode(svg, true));
}

export interface Ui {
    /** Re-read the doc, connection and board and update everything shown. */
    sync(): void;
    toast(text: string): void;
    announce(text: string): void;
    /** Why the connection gave up, shown in the status line once it has. */
    setDetail(text: string | undefined): void;
}

interface Elements {
    toolbar: HTMLElement;
    status: HTMLElement;
    people: HTMLElement;
    toasts: HTMLElement;
    announcer: HTMLElement;
    notes: HTMLElement;
    nameInput: HTMLInputElement;
    top: HTMLCanvasElement;
    viewbar: HTMLElement;
    actions: HTMLElement;
}

export interface UiHandlers {
    rename(name: string): void;
    exportPng(): void;
    exportSvg(): void;
    copyLink(): void;
}

export function mountUi(el: Elements, board: Whiteboard, doc: Doc, conn: Connection, handlers: UiHandlers): Ui {
    // ---- tools
    const toolButtons = new Map<ToolId, HTMLButtonElement>();
    const tools = h('div', { class: 'group', role: 'group', 'aria-label': 'Tools' });
    for (const t of TOOLS) {
        const b = h('button', {
            type: 'button',
            class: 'tool',
            title: `${t.label} (${t.key.toUpperCase()})`,
            'aria-label': t.label,
            'aria-keyshortcuts': t.key.toUpperCase(),
            'aria-pressed': 'false',
        }, svgIcon(t.id));
        b.addEventListener('click', () => board.setTool(t.id));
        toolButtons.set(t.id, b);
        tools.append(b);
    }

    // ---- colours: native radios, so keyboard and screen reader behaviour comes for free
    const swatches = h('div', { class: 'group swatches', role: 'radiogroup', 'aria-label': 'Colour' });
    const custom = h('input', { type: 'color', class: 'custom-color', 'aria-label': 'Custom colour', value: COLORS[0] });
    custom.addEventListener('input', () => board.setColor(custom.value));
    const rebuildSwatches = () => {
        const palette = board.tool === 'sticky' ? STICKY_COLORS : COLORS;
        swatches.replaceChildren(
            ...palette.map((c) => {
                const input = h('input', { type: 'radio', name: 'color', value: c, 'aria-label': `Colour ${c}` });
                input.checked = c.toLowerCase() === board.activeColor().toLowerCase();
                input.addEventListener('change', () => board.setColor(c));
                const label = h('label', { class: 'swatch' }, input, h('span', { class: 'chip' }));
                label.querySelector<HTMLElement>('.chip')!.style.background = c;
                return label;
            }),
            custom,
        );
    };

    const sizes = h('div', { class: 'group sizes', role: 'radiogroup', 'aria-label': 'Size' });
    SIZE_NAMES.forEach((name, i) => {
        const input = h('input', { type: 'radio', name: 'size', value: String(i), 'aria-label': name });
        input.addEventListener('change', () => board.setSize(i));
        const dot = h('span', { class: 'dot' });
        dot.style.setProperty('--d', `${6 + i * 5}px`);
        sizes.append(h('label', { class: 'size', title: name }, input, dot));
    });

    const undo = h('button', { type: 'button', class: 'tool', title: 'Undo (Ctrl+Z)', 'aria-label': 'Undo' }, svgIcon('undo'));
    const redo = h('button', { type: 'button', class: 'tool', title: 'Redo (Ctrl+Shift+Z)', 'aria-label': 'Redo' }, svgIcon('redo'));
    undo.addEventListener('click', () => doc.undo());
    redo.addEventListener('click', () => doc.redo());
    const history = h('div', { class: 'group', role: 'group', 'aria-label': 'History' }, undo, redo);
    el.toolbar.replaceChildren(tools, swatches, sizes, history);

    // ---- view controls
    const zoomOut = h('button', { type: 'button', class: 'tool', title: 'Zoom out (-)', 'aria-label': 'Zoom out' }, svgIcon('minus'));
    const zoomIn = h('button', { type: 'button', class: 'tool', title: 'Zoom in (+)', 'aria-label': 'Zoom in' }, svgIcon('plus'));
    const zoomLabel = h('button', { type: 'button', class: 'zoom', title: 'Reset to 100% (Ctrl+0)', 'aria-label': 'Reset zoom to 100%' });
    const fit = h('button', { type: 'button', class: 'tool', title: 'Fit everything (Shift+1)', 'aria-label': 'Fit everything' }, svgIcon('fit'));
    zoomOut.addEventListener('click', () => board.zoomBy(0.8));
    zoomIn.addEventListener('click', () => board.zoomBy(1.25));
    zoomLabel.addEventListener('click', () => board.resetZoom());
    fit.addEventListener('click', () => board.fitContent());
    el.viewbar.replaceChildren(zoomOut, zoomLabel, zoomIn, fit);

    // ---- actions in the top bar
    const png = h('button', { type: 'button', class: 'btn' }, svgIcon('download'), h('span', {}, 'PNG'));
    const svg = h('button', { type: 'button', class: 'btn' }, svgIcon('download'), h('span', {}, 'SVG'));
    const link = h('button', { type: 'button', class: 'btn' }, svgIcon('link'), h('span', {}, 'Copy link'));
    link.setAttribute('aria-label', 'Copy link to this board');
    png.setAttribute('aria-label', 'Export as PNG');
    svg.setAttribute('aria-label', 'Export as SVG');
    png.addEventListener('click', handlers.exportPng);
    svg.addEventListener('click', handlers.exportSvg);
    link.addEventListener('click', handlers.copyLink);
    el.actions.replaceChildren(link, png, svg);

    el.nameInput.maxLength = MAX_NAME;
    el.nameInput.addEventListener('change', () => {
        const name = el.nameInput.value.trim();
        if (name) handlers.rename(name);
    });

    // ---- toasts and announcements
    const toast = (text: string) => {
        const node = h('p', { class: 'toast' }, text);
        el.toasts.append(node);
        setTimeout(() => node.remove(), 4500);
    };
    let announceTimer: ReturnType<typeof setTimeout> | undefined;
    const announce = (text: string) => {
        // Clearing first makes screen readers re-announce identical text.
        el.announcer.textContent = '';
        clearTimeout(announceTimer);
        announceTimer = setTimeout(() => (el.announcer.textContent = text), 50);
    };

    // ---- sync
    let statusDetail: string | undefined;
    let lastUsers = new Map<string, string>();
    let wasSynced = false;
    let seenBoard = -1;
    let describeTimer: ReturnType<typeof setTimeout> | undefined;

    const describe = () => {
        let strokes = 0;
        let shapes = 0;
        const texts: string[] = [];
        doc.forEachVisible(({ item }) => {
            if (item.t === 'stroke') strokes++;
            else if (item.t === 'shape') shapes++;
            else texts.push(item.text);
        });
        const people = doc.users.size;
        el.top.setAttribute(
            'aria-label',
            `Whiteboard drawing surface. ${strokes} strokes, ${shapes} shapes, ${texts.length} text items. ${people} ${people === 1 ? 'person' : 'people'} on this board.`,
        );
        el.notes.replaceChildren(...texts.slice(0, 100).map((t) => h('li', {}, t.slice(0, 300))));
    };

    const sync = () => {
        for (const [id, b] of toolButtons) b.setAttribute('aria-pressed', String(board.tool === id));
        rebuildSwatchesIfNeeded();
        for (const input of sizes.querySelectorAll<HTMLInputElement>('input')) input.checked = Number(input.value) === board.sizeStep;
        undo.disabled = !doc.canUndo;
        redo.disabled = !doc.canRedo;
        zoomLabel.textContent = `${Math.round(board.camera.zoom * 100)}%`;

        // status line
        const waiting = doc.pendingCount;
        const people = doc.users.size;
        const text =
            conn.status === 'online'
                ? waiting > 0
                    ? `Syncing ${waiting} ${waiting === 1 ? 'change' : 'changes'}…`
                    : `Connected · ${people} ${people === 1 ? 'person' : 'people'}`
                : conn.status === 'closed'
                  ? (statusDetail ?? 'Disconnected')
                  : waiting > 0
                    ? `Offline · ${waiting} ${waiting === 1 ? 'change' : 'changes'} will sync when you are back`
                    : 'Connecting…';
        if (el.status.textContent !== text) el.status.textContent = text;
        el.status.dataset.state = conn.status;

        // people
        el.people.replaceChildren(
            ...[...doc.users.values()].map((u) => {
                const dot = h('span', { class: 'dot' });
                dot.style.background = u.color;
                return h('li', { class: 'person' }, dot, h('span', {}, u.id === doc.me?.id ? `${u.name} (you)` : u.name));
            }),
        );
        if (document.activeElement !== el.nameInput && doc.me && el.nameInput.value !== doc.me.name) el.nameInput.value = doc.me.name;

        // join and leave announcements, but not the initial crowd after (re)connecting
        const now = new Map([...doc.users].map(([id, u]) => [id, u.name]));
        if (doc.synced && wasSynced) {
            for (const [id, name] of now) if (!lastUsers.has(id)) announce(`${name} joined`);
            for (const [id, name] of lastUsers) if (!now.has(id)) announce(`${name} left`);
        }
        lastUsers = now;
        wasSynced = doc.synced;

        if (doc.boardVersion !== seenBoard) {
            seenBoard = doc.boardVersion;
            clearTimeout(describeTimer);
            describeTimer = setTimeout(describe, 500);
        }
    };

    let lastPalette = '';
    const rebuildSwatchesIfNeeded = () => {
        const key = `${board.tool === 'sticky' ? 's' : 'i'}:${board.activeColor()}`;
        if (key === lastPalette) return;
        lastPalette = key;
        custom.value = /^#[0-9a-f]{6}$/i.test(board.activeColor()) ? board.activeColor() : '#000000';
        rebuildSwatches();
    };

    return {
        sync,
        toast,
        announce,
        setDetail(text) {
            statusDetail = text;
            sync();
        },
    };
}

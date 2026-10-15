import { MAX_IDS_PER_OP, MAX_POINTS_PER_LIVE_CHUNK, MAX_STROKE_POINTS, MAX_TEXT } from '../shared/limits.ts';
import type { ClientMsg, ShapeItem, StrokeItem, TextItem } from '../shared/types.ts';
import { type Bounds, Camera } from './camera.ts';
import type { Doc } from './doc.ts';
import { sceneBounds } from './export.ts';
import { FONT_FAMILY, LINE_HEIGHT, type Measure, distanceTo, itemBounds, layoutText, shift, touches } from './geometry.ts';
import { type CursorView, type Moving, type Preview, drawBoard, drawOverlay } from './render.ts';
import { COLORS, HIGHLIGHTER_DEFAULT, SHAPE_TOOLS, STICKY_COLORS, TOOLS, type ToolId, shapeKind, sizeFor } from './tools.ts';

export interface BoardElements {
    stage: HTMLElement;
    base: HTMLCanvasElement;
    top: HTMLCanvasElement;
    editor: HTMLTextAreaElement;
}

interface Editing {
    /** The item being edited, or `null` for a new one. */
    id: string | null;
    x: number;
    y: number;
    sticky: boolean;
    color: string;
    size: number;
    original: string;
}

type Gesture =
    | { kind: 'pan'; id: number; x: number; y: number }
    | { kind: 'pinch'; dist: number; cx: number; cy: number }
    | {
        kind: 'stroke';
        id: number;
        sid: string;
        pts: number[];
        pr: number[] | null;
        sent: number;
        color: string;
        size: number;
        hl: boolean;
    }
    | { kind: 'shape'; id: number; shape: ShapeItem; sx: number; sy: number }
    | { kind: 'erase'; id: number; hit: Set<string>; x: number; y: number }
    | { kind: 'move'; id: number; itemId: string; x: number; y: number; dx: number; dy: number }
    | { kind: 'click'; id: number; sx: number; sy: number };

/** Round to 0.1 world units: invisible on screen, and it keeps messages small. */
const q = (v: number) => Math.round(v * 10) / 10;
const CURSOR_MS = 33;
const CURSOR_FADE_AFTER = 8000;
const CURSOR_GONE_AFTER = 11_000;
const LIVE_STALE_MS = 6000;
const ERASER_PX = 10;
const PICK_PX = 6;
const CLICK_SLOP_PX = 5;

/**
 * The drawing surface: two canvases (committed items below, previews and cursors above), pointer
 * and keyboard input, and the tools. It never talks to the network itself; it reads a `Doc` and
 * hands it ops, and sends ephemeral messages (live strokes, cursor) through `sendLive`.
 */
export class Whiteboard {
    readonly camera = new Camera();
    tool: ToolId = 'pen';
    color: string = COLORS[0];
    stickyColor: string = STICKY_COLORS[0];
    highlightColor: string = HIGHLIGHTER_DEFAULT;
    sizeStep = 1;
    selected: string | null = null;
    /** Called when something the toolbar shows changed (tool, zoom, selection). */
    onState: () => void = () => {};
    /** Called with a short message the UI should announce or toast. */
    onNotice: (text: string) => void = () => {};

    private readonly doc: Doc;
    private readonly el: BoardElements;
    private readonly sendLive: (msg: ClientMsg) => void;
    private readonly newId: () => string;
    private readonly baseCtx: CanvasRenderingContext2D;
    private readonly topCtx: CanvasRenderingContext2D;
    private readonly measureCtx: CanvasRenderingContext2D;
    private width = 0;
    private height = 0;
    private dpr = 1;
    private raf = 0;
    private baseDirty = true;
    private seenBoardVersion = -1;
    private idleTimer: ReturnType<typeof setTimeout> | undefined;
    private readonly pointers = new Map<number, { x: number; y: number }>();
    private gesture: Gesture | null = null;
    private spaceDown = false;
    private penDown = false;
    private preview: Preview | null = null;
    private erasing: ReadonlySet<string> = new Set();
    private moving: Moving | null = null;
    private editing: Editing | null = null;
    private lastCursorSent = 0;
    private liveTimer: ReturnType<typeof setTimeout> | undefined;
    private readonly widths = new Map<string, number>();
    readonly measure: Measure;

    constructor(el: BoardElements, doc: Doc, sendLive: (msg: ClientMsg) => void, newId: () => string) {
        this.el = el;
        this.doc = doc;
        this.sendLive = sendLive;
        this.newId = newId;
        this.baseCtx = el.base.getContext('2d', { alpha: false })!;
        this.topCtx = el.top.getContext('2d')!;
        this.measureCtx = document.createElement('canvas').getContext('2d')!;
        this.measure = (text, fs) => {
            const key = `${fs}|${text}`;
            let w = this.widths.get(key);
            if (w === undefined) {
                this.measureCtx.font = `${fs}px ${FONT_FAMILY}`;
                w = this.measureCtx.measureText(text).width;
                if (this.widths.size > 5000) this.widths.clear();
                this.widths.set(key, w);
            }
            return w;
        };
        doc.onChange = () => this.invalidate(false);

        const top = el.top;
        top.addEventListener('pointerdown', this.onPointerDown);
        top.addEventListener('pointermove', this.onPointerMove);
        top.addEventListener('pointerup', this.onPointerUp);
        top.addEventListener('pointercancel', this.onPointerCancel);
        top.addEventListener('lostpointercapture', this.onPointerCancel);
        top.addEventListener('dblclick', this.onDoubleClick);
        top.addEventListener('contextmenu', (e) => e.preventDefault());
        top.addEventListener('wheel', this.onWheel, { passive: false });
        document.addEventListener('keydown', this.onKeyDown);
        document.addEventListener('keyup', this.onKeyUp);
        el.editor.addEventListener('blur', () => this.commitEditor());
        el.editor.addEventListener('keydown', this.onEditorKey);
        el.editor.addEventListener('input', () => this.sizeEditor());
        new ResizeObserver(() => this.resize()).observe(el.stage);
        this.resize();
        // Start with the origin in the middle of the screen.
        this.camera.x = -this.width / 2;
        this.camera.y = -this.height / 2;
        this.setTool('pen');
    }

    // ---------------------------------------------------------------- state the toolbar drives

    setTool(tool: ToolId): void {
        this.commitEditor();
        this.cancelGesture();
        this.tool = tool;
        if (tool !== 'select') this.selected = null;
        this.preview = null;
        this.applyCursor();
        this.invalidate(false);
        this.onState();
    }

    private applyCursor(): void {
        const t = this.tool;
        this.el.top.style.cursor =
            t === 'hand' ? 'grab' : t === 'select' ? 'default' : t === 'text' || t === 'sticky' ? 'text' : t === 'eraser' ? 'none' : 'crosshair';
    }

    setColor(color: string): void {
        if (this.tool === 'sticky') this.stickyColor = color;
        else if (this.tool === 'highlighter') this.highlightColor = color;
        else this.color = color;
        this.onState();
    }

    setSize(step: number): void {
        this.sizeStep = step;
        this.onState();
    }

    activeColor(): string {
        return this.tool === 'sticky' ? this.stickyColor : this.tool === 'highlighter' ? this.highlightColor : this.color;
    }

    zoomBy(factor: number): void {
        this.camera.zoomAt(this.width / 2, this.height / 2, factor);
        this.viewChanged();
    }

    resetZoom(): void {
        this.camera.zoomTo(this.width / 2, this.height / 2, 1);
        this.viewChanged();
    }

    fitContent(): void {
        const bounds = sceneBounds(this.doc.visible(), this.measure);
        if (bounds) this.camera.fit(bounds, this.width, this.height);
        else {
            this.camera.zoom = 1;
            this.camera.x = -this.width / 2;
            this.camera.y = -this.height / 2;
        }
        this.viewChanged();
    }

    /** Where the pointer or the middle of the screen is, for placing a keyboard-created note. */
    centerWorld(): [number, number] {
        return this.camera.toWorld(this.width / 2, this.height / 2);
    }

    /** Create a text item or note in the middle of the screen (the keyboard route to text). */
    newTextAtCenter(sticky: boolean): void {
        const [x, y] = this.centerWorld();
        this.openEditor({ id: null, x, y, sticky, color: sticky ? this.stickyColor : this.color, size: sizeFor(sticky ? 'sticky' : 'text', this.sizeStep), original: '' });
    }

    deleteSelected(): void {
        if (!this.selected) return;
        this.doc.submit({ k: 'del', ids: [this.selected] });
        this.selected = null;
        this.onState();
        this.invalidate(true);
    }

    destroy(): void {
        cancelAnimationFrame(this.raf);
        clearTimeout(this.liveTimer);
        clearTimeout(this.idleTimer);
        document.removeEventListener('keydown', this.onKeyDown);
        document.removeEventListener('keyup', this.onKeyUp);
    }

    // ---------------------------------------------------------------- drawing loop

    invalidate(base: boolean): void {
        if (base) this.baseDirty = true;
        if (!this.raf) this.raf = requestAnimationFrame(() => this.paint());
    }

    private viewChanged(): void {
        this.invalidate(true);
        this.onState();
    }

    private resize(): void {
        const rect = this.el.stage.getBoundingClientRect();
        this.width = Math.max(1, Math.round(rect.width));
        this.height = Math.max(1, Math.round(rect.height));
        this.dpr = Math.min(2, window.devicePixelRatio || 1);
        for (const c of [this.el.base, this.el.top]) {
            c.width = Math.round(this.width * this.dpr);
            c.height = Math.round(this.height * this.dpr);
        }
        if (this.editing) this.placeEditor();
        this.invalidate(true);
    }

    private paint(): void {
        this.raf = 0;
        const now = performance.now();
        if (this.doc.boardVersion !== this.seenBoardVersion) {
            this.seenBoardVersion = this.doc.boardVersion;
            this.baseDirty = true;
        }
        const selected = this.selected ? this.doc.find(this.selected) : undefined;
        if (this.selected && !selected) {
            this.selected = null;
            this.onState();
        }
        if (this.baseDirty) {
            this.baseDirty = false;
            drawBoard(this.baseCtx, this.doc.visible(), this.camera, this.width, this.height, this.dpr, this.measure, {
                dim: this.erasing,
                moving: this.moving,
                hide: this.editing?.id ? new Set([this.editing.id]) : undefined,
            });
        }
        for (const [key, live] of this.doc.live) if (now - live.at > LIVE_STALE_MS) this.doc.live.delete(key);
        const cursors: CursorView[] = [];
        for (const [id, c] of this.doc.cursors) {
            const user = this.doc.users.get(id);
            const age = now - c.at;
            if (!user || id === this.doc.me?.id || age > CURSOR_GONE_AFTER) continue;
            const alpha = age < CURSOR_FADE_AFTER ? 1 : 1 - (age - CURSOR_FADE_AFTER) / (CURSOR_GONE_AFTER - CURSOR_FADE_AFTER);
            cursors.push({ x: c.x, y: c.y, name: user.name, color: user.color, alpha });
        }
        let selection: Bounds | null = null;
        if (selected) {
            const pad = 4 / this.camera.zoom;
            const m = this.moving?.ids.has(selected.item.id) ? this.moving : null;
            const b = shift(itemBounds(selected.item, this.measure), selected.dx + (m?.dx ?? 0), selected.dy + (m?.dy ?? 0));
            selection = { minX: b.minX - pad, minY: b.minY - pad, maxX: b.maxX + pad, maxY: b.maxY + pad };
        }
        drawOverlay(this.topCtx, {
            camera: this.camera,
            width: this.width,
            height: this.height,
            dpr: this.dpr,
            live: this.doc.live.values(),
            cursors,
            preview: this.preview,
            selection,
        });
        // Cursors fade and stale live strokes expire with time, so check back while any exist.
        clearTimeout(this.idleTimer);
        if (cursors.length > 0 || this.doc.live.size > 0) this.idleTimer = setTimeout(() => this.invalidate(false), 400);
    }

    // ---------------------------------------------------------------- pointer input

    private local(e: PointerEvent): { x: number; y: number } {
        const r = this.el.top.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    private readonly onPointerDown = (e: PointerEvent): void => {
        if (e.button !== 0 && e.button !== 1) return;
        if (e.pointerType === 'touch' && this.penDown) return; // palm rejection while a pen is in use
        this.el.top.focus({ preventScroll: true });
        const p = this.local(e);
        this.pointers.set(e.pointerId, p);
        this.el.top.setPointerCapture(e.pointerId);
        if (e.pointerType === 'pen') this.penDown = true;

        if (this.pointers.size === 2 && e.pointerType === 'touch') {
            this.cancelGesture(); // a second finger means zoom, not draw
            const [a, b] = [...this.pointers.values()] as [{ x: number; y: number }, { x: number; y: number }];
            this.gesture = { kind: 'pinch', dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2 };
            return;
        }
        if (this.gesture) return;
        e.preventDefault();

        const [wx, wy] = this.camera.toWorld(p.x, p.y);
        if (e.button === 1 || this.tool === 'hand' || this.spaceDown) {
            this.gesture = { kind: 'pan', id: e.pointerId, x: p.x, y: p.y };
            this.el.top.style.cursor = 'grabbing';
            return;
        }
        switch (this.tool) {
            case 'pen':
            case 'highlighter': {
                const pressure = e.pointerType === 'pen';
                this.gesture = {
                    kind: 'stroke',
                    id: e.pointerId,
                    sid: this.newId(),
                    pts: [q(wx), q(wy)],
                    pr: pressure ? [round2(e.pressure || 0.5)] : null,
                    sent: 0,
                    color: this.activeColor(),
                    size: sizeFor(this.tool, this.sizeStep),
                    hl: this.tool === 'highlighter',
                };
                this.strokePreview(this.gesture);
                this.scheduleLive();
                break;
            }
            case 'eraser':
                this.gesture = { kind: 'erase', id: e.pointerId, hit: new Set(), x: wx, y: wy };
                this.eraseTo(this.gesture, wx, wy);
                break;
            case 'line':
            case 'arrow':
            case 'rect':
            case 'ellipse':
                this.gesture = {
                    kind: 'shape',
                    id: e.pointerId,
                    sx: p.x,
                    sy: p.y,
                    shape: {
                        t: 'shape',
                        id: this.newId(),
                        author: '',
                        k: shapeKind(this.tool),
                        color: this.color,
                        size: sizeFor(this.tool, this.sizeStep),
                        x1: q(wx),
                        y1: q(wy),
                        x2: q(wx),
                        y2: q(wy),
                    },
                };
                break;
            case 'text':
            case 'sticky':
                this.gesture = { kind: 'click', id: e.pointerId, sx: p.x, sy: p.y };
                break;
            case 'select': {
                const hit = this.pick(wx, wy);
                this.selected = hit?.item.id ?? null;
                if (hit) this.gesture = { kind: 'move', id: e.pointerId, itemId: hit.item.id, x: wx, y: wy, dx: 0, dy: 0 };
                this.onState();
                this.invalidate(false);
                break;
            }
        }
    };

    private readonly onPointerMove = (e: PointerEvent): void => {
        const p = this.local(e);
        if (this.pointers.has(e.pointerId)) this.pointers.set(e.pointerId, p);
        const [wx, wy] = this.camera.toWorld(p.x, p.y);
        this.sendCursor(wx, wy);
        const g = this.gesture;
        if (!g) {
            if (this.tool === 'eraser' && e.pointerType !== 'touch') {
                this.preview = { kind: 'eraser', x: wx, y: wy, r: ERASER_PX / this.camera.zoom };
                this.invalidate(false);
            }
            return;
        }
        switch (g.kind) {
            case 'pinch': {
                const pts = [...this.pointers.values()];
                if (pts.length < 2) return;
                const [a, b] = pts as [{ x: number; y: number }, { x: number; y: number }];
                const dist = Math.hypot(a.x - b.x, a.y - b.y) || 1;
                const cx = (a.x + b.x) / 2;
                const cy = (a.y + b.y) / 2;
                this.camera.zoomAt(cx, cy, dist / g.dist);
                this.camera.panBy(cx - g.cx, cy - g.cy);
                g.dist = dist;
                g.cx = cx;
                g.cy = cy;
                this.viewChanged();
                return;
            }
            case 'pan':
                if (e.pointerId !== g.id) return;
                this.camera.panBy(p.x - g.x, p.y - g.y);
                g.x = p.x;
                g.y = p.y;
                this.viewChanged();
                return;
            case 'stroke': {
                if (e.pointerId !== g.id) return;
                const events = e.getCoalescedEvents?.() ?? [];
                for (const ce of events.length > 0 ? events : [e]) {
                    const cp = this.local(ce);
                    const [cx, cy] = this.camera.toWorld(cp.x, cp.y);
                    this.addStrokePoint(g, cx, cy, ce.pointerType === 'pen' ? ce.pressure : null);
                    if (this.gesture !== g) return; // the stroke was split and replaced
                }
                this.strokePreview(g);
                this.scheduleLive();
                return;
            }
            case 'shape': {
                if (e.pointerId !== g.id) return;
                this.updateShape(g, wx, wy, e.shiftKey);
                this.preview = { kind: 'shape', shape: g.shape };
                this.invalidate(false);
                return;
            }
            case 'erase':
                if (e.pointerId !== g.id) return;
                this.eraseTo(g, wx, wy);
                this.preview = { kind: 'eraser', x: wx, y: wy, r: ERASER_PX / this.camera.zoom };
                return;
            case 'move':
                if (e.pointerId !== g.id) return;
                g.dx = q(wx - g.x);
                g.dy = q(wy - g.y);
                this.moving = { ids: new Set([g.itemId]), dx: g.dx, dy: g.dy };
                this.invalidate(true);
                return;
            case 'click':
                if (e.pointerId === g.id && Math.hypot(p.x - g.sx, p.y - g.sy) > CLICK_SLOP_PX) this.gesture = null;
                return;
        }
    };

    private readonly onPointerUp = (e: PointerEvent): void => {
        const p = this.local(e);
        this.pointers.delete(e.pointerId);
        if (e.pointerType === 'pen') this.penDown = false;
        const g = this.gesture;
        if (!g) return;
        if (g.kind === 'pinch') {
            if (this.pointers.size < 2) this.gesture = null;
            return;
        }
        if (e.pointerId !== g.id) return;
        this.gesture = null;
        const [wx, wy] = this.camera.toWorld(p.x, p.y);
        switch (g.kind) {
            case 'pan':
                this.applyCursor();
                break;
            case 'stroke':
                this.finishStroke(g);
                break;
            case 'shape': {
                this.updateShape(g, wx, wy, e.shiftKey);
                this.preview = null;
                if (Math.hypot(p.x - g.sx, p.y - g.sy) >= CLICK_SLOP_PX) this.doc.submit({ k: 'add', item: { ...g.shape, author: this.doc.me?.id ?? '' } });
                else this.invalidate(false);
                break;
            }
            case 'erase': {
                const ids = [...g.hit];
                this.erasing = new Set();
                this.preview = null;
                for (let i = 0; i < ids.length; i += MAX_IDS_PER_OP) {
                    this.doc.submit({ k: 'del', ids: ids.slice(i, i + MAX_IDS_PER_OP) });
                }
                this.invalidate(true);
                break;
            }
            case 'move':
                this.moving = null;
                if (Math.hypot(g.dx, g.dy) * this.camera.zoom >= 2) this.doc.submit({ k: 'mv', ids: [g.itemId], dx: g.dx, dy: g.dy });
                else this.invalidate(true);
                break;
            case 'click':
                if (this.tool === 'text' || this.tool === 'sticky') {
                    const sticky = this.tool === 'sticky';
                    this.openEditor({ id: null, x: wx, y: wy, sticky, color: sticky ? this.stickyColor : this.color, size: sizeFor(this.tool, this.sizeStep), original: '' });
                }
                break;
        }
    };

    /** The pointer was taken away (a scroll gesture, an alert): throw the half-made thing away. */
    private readonly onPointerCancel = (e: PointerEvent): void => {
        this.pointers.delete(e.pointerId);
        if (e.pointerType === 'pen') this.penDown = false;
        if (this.gesture && (this.gesture.kind === 'pinch' ? this.pointers.size < 2 : this.gesture.id === e.pointerId)) {
            this.cancelGesture();
        }
    };

    private cancelGesture(): void {
        const g = this.gesture;
        this.gesture = null;
        if (g?.kind === 'stroke' && g.sent > 0) this.sendLive({ t: 'liveEnd', sid: g.sid });
        clearTimeout(this.liveTimer);
        this.preview = null;
        this.erasing = new Set();
        this.moving = null;
        this.invalidate(true);
    }

    private readonly onDoubleClick = (e: MouseEvent): void => {
        if (this.tool !== 'select') return;
        const r = this.el.top.getBoundingClientRect();
        const [wx, wy] = this.camera.toWorld(e.clientX - r.left, e.clientY - r.top);
        const hit = this.pick(wx, wy);
        if (!hit || hit.item.t !== 'text') return;
        this.cancelGesture();
        this.openEditor({ id: hit.item.id, x: hit.item.x + hit.dx, y: hit.item.y + hit.dy, sticky: hit.item.sticky, color: hit.item.color, size: hit.item.size, original: hit.item.text });
    };

    private readonly onWheel = (e: WheelEvent): void => {
        e.preventDefault();
        const r = this.el.top.getBoundingClientRect();
        if (e.ctrlKey || e.metaKey) {
            // Pinch on a trackpad arrives as ctrl+wheel.
            this.camera.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.01));
        } else {
            const scale = e.deltaMode === 1 ? 16 : 1;
            this.camera.panBy(-e.deltaX * scale, -e.deltaY * scale);
        }
        this.viewChanged();
    };

    // ---------------------------------------------------------------- tools

    private pick(wx: number, wy: number) {
        const r = PICK_PX / this.camera.zoom;
        const items = this.doc.visible();
        for (let i = items.length - 1; i >= 0; i--) {
            const v = items[i]!;
            if (distanceTo(v.item, wx - v.dx, wy - v.dy, this.measure) <= r) return v;
        }
        return undefined;
    }

    private addStrokePoint(g: Extract<Gesture, { kind: 'stroke' }>, wx: number, wy: number, pressure: number | null): void {
        const n = g.pts.length;
        const minGap = 1.2 / this.camera.zoom; // skip points closer than ~1 screen pixel
        if (Math.hypot(wx - g.pts[n - 2]!, wy - g.pts[n - 1]!) < minGap) return;
        g.pts.push(q(wx), q(wy));
        g.pr?.push(round2(pressure ?? 0.5));
        if (g.pts.length / 2 >= MAX_STROKE_POINTS) {
            // Keep the message size bounded: end this stroke here and carry on in a new one.
            this.finishStroke(g);
            this.gesture = {
                ...g,
                sid: this.newId(),
                pts: [q(wx), q(wy)],
                pr: g.pr ? [round2(pressure ?? 0.5)] : null,
                sent: 0,
            };
        }
    }

    private strokePreview(g: Extract<Gesture, { kind: 'stroke' }>): void {
        this.preview = { kind: 'stroke', pts: g.pts, pr: g.pr, color: g.color, size: g.size, hl: g.hl };
        this.invalidate(false);
    }

    private finishStroke(g: Extract<Gesture, { kind: 'stroke' }>): void {
        clearTimeout(this.liveTimer);
        this.flushLive(g);
        this.preview = null;
        const item: StrokeItem = {
            t: 'stroke',
            id: g.sid,
            author: this.doc.me?.id ?? '',
            color: g.color,
            size: g.size,
            pts: g.pts,
            ...(g.pr ? { pr: g.pr } : {}),
            ...(g.hl ? { hl: true } : {}),
        };
        this.doc.submit({ k: 'add', item });
    }

    /** Send new points of the stroke being drawn, a few times a second, so others see it grow. */
    private scheduleLive(): void {
        if (this.liveTimer) return;
        this.liveTimer = setTimeout(() => {
            this.liveTimer = undefined;
            if (this.gesture?.kind === 'stroke') this.flushLive(this.gesture);
        }, CURSOR_MS);
    }

    private flushLive(g: Extract<Gesture, { kind: 'stroke' }>): void {
        while (g.sent * 2 < g.pts.length) {
            const from = g.sent * 2;
            const to = Math.min(g.pts.length, from + MAX_POINTS_PER_LIVE_CHUNK * 2);
            this.sendLive({
                t: 'live',
                sid: g.sid,
                pts: g.pts.slice(from, to),
                ...(g.pr ? { pr: g.pr.slice(from / 2, to / 2) } : {}),
                ...(from === 0 ? { start: { color: g.color, size: g.size, ...(g.hl ? { hl: true } : {}) } } : {}),
            });
            g.sent = to / 2;
        }
    }

    private updateShape(g: Extract<Gesture, { kind: 'shape' }>, wx: number, wy: number, constrain: boolean): void {
        let x2 = wx;
        let y2 = wy;
        const s = g.shape;
        if (constrain) {
            const dx = wx - s.x1;
            const dy = wy - s.y1;
            if (s.k === 'rect' || s.k === 'ellipse') {
                const m = Math.max(Math.abs(dx), Math.abs(dy));
                x2 = s.x1 + Math.sign(dx || 1) * m;
                y2 = s.y1 + Math.sign(dy || 1) * m;
            } else {
                const len = Math.hypot(dx, dy);
                const a = Math.round(Math.atan2(dy, dx) / (Math.PI / 4)) * (Math.PI / 4);
                x2 = s.x1 + len * Math.cos(a);
                y2 = s.y1 + len * Math.sin(a);
            }
        }
        g.shape = { ...s, x2: q(x2), y2: q(y2) };
    }

    private eraseTo(g: Extract<Gesture, { kind: 'erase' }>, wx: number, wy: number): void {
        const r = ERASER_PX / this.camera.zoom;
        const minX = Math.min(g.x, wx) - r;
        const maxX = Math.max(g.x, wx) + r;
        const minY = Math.min(g.y, wy) - r;
        const maxY = Math.max(g.y, wy) + r;
        let added = false;
        this.doc.forEachVisible((v) => {
            if (g.hit.has(v.item.id)) return;
            const b = shift(itemBounds(v.item, this.measure), v.dx, v.dy);
            if (b.maxX < minX || b.minX > maxX || b.maxY < minY || b.minY > maxY) return;
            if (touches(v.item, g.x - v.dx, g.y - v.dy, wx - v.dx, wy - v.dy, r, this.measure)) {
                g.hit.add(v.item.id);
                added = true;
            }
        });
        g.x = wx;
        g.y = wy;
        if (added) {
            this.erasing = new Set(g.hit);
            this.invalidate(true);
        } else this.invalidate(false);
    }

    private sendCursor(wx: number, wy: number): void {
        const now = performance.now();
        if (now - this.lastCursorSent < CURSOR_MS) return;
        this.lastCursorSent = now;
        this.sendLive({ t: 'cur', x: q(wx), y: q(wy) });
    }

    // ---------------------------------------------------------------- text editing

    private openEditor(editing: Editing): void {
        this.commitEditor();
        this.editing = editing;
        const ed = this.el.editor;
        ed.value = editing.original;
        ed.maxLength = MAX_TEXT;
        ed.hidden = false;
        ed.setAttribute('aria-label', editing.sticky ? 'Sticky note text' : 'Text');
        this.placeEditor();
        this.invalidate(true); // hides the item being edited on the canvas
        ed.focus({ preventScroll: true });
        ed.select();
    }

    private placeEditor(): void {
        const e = this.editing;
        if (!e) return;
        const ed = this.el.editor;
        const z = this.camera.zoom;
        const [sx, sy] = this.camera.toScreen(e.x, e.y);
        ed.style.left = `${sx}px`;
        ed.style.top = `${sy}px`;
        ed.style.fontSize = `${e.size * z}px`;
        ed.style.lineHeight = String(LINE_HEIGHT);
        ed.style.color = e.sticky ? '#1f2328' : e.color;
        ed.style.background = e.sticky ? e.color : 'transparent';
        ed.style.padding = e.sticky ? `${e.size * 0.5 * z}px` : '0';
        ed.classList.toggle('sticky', e.sticky);
        this.sizeEditor();
    }

    private sizeEditor(): void {
        const e = this.editing;
        if (!e) return;
        const ed = this.el.editor;
        const z = this.camera.zoom;
        const probe: TextItem = { t: 'text', id: 'probe', author: '', color: e.color, size: e.size, x: e.x, y: e.y, text: ed.value || ' ', sticky: e.sticky };
        const l = layoutText(probe, this.measure);
        const pad = e.sticky ? e.size * 0.5 : 0;
        ed.style.width = `${(e.sticky ? l.w : Math.max(l.w + e.size, e.size * 4)) * z + (e.sticky ? 0 : 4)}px`;
        ed.style.boxSizing = 'border-box';
        ed.style.height = 'auto';
        ed.style.height = `${Math.max(ed.scrollHeight, (e.sticky ? l.h : l.h + pad) * z)}px`;
    }

    private commitEditor(): void {
        const e = this.editing;
        if (!e) return;
        this.editing = null;
        const text = this.el.editor.value.replace(/\s+$/, '');
        this.el.editor.hidden = true;
        this.el.top.focus({ preventScroll: true });
        if (e.id) {
            if (text && text !== e.original) this.doc.submit({ k: 'edit', id: e.id, text });
        } else if (text) {
            const item: TextItem = { t: 'text', id: this.newId(), author: this.doc.me?.id ?? '', color: e.color, size: e.size, x: q(e.x), y: q(e.y), text, sticky: e.sticky };
            this.doc.submit({ k: 'add', item });
        }
        this.invalidate(true);
    }

    private cancelEditor(): void {
        if (!this.editing) return;
        this.editing = null;
        this.el.editor.hidden = true;
        this.el.top.focus({ preventScroll: true });
        this.invalidate(true);
    }

    private readonly onEditorKey = (e: KeyboardEvent): void => {
        if (e.key === 'Escape') {
            e.preventDefault();
            this.cancelEditor();
        } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            this.commitEditor();
        }
        e.stopPropagation(); // typing must never trigger a tool shortcut
    };

    // ---------------------------------------------------------------- keyboard

    private readonly onKeyUp = (e: KeyboardEvent): void => {
        if (e.code === 'Space') this.spaceDown = false;
    };

    private readonly onKeyDown = (e: KeyboardEvent): void => {
        const t = e.target as HTMLElement | null;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
        const mod = e.ctrlKey || e.metaKey;
        if (e.code === 'Space' && !mod) {
            if (t?.tagName === 'BUTTON') return; // Space activates the focused button
            this.spaceDown = true;
            e.preventDefault();
            return;
        }
        if (mod && e.key.toLowerCase() === 'z') {
            e.preventDefault();
            if (e.shiftKey) this.doc.redo();
            else this.doc.undo();
        } else if (mod && e.key.toLowerCase() === 'y') {
            e.preventDefault();
            this.doc.redo();
        } else if (mod && e.key === '0') {
            e.preventDefault();
            this.resetZoom();
        } else if (mod) {
            return;
        } else if (e.key === 'Escape') {
            this.cancelGesture();
            this.selected = null;
            this.onState();
            this.invalidate(false);
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (this.selected) {
                e.preventDefault();
                this.deleteSelected();
            }
        } else if (e.key === '+' || e.key === '=') {
            this.zoomBy(1.25);
        } else if (e.key === '-' || e.key === '_') {
            this.zoomBy(0.8);
        } else if (e.key === '!' || (e.shiftKey && e.code === 'Digit1')) {
            this.fitContent();
        } else if (e.key.startsWith('Arrow') && (t === this.el.top || t === document.body)) {
            e.preventDefault();
            const step = (e.shiftKey ? 400 : 80) / this.camera.zoom;
            this.camera.x += e.key === 'ArrowRight' ? step : e.key === 'ArrowLeft' ? -step : 0;
            this.camera.y += e.key === 'ArrowDown' ? step : e.key === 'ArrowUp' ? -step : 0;
            this.viewChanged();
        } else if (e.key === 'Enter' && t === this.el.top && (this.tool === 'text' || this.tool === 'sticky')) {
            e.preventDefault();
            this.newTextAtCenter(this.tool === 'sticky');
        } else if (!e.altKey && t?.tagName !== 'BUTTON') {
            const tool = TOOLS.find((x) => x.key === e.key.toLowerCase());
            if (tool) {
                e.preventDefault();
                this.setTool(tool.id);
            }
        }
    };

    get isDrawing(): boolean {
        return this.gesture !== null;
    }

    get shapeToolActive(): boolean {
        return SHAPE_TOOLS.has(this.tool);
    }
}

const round2 = (v: number) => Math.round(v * 100) / 100;

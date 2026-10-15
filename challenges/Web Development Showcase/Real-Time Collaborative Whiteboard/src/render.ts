import type { ShapeItem } from '../shared/types.ts';
import { type Bounds, Camera, intersects } from './camera.ts';
import type { LiveStroke, Visible } from './doc.ts';
import {
    FONT_FAMILY,
    LINE_HEIGHT,
    type Measure,
    itemBounds,
    layoutText,
    shift,
    strokeOutline,
    strokePath,
    svgPath,
} from './geometry.ts';

export const PAPER = '#ffffff';
const HIGHLIGHT_ALPHA = 0.35;
const GRID = '#cfd3d9';
const STICKY_INK = '#1f2328';

export interface Moving {
    ids: ReadonlySet<string>;
    dx: number;
    dy: number;
}

export interface BoardOptions {
    /** Items to draw faded (the eraser is about to remove them). */
    dim?: ReadonlySet<string>;
    /** Items not drawn at all (the one being edited in a text box). */
    hide?: ReadonlySet<string> | undefined;
    /** Items being dragged, drawn at their dragged position. */
    moving?: Moving | null;
    grid?: boolean;
}

/** Map world units to canvas pixels for `camera` on a canvas with the given device pixel ratio. */
export function applyCamera(ctx: CanvasRenderingContext2D, camera: Camera, dpr: number): void {
    const k = camera.zoom * dpr;
    ctx.setTransform(k, 0, 0, k, -camera.x * k, -camera.y * k);
}

/** Paint the whole committed board: paper, dot grid, then every visible item in z-order. */
export function drawBoard(
    ctx: CanvasRenderingContext2D,
    items: readonly Visible[],
    camera: Camera,
    width: number,
    height: number,
    dpr: number,
    measure: Measure,
    options: BoardOptions = {},
): void {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, width * dpr, height * dpr);
    applyCamera(ctx, camera, dpr);
    const view = camera.view(width, height);
    if (options.grid !== false) drawGrid(ctx, camera, view);
    for (const v of items) {
        if (options.hide?.has(v.item.id)) continue;
        const moved = options.moving?.ids.has(v.item.id) ? options.moving : null;
        const dx = v.dx + (moved?.dx ?? 0);
        const dy = v.dy + (moved?.dy ?? 0);
        if (!intersects(shift(itemBounds(v.item, measure), dx, dy), view)) continue;
        ctx.save();
        if (options.dim?.has(v.item.id)) ctx.globalAlpha = 0.25;
        drawItem(ctx, { item: v.item, dx, dy }, measure);
        ctx.restore();
    }
}

function drawGrid(ctx: CanvasRenderingContext2D, camera: Camera, view: Bounds): void {
    if (camera.zoom < 0.25) return; // dots would be a haze
    const step = camera.zoom < 0.6 ? 80 : 40;
    const size = Math.max(1.2, 1.4 / camera.zoom);
    ctx.fillStyle = GRID;
    for (let x = Math.floor(view.minX / step) * step; x <= view.maxX; x += step) {
        for (let y = Math.floor(view.minY / step) * step; y <= view.maxY; y += step) {
            ctx.fillRect(x - size / 2, y - size / 2, size, size);
        }
    }
}

/** Draw one item. Also used to draw the PNG export, so the board and its export always agree. */
export function drawItem(ctx: CanvasRenderingContext2D, v: Visible, measure: Measure): void {
    const { item } = v;
    ctx.translate(v.dx, v.dy);
    switch (item.t) {
        case 'stroke':
            ctx.fillStyle = item.color;
            if (item.hl) ctx.globalAlpha *= HIGHLIGHT_ALPHA;
            ctx.fill(strokePath(item));
            break;
        case 'shape':
            drawShape(ctx, item);
            break;
        case 'text': {
            const l = layoutText(item, measure);
            ctx.font = `${l.fs}px ${FONT_FAMILY}`;
            ctx.textBaseline = 'top';
            if (item.sticky) {
                ctx.fillStyle = 'rgba(0,0,0,0.16)';
                ctx.beginPath();
                ctx.roundRect(l.x, l.y + 3, l.w, l.h, 6);
                ctx.fill();
                ctx.fillStyle = item.color;
                ctx.beginPath();
                ctx.roundRect(l.x, l.y, l.w, l.h, 6);
                ctx.fill();
            }
            ctx.fillStyle = item.sticky ? STICKY_INK : item.color;
            l.lines.forEach((line, i) => ctx.fillText(line, l.x + l.pad, l.y + l.pad + i * l.fs * LINE_HEIGHT));
            break;
        }
    }
}

export function arrowHead(s: Pick<ShapeItem, 'x1' | 'y1' | 'x2' | 'y2' | 'size'>): [number, number, number, number] {
    const angle = Math.atan2(s.y2 - s.y1, s.x2 - s.x1);
    const len = Math.max(12, s.size * 4);
    const a = Math.PI / 7;
    return [
        s.x2 - len * Math.cos(angle - a),
        s.y2 - len * Math.sin(angle - a),
        s.x2 - len * Math.cos(angle + a),
        s.y2 - len * Math.sin(angle + a),
    ];
}

export function drawShape(ctx: CanvasRenderingContext2D, s: ShapeItem): void {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.size;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    switch (s.k) {
        case 'line':
        case 'arrow':
            ctx.moveTo(s.x1, s.y1);
            ctx.lineTo(s.x2, s.y2);
            if (s.k === 'arrow') {
                const [ax, ay, bx, by] = arrowHead(s);
                ctx.moveTo(ax, ay);
                ctx.lineTo(s.x2, s.y2);
                ctx.lineTo(bx, by);
            }
            break;
        case 'rect':
            ctx.rect(Math.min(s.x1, s.x2), Math.min(s.y1, s.y2), Math.abs(s.x2 - s.x1), Math.abs(s.y2 - s.y1));
            break;
        case 'ellipse':
            ctx.ellipse(
                (s.x1 + s.x2) / 2,
                (s.y1 + s.y2) / 2,
                Math.max(Math.abs(s.x2 - s.x1) / 2, 0.5),
                Math.max(Math.abs(s.y2 - s.y1) / 2, 0.5),
                0,
                0,
                Math.PI * 2,
            );
            break;
    }
    ctx.stroke();
}

export type Preview =
    | { kind: 'stroke'; pts: number[]; pr: number[] | null; color: string; size: number; hl: boolean }
    | { kind: 'shape'; shape: ShapeItem }
    | { kind: 'eraser'; x: number; y: number; r: number };

export interface CursorView {
    x: number;
    y: number;
    name: string;
    color: string;
    alpha: number;
}

export interface OverlayState {
    camera: Camera;
    width: number;
    height: number;
    dpr: number;
    live: Iterable<LiveStroke>;
    cursors: Iterable<CursorView>;
    preview: Preview | null;
    selection: Bounds | null;
}

/** The top layer: other people's strokes in progress, your own preview, selection and cursors. */
export function drawOverlay(ctx: CanvasRenderingContext2D, o: OverlayState): void {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, o.width * o.dpr, o.height * o.dpr);
    applyCamera(ctx, o.camera, o.dpr);
    for (const live of o.live) {
        ctx.save();
        ctx.fillStyle = live.start.color;
        if (live.start.hl) ctx.globalAlpha = HIGHLIGHT_ALPHA;
        drawStrokePreview(ctx, live.pts, live.pr, live.start.size, !!live.start.hl);
        ctx.restore();
    }
    const p = o.preview;
    if (p?.kind === 'stroke') {
        ctx.save();
        ctx.fillStyle = p.color;
        if (p.hl) ctx.globalAlpha = HIGHLIGHT_ALPHA;
        drawStrokePreview(ctx, p.pts, p.pr, p.size, p.hl);
        ctx.restore();
    } else if (p?.kind === 'shape') {
        ctx.save();
        drawShape(ctx, p.shape);
        ctx.restore();
    }
    if (o.selection) {
        const s = o.selection;
        ctx.save();
        ctx.strokeStyle = '#2f6df6';
        ctx.lineWidth = 1.5 / o.camera.zoom;
        ctx.setLineDash([6 / o.camera.zoom, 4 / o.camera.zoom]);
        ctx.strokeRect(s.minX, s.minY, s.maxX - s.minX, s.maxY - s.minY);
        ctx.restore();
    }
    if (p?.kind === 'eraser') {
        ctx.save();
        ctx.strokeStyle = '#555';
        ctx.lineWidth = 1.5 / o.camera.zoom;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }
    // Cursors are drawn in screen space so they stay the same size at any zoom.
    ctx.setTransform(o.dpr, 0, 0, o.dpr, 0, 0);
    for (const c of o.cursors) drawCursor(ctx, o.camera, c);
}

function drawStrokePreview(
    ctx: CanvasRenderingContext2D,
    pts: number[],
    pr: number[] | null,
    size: number,
    hl: boolean,
): void {
    if (pts.length < 2) return;
    ctx.fill(new Path2D(svgPath(strokeOutline({ pts, pr: pr ?? undefined, size, hl }, false))));
}

function drawCursor(ctx: CanvasRenderingContext2D, camera: Camera, c: CursorView): void {
    const [sx, sy] = camera.toScreen(c.x, c.y);
    ctx.save();
    ctx.globalAlpha = c.alpha;
    ctx.translate(sx, sy);
    ctx.fillStyle = c.color;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, 16);
    ctx.lineTo(4.5, 12);
    ctx.lineTo(8, 19);
    ctx.lineTo(11, 17.5);
    ctx.lineTo(7.5, 11);
    ctx.lineTo(13, 11);
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    ctx.font = `600 12px ${FONT_FAMILY}`;
    ctx.textBaseline = 'middle';
    const w = ctx.measureText(c.name).width + 12;
    ctx.fillStyle = c.color;
    ctx.beginPath();
    ctx.roundRect(12, 18, w, 20, 10);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(c.name, 18, 28.5);
    ctx.restore();
}

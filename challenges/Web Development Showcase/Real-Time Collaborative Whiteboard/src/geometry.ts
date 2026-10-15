import { getStroke } from 'perfect-freehand';
import type { Item, ShapeItem, StrokeItem, TextItem } from '../shared/types.ts';
import type { Bounds } from './camera.ts';

/** Width of `text` at font size `fs`, in world units. The browser backs this with canvas `measureText`. */
export type Measure = (text: string, fs: number) => number;

export const FONT_FAMILY = 'ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif';
export const LINE_HEIGHT = 1.3;
export const STICKY_PAD = 0.5; // in font sizes

export interface TextLayout {
    lines: string[];
    /** Top-left of the box and its size, in world units, before any move offset. */
    x: number;
    y: number;
    w: number;
    h: number;
    fs: number;
    pad: number;
}

/** Break `text` into lines no wider than `max`. Explicit newlines are kept; long words are split. */
export function wrapText(text: string, max: number, fs: number, measure: Measure): string[] {
    const out: string[] = [];
    // Trailing spaces do not count towards a line's width.
    const fits = (s: string) => measure(s.trimEnd(), fs) <= max;
    for (const paragraph of text.split('\n')) {
        let line = '';
        for (const word of paragraph.split(/(?<=\s)/)) {
            if (line === '' || fits(line + word)) {
                line += word;
            } else {
                out.push(line.trimEnd());
                line = word;
            }
            while (!fits(line) && line.trimEnd().length > 1) {
                // A single word wider than the box: cut after the longest prefix that still fits. Width
                // only grows with length, so a binary search finds it in a few steps (a linear scan made
                // a 2,000-letter word cost hundreds of thousands of measurements).
                let lo = 1;
                let hi = line.length - 1;
                while (lo < hi) {
                    const mid = (lo + hi + 1) >> 1;
                    if (fits(line.slice(0, mid))) lo = mid;
                    else hi = mid - 1;
                }
                out.push(line.slice(0, lo));
                line = line.slice(lo);
            }
        }
        out.push(line.trimEnd());
    }
    return out;
}

const layouts = new WeakMap<TextItem, { measure: Measure; layout: TextLayout }>();

/** Text items never change once added (an edit makes a new object), so their layout is computed once. */
export function layoutText(item: TextItem, measure: Measure): TextLayout {
    const cached = layouts.get(item);
    if (cached && cached.measure === measure) return cached.layout;
    const layout = computeLayout(item, measure);
    layouts.set(item, { measure, layout });
    return layout;
}

function computeLayout(item: TextItem, measure: Measure): TextLayout {
    const fs = item.size;
    if (!item.sticky) {
        const lines = item.text.split('\n');
        const w = Math.max(fs * 0.6, ...lines.map((l) => measure(l, fs)));
        return { lines, x: item.x, y: item.y, w, h: lines.length * fs * LINE_HEIGHT, fs, pad: 0 };
    }
    const pad = fs * STICKY_PAD;
    const w = Math.max(160, fs * 9);
    const lines = wrapText(item.text, w - 2 * pad, fs, measure);
    const h = Math.max(w * 0.8, lines.length * fs * LINE_HEIGHT + 2 * pad);
    return { lines, x: item.x, y: item.y, w, h, fs, pad };
}

const outlines = new WeakMap<StrokeItem, Path2D>();

/** The filled outline of a stroke, cached per item (items never change once added). */
export function strokePath(item: StrokeItem): Path2D {
    let path = outlines.get(item);
    if (!path) {
        path = new Path2D(svgPath(strokeOutline(item)));
        outlines.set(item, path);
    }
    return path;
}

export function strokeOutline(item: Pick<StrokeItem, 'pts' | 'pr' | 'size' | 'hl'>, last = true): number[][] {
    const points: number[][] = [];
    for (let i = 0; i < item.pts.length; i += 2) {
        const p = item.pr?.[i / 2];
        points.push(p === undefined ? [item.pts[i]!, item.pts[i + 1]!] : [item.pts[i]!, item.pts[i + 1]!, p]);
    }
    return getStroke(points, {
        size: item.size,
        thinning: item.hl ? 0 : 0.5,
        smoothing: 0.5,
        streamline: 0.5,
        simulatePressure: item.pr === undefined,
        last,
    });
}

/** Outline points to an SVG path (quadratic curves through midpoints), as in the perfect-freehand docs. */
export function svgPath(points: number[][]): string {
    const len = points.length;
    if (len < 4) return '';
    const mid = (a: number[], b: number[]) => `${((a[0]! + b[0]!) / 2).toFixed(2)},${((a[1]! + b[1]!) / 2).toFixed(2)}`;
    let d = `M${points[0]![0]!.toFixed(2)},${points[0]![1]!.toFixed(2)} Q${points[1]![0]!.toFixed(2)},${points[1]![1]!.toFixed(2)} ${mid(points[1]!, points[2]!)} T`;
    for (let i = 2; i < len - 1; i++) d += `${mid(points[i]!, points[i + 1]!)} `;
    return `${d}Z`;
}

const boundsCache = new WeakMap<Item, Bounds>();

/**
 * World-space bounds of an item before its move offset, with its line width included. Strokes and
 * shapes never change once added, so their bounds are computed once (every repaint culls by them).
 */
export function itemBounds(item: Item, measure: Measure): Bounds {
    if (item.t === 'text') return computeBounds(item, measure); // depends on font metrics, so not cached
    let bounds = boundsCache.get(item);
    if (!bounds) boundsCache.set(item, (bounds = computeBounds(item, measure)));
    return bounds;
}

function computeBounds(item: Item, measure: Measure): Bounds {
    switch (item.t) {
        case 'stroke': {
            let minX = Infinity;
            let minY = Infinity;
            let maxX = -Infinity;
            let maxY = -Infinity;
            for (let i = 0; i < item.pts.length; i += 2) {
                minX = Math.min(minX, item.pts[i]!);
                maxX = Math.max(maxX, item.pts[i]!);
                minY = Math.min(minY, item.pts[i + 1]!);
                maxY = Math.max(maxY, item.pts[i + 1]!);
            }
            const r = item.size / 2 + 1;
            return { minX: minX - r, minY: minY - r, maxX: maxX + r, maxY: maxY + r };
        }
        case 'shape': {
            const r = item.size / 2 + (item.k === 'arrow' ? Math.max(12, item.size * 4) : 1);
            return {
                minX: Math.min(item.x1, item.x2) - r,
                minY: Math.min(item.y1, item.y2) - r,
                maxX: Math.max(item.x1, item.x2) + r,
                maxY: Math.max(item.y1, item.y2) + r,
            };
        }
        case 'text': {
            const l = layoutText(item, measure);
            return { minX: l.x - 4, minY: l.y - 4, maxX: l.x + l.w + 4, maxY: l.y + l.h + 4 };
        }
    }
}

function segDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 === 0 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** Smallest distance from a point to an item's drawn ink (an unfilled shape is only its outline). */
export function distanceTo(item: Item, px: number, py: number, measure: Measure): number {
    switch (item.t) {
        case 'stroke': {
            const { pts } = item;
            if (pts.length === 2) return Math.hypot(px - pts[0]!, py - pts[1]!) - item.size / 2;
            let best = Infinity;
            for (let i = 0; i + 3 < pts.length; i += 2) {
                best = Math.min(best, segDistance(px, py, pts[i]!, pts[i + 1]!, pts[i + 2]!, pts[i + 3]!));
            }
            return best - item.size / 2;
        }
        case 'shape':
            return shapeDistance(item, px, py) - item.size / 2;
        case 'text': {
            const l = layoutText(item, measure);
            const dx = Math.max(l.x - px, 0, px - (l.x + l.w));
            const dy = Math.max(l.y - py, 0, py - (l.y + l.h));
            return Math.hypot(dx, dy);
        }
    }
}

function shapeDistance(s: ShapeItem, px: number, py: number): number {
    switch (s.k) {
        case 'line':
        case 'arrow':
            return segDistance(px, py, s.x1, s.y1, s.x2, s.y2);
        case 'rect': {
            const [l, r] = [Math.min(s.x1, s.x2), Math.max(s.x1, s.x2)];
            const [t, b] = [Math.min(s.y1, s.y2), Math.max(s.y1, s.y2)];
            return Math.min(
                segDistance(px, py, l, t, r, t),
                segDistance(px, py, r, t, r, b),
                segDistance(px, py, r, b, l, b),
                segDistance(px, py, l, b, l, t),
            );
        }
        case 'ellipse': {
            const cx = (s.x1 + s.x2) / 2;
            const cy = (s.y1 + s.y2) / 2;
            const rx = Math.max(Math.abs(s.x2 - s.x1) / 2, 0.5);
            const ry = Math.max(Math.abs(s.y2 - s.y1) / 2, 0.5);
            // Radial distance scaled back to world units: exact on the axes, close elsewhere.
            return Math.abs(Math.hypot((px - cx) / rx, (py - cy) / ry) - 1) * Math.min(rx, ry);
        }
    }
}

/** Does the segment (the eraser's path this frame) touch the item within `radius`? */
export function touches(item: Item, ax: number, ay: number, bx: number, by: number, radius: number, measure: Measure): boolean {
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / Math.max(1, radius / 2)));
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        if (distanceTo(item, ax + (bx - ax) * t, ay + (by - ay) * t, measure) <= radius) return true;
    }
    return false;
}

export function shift(b: Bounds, dx: number, dy: number): Bounds {
    return { minX: b.minX + dx, minY: b.minY + dy, maxX: b.maxX + dx, maxY: b.maxY + dy };
}

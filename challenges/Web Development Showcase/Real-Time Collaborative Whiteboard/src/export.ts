import { type Bounds, Camera, union } from './camera.ts';
import type { Visible } from './doc.ts';
import { FONT_FAMILY, LINE_HEIGHT, type Measure, itemBounds, layoutText, shift, strokeOutline, svgPath } from './geometry.ts';
import { arrowHead, drawBoard } from './render.ts';

const PAD = 24;
/** Largest side of an exported PNG in pixels; bigger boards are exported at a lower scale. */
const MAX_PNG_SIDE = 4096;

export function sceneBounds(items: readonly Visible[], measure: Measure): Bounds | null {
    let bounds: Bounds | null = null;
    for (const v of items) bounds = union(bounds, shift(itemBounds(v.item, measure), v.dx, v.dy));
    return bounds;
}

/** XML text and attribute escaping, plus removal of characters XML 1.0 cannot carry at all. */
export function esc(s: string): string {
    return s
        .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f￾￿]/g, '')
        .replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c]!);
}

const num = (v: number) => String(+v.toFixed(2));
const color = (c: string) => (/^#[0-9a-fA-F]{6}$/.test(c) ? c : '#000000');

/** The board as a standalone SVG document, or `null` when there is nothing to export. */
export function toSvg(items: readonly Visible[], measure: Measure): string | null {
    const b = sceneBounds(items, measure);
    if (!b) return null;
    const x = b.minX - PAD;
    const y = b.minY - PAD;
    const w = b.maxX - b.minX + 2 * PAD;
    const h = b.maxY - b.minY + 2 * PAD;
    const body: string[] = [];
    for (const { item, dx, dy } of items) {
        const at = dx || dy ? ` transform="translate(${num(dx)} ${num(dy)})"` : '';
        const c = color(item.color);
        switch (item.t) {
            case 'stroke': {
                const d = svgPath(strokeOutline(item));
                if (d) body.push(`<path d="${d}" fill="${c}"${item.hl ? ' fill-opacity="0.35"' : ''}${at}/>`);
                break;
            }
            case 'shape': {
                const line = `fill="none" stroke="${c}" stroke-width="${num(item.size)}" stroke-linecap="round" stroke-linejoin="round"${at}`;
                if (item.k === 'line') {
                    body.push(`<line x1="${num(item.x1)}" y1="${num(item.y1)}" x2="${num(item.x2)}" y2="${num(item.y2)}" ${line}/>`);
                } else if (item.k === 'arrow') {
                    const [ax, ay, bx, by] = arrowHead(item);
                    body.push(
                        `<path d="M${num(item.x1)} ${num(item.y1)}L${num(item.x2)} ${num(item.y2)}M${num(ax)} ${num(ay)}L${num(item.x2)} ${num(item.y2)}L${num(bx)} ${num(by)}" ${line}/>`,
                    );
                } else if (item.k === 'rect') {
                    body.push(
                        `<rect x="${num(Math.min(item.x1, item.x2))}" y="${num(Math.min(item.y1, item.y2))}" width="${num(Math.abs(item.x2 - item.x1))}" height="${num(Math.abs(item.y2 - item.y1))}" ${line}/>`,
                    );
                } else {
                    body.push(
                        `<ellipse cx="${num((item.x1 + item.x2) / 2)}" cy="${num((item.y1 + item.y2) / 2)}" rx="${num(Math.abs(item.x2 - item.x1) / 2)}" ry="${num(Math.abs(item.y2 - item.y1) / 2)}" ${line}/>`,
                    );
                }
                break;
            }
            case 'text': {
                const l = layoutText(item, measure);
                const parts: string[] = [];
                if (item.sticky) {
                    parts.push(
                        `<rect x="${num(l.x)}" y="${num(l.y + 3)}" width="${num(l.w)}" height="${num(l.h)}" rx="6" fill="#000" fill-opacity="0.16"/>`,
                        `<rect x="${num(l.x)}" y="${num(l.y)}" width="${num(l.w)}" height="${num(l.h)}" rx="6" fill="${c}"/>`,
                    );
                }
                const tspans = l.lines
                    .map((line, i) => `<tspan x="${num(l.x + l.pad)}" y="${num(l.y + l.pad + i * l.fs * LINE_HEIGHT)}">${esc(line)}</tspan>`)
                    .join('');
                parts.push(
                    `<text font-family="${esc(FONT_FAMILY)}" font-size="${num(l.fs)}" dominant-baseline="text-before-edge" xml:space="preserve" fill="${item.sticky ? '#1f2328' : c}">${tspans}</text>`,
                );
                body.push(at ? `<g${at}>${parts.join('')}</g>` : parts.join(''));
                break;
            }
        }
    }
    return [
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${num(x)} ${num(y)} ${num(w)} ${num(h)}" width="${num(w)}" height="${num(h)}">`,
        `<rect x="${num(x)}" y="${num(y)}" width="${num(w)}" height="${num(h)}" fill="#ffffff"/>`,
        ...body,
        '</svg>',
    ].join('\n');
}

/** The board as a PNG, drawn by the same code as the screen. `null` when there is nothing to export. */
export function toPng(items: readonly Visible[], measure: Measure): Promise<Blob | null> {
    const b = sceneBounds(items, measure);
    if (!b) return Promise.resolve(null);
    const w = b.maxX - b.minX + 2 * PAD;
    const h = b.maxY - b.minY + 2 * PAD;
    const scale = Math.min(2, MAX_PNG_SIDE / Math.max(w, h));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(w * scale));
    canvas.height = Math.max(1, Math.round(h * scale));
    const camera = new Camera();
    camera.x = b.minX - PAD;
    camera.y = b.minY - PAD;
    camera.zoom = scale;
    const ctx = canvas.getContext('2d');
    if (!ctx) return Promise.resolve(null);
    drawBoard(ctx, items, camera, canvas.width, canvas.height, 1, measure, { grid: false });
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
}

import { z } from 'zod';
import {
    HEX_COLOR,
    MAX_COORD,
    MAX_IDS_PER_OP,
    MAX_NAME,
    MAX_POINTS_PER_LIVE_CHUNK,
    MAX_SIZE,
    MAX_STROKE_POINTS,
    MAX_TEXT,
} from '../shared/limits.ts';
import type { ClientMsg } from '../shared/types.ts';

// Everything a client sends crosses a trust boundary, so each field is bounded here.
const coord = z.number().min(-MAX_COORD).max(MAX_COORD);
const id = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const color = z.string().regex(HEX_COLOR);
const size = z.number().min(0.5).max(MAX_SIZE);
const pressure = z.number().min(0).max(1);
const name = z.string().max(MAX_NAME * 4);

const pointsOk = (m: { pts: number[]; pr?: number[] | undefined }) =>
    m.pts.length % 2 === 0 && (m.pr === undefined || m.pr.length * 2 === m.pts.length);

const itemBase = { id, color, size };

const item = z.discriminatedUnion('t', [
    z
        .object({
            ...itemBase,
            t: z.literal('stroke'),
            pts: z.array(coord).min(2).max(MAX_STROKE_POINTS * 2),
            pr: z.array(pressure).max(MAX_STROKE_POINTS).optional(),
            hl: z.boolean().optional(),
        })
        .refine(pointsOk, 'pts and pr disagree'),
    z.object({
        ...itemBase,
        t: z.literal('shape'),
        k: z.enum(['line', 'arrow', 'rect', 'ellipse']),
        x1: coord,
        y1: coord,
        x2: coord,
        y2: coord,
    }),
    z.object({
        ...itemBase,
        t: z.literal('text'),
        x: coord,
        y: coord,
        text: z.string().max(MAX_TEXT),
        sticky: z.boolean(),
    }),
]);

const ids = z.array(id).min(1).max(MAX_IDS_PER_OP);

const op = z.discriminatedUnion('k', [
    z.object({ k: z.literal('add'), item }),
    z.object({ k: z.literal('del'), ids }),
    z.object({ k: z.literal('mv'), ids, dx: coord, dy: coord }),
    z.object({ k: z.literal('edit'), id, text: z.string().max(MAX_TEXT) }),
]);

export const clientMsg = z.discriminatedUnion('t', [
    z.object({
        t: z.literal('hello'),
        v: z.literal(1),
        id,
        name,
        last: z.number().int().min(0).nullable(),
    }),
    z.object({ t: z.literal('op'), opId: id, op }),
    z.object({ t: z.literal('undo'), opId: id }),
    z.object({ t: z.literal('redo'), opId: id }),
    z
        .object({
            t: z.literal('live'),
            sid: id,
            pts: z.array(coord).min(2).max(MAX_POINTS_PER_LIVE_CHUNK * 2),
            pr: z.array(pressure).max(MAX_POINTS_PER_LIVE_CHUNK).optional(),
            start: z.object({ color, size, hl: z.boolean().optional() }).optional(),
        })
        .refine(pointsOk, 'pts and pr disagree'),
    z.object({ t: z.literal('liveEnd'), sid: id }),
    z.object({ t: z.literal('cur'), x: coord, y: coord }),
    z.object({ t: z.literal('name'), name }),
    z.object({ t: z.literal('ping') }),
]);

/** Parse one text frame into a validated message, or `null` when it is not one. */
export function parseClientMsg(raw: string): ClientMsg | null {
    let json: unknown;
    try {
        json = JSON.parse(raw);
    } catch {
        return null;
    }
    const parsed = clientMsg.safeParse(json);
    return parsed.success ? (parsed.data as ClientMsg) : null;
}

/** Items loaded from disk go through the same schema as items from the wire. */
export const persistedItem = item;

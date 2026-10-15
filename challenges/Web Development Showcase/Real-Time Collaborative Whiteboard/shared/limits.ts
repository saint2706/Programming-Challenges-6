/** Limits the server enforces and the client respects, so honest clients never hit them. */

export const MAX_FRAME_BYTES = 256 * 1024;
/** A stroke that grows past this is committed and continued as a new stroke. */
export const MAX_STROKE_POINTS = 4000;
export const MAX_POINTS_PER_LIVE_CHUNK = 400;
export const MAX_TEXT = 2000;
export const MAX_NAME = 24;
export const MAX_IDS_PER_OP = 500;
/** Visible items per room. Erasing frees room: erased items are compacted away once nothing can undo them. */
export const MAX_ITEMS_PER_ROOM = 20_000;
/** Rough size of a room's items in memory and on disk. Counting bytes as well as items bounds the worst case. */
export const MAX_ROOM_BYTES = 64 * 1024 * 1024;
export const MAX_MEMBERS_PER_ROOM = 50;
/** Largest accumulated move of one item. Far beyond anything on screen, small enough to stay exact. */
export const MAX_OFFSET = 1e7;
/** World coordinates beyond this are rejected (also keeps arithmetic far from float trouble). */
export const MAX_COORD = 1e6;
export const MAX_SIZE = 200;

export const ROOM_ID = /^[A-Za-z0-9_-]{1,40}$/;
export const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;

import type { ShapeKind } from '../shared/types.ts';

export type ToolId =
    | 'select'
    | 'pen'
    | 'highlighter'
    | 'eraser'
    | 'line'
    | 'arrow'
    | 'rect'
    | 'ellipse'
    | 'text'
    | 'sticky'
    | 'hand';

export interface ToolInfo {
    id: ToolId;
    label: string;
    key: string;
}

/** In toolbar order. `key` is the single-letter shortcut. */
export const TOOLS: readonly ToolInfo[] = [
    { id: 'select', label: 'Select and move', key: 'v' },
    { id: 'hand', label: 'Pan', key: 'h' },
    { id: 'pen', label: 'Pen', key: 'p' },
    { id: 'highlighter', label: 'Highlighter', key: 'k' },
    { id: 'eraser', label: 'Eraser', key: 'e' },
    { id: 'line', label: 'Line', key: 'l' },
    { id: 'arrow', label: 'Arrow', key: 'a' },
    { id: 'rect', label: 'Rectangle', key: 'r' },
    { id: 'ellipse', label: 'Ellipse', key: 'o' },
    { id: 'text', label: 'Text', key: 't' },
    { id: 'sticky', label: 'Sticky note', key: 'n' },
];

export const SHAPE_TOOLS: ReadonlySet<ToolId> = new Set<ToolId>(['line', 'arrow', 'rect', 'ellipse']);

export function shapeKind(tool: ToolId): ShapeKind {
    return tool as ShapeKind;
}

/** Ink colours, chosen to stay distinct for common colour-vision differences. */
export const COLORS = ['#1f2328', '#d1242f', '#e16f24', '#bf8700', '#1a7f37', '#0969da', '#8250df', '#6e7781'] as const;

/** The highlighter starts yellow-gold so it looks like a highlighter, not grey ink. */
export const HIGHLIGHTER_DEFAULT = '#bf8700';

export const STICKY_COLORS = ['#fff3a3', '#ffd0d6', '#c9f0d2', '#c7e3ff', '#e5d4ff'] as const;

export const SIZE_NAMES = ['Small', 'Medium', 'Large'] as const;

/** Stroke width for pens and shapes, per size step. */
export const STROKE_SIZES = [3, 6, 12] as const;
/** Highlighter is always wide. */
export const HIGHLIGHT_SIZES = [14, 22, 34] as const;
/** Font size for text and notes, per size step. */
export const TEXT_SIZES = [18, 28, 44] as const;

export function sizeFor(tool: ToolId, step: number): number {
    const i = Math.min(2, Math.max(0, step));
    if (tool === 'highlighter') return HIGHLIGHT_SIZES[i]!;
    if (tool === 'text' || tool === 'sticky') return TEXT_SIZES[i]!;
    return STROKE_SIZES[i]!;
}

export const MIN_ZOOM = 0.05;
export const MAX_ZOOM = 8;

export interface Bounds {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
}

/**
 * The view onto the infinite board. `x, y` is the world point at the top-left corner of the screen
 * and `zoom` is screen pixels per world unit, so `screen = (world - xy) * zoom`.
 */
export class Camera {
    x = 0;
    y = 0;
    zoom = 1;

    toScreen(wx: number, wy: number): [number, number] {
        return [(wx - this.x) * this.zoom, (wy - this.y) * this.zoom];
    }

    toWorld(sx: number, sy: number): [number, number] {
        return [sx / this.zoom + this.x, sy / this.zoom + this.y];
    }

    /** Pan by a screen-space delta (dragging the board by the pointer). */
    panBy(dsx: number, dsy: number): void {
        this.x -= dsx / this.zoom;
        this.y -= dsy / this.zoom;
    }

    /** Zoom by `factor`, keeping the world point under the screen point `(sx, sy)` where it is. */
    zoomAt(sx: number, sy: number, factor: number): void {
        const [wx, wy] = this.toWorld(sx, sy);
        this.zoom = clampZoom(this.zoom * factor);
        this.x = wx - sx / this.zoom;
        this.y = wy - sy / this.zoom;
    }

    /** Set an exact zoom around a screen point (used by the 100% button). */
    zoomTo(sx: number, sy: number, zoom: number): void {
        this.zoomAt(sx, sy, clampZoom(zoom) / this.zoom);
    }

    /** Frame `bounds` in a `width` x `height` screen with `margin` pixels around it. */
    fit(bounds: Bounds, width: number, height: number, margin = 48): void {
        const w = Math.max(1, bounds.maxX - bounds.minX);
        const h = Math.max(1, bounds.maxY - bounds.minY);
        const zoom = clampZoom(Math.min((width - 2 * margin) / w, (height - 2 * margin) / h, 1.5));
        this.zoom = zoom;
        this.x = (bounds.minX + bounds.maxX) / 2 - width / 2 / zoom;
        this.y = (bounds.minY + bounds.maxY) / 2 - height / 2 / zoom;
    }

    /** The world rectangle on screen. */
    view(width: number, height: number): Bounds {
        return { minX: this.x, minY: this.y, maxX: this.x + width / this.zoom, maxY: this.y + height / this.zoom };
    }
}

export function clampZoom(zoom: number): number {
    return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

export function intersects(a: Bounds, b: Bounds): boolean {
    return a.minX <= b.maxX && a.maxX >= b.minX && a.minY <= b.maxY && a.maxY >= b.minY;
}

export function union(a: Bounds | null, b: Bounds): Bounds {
    if (!a) return { ...b };
    return {
        minX: Math.min(a.minX, b.minX),
        minY: Math.min(a.minY, b.minY),
        maxX: Math.max(a.maxX, b.maxX),
        maxY: Math.max(a.maxY, b.maxY),
    };
}

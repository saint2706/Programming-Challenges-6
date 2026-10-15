import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import { Camera, MAX_ZOOM, MIN_ZOOM, clampZoom, intersects, union } from '../src/camera.ts';

const cam = fc.record({ x: fc.double({ min: -1e5, max: 1e5, noNaN: true }), y: fc.double({ min: -1e5, max: 1e5, noNaN: true }), zoom: fc.double({ min: MIN_ZOOM, max: MAX_ZOOM, noNaN: true }) });

function make(v: { x: number; y: number; zoom: number }): Camera {
    const c = new Camera();
    Object.assign(c, v);
    return c;
}

describe('Camera', () => {
    test('toWorld inverts toScreen', () => {
        fc.assert(
            fc.property(cam, fc.double({ min: -1e4, max: 1e4, noNaN: true }), fc.double({ min: -1e4, max: 1e4, noNaN: true }), (v, wx, wy) => {
                const c = make(v);
                const [sx, sy] = c.toScreen(wx, wy);
                const [rx, ry] = c.toWorld(sx, sy);
                expect(rx).toBeCloseTo(wx, 3);
                expect(ry).toBeCloseTo(wy, 3);
            }),
        );
    });

    test('zooming keeps the world point under the pointer fixed', () => {
        fc.assert(
            fc.property(cam, fc.integer({ min: 0, max: 1500 }), fc.integer({ min: 0, max: 900 }), fc.double({ min: 0.2, max: 5, noNaN: true }), (v, sx, sy, factor) => {
                const c = make(v);
                const [wx, wy] = c.toWorld(sx, sy);
                c.zoomAt(sx, sy, factor);
                const [ax, ay] = c.toWorld(sx, sy);
                expect(ax).toBeCloseTo(wx, 3);
                expect(ay).toBeCloseTo(wy, 3);
                expect(c.zoom).toBeGreaterThanOrEqual(MIN_ZOOM);
                expect(c.zoom).toBeLessThanOrEqual(MAX_ZOOM);
            }),
        );
    });

    test('zoom is clamped at both ends', () => {
        const c = new Camera();
        c.zoomAt(0, 0, 1e9);
        expect(c.zoom).toBe(MAX_ZOOM);
        c.zoomAt(0, 0, 1e-9);
        expect(c.zoom).toBe(MIN_ZOOM);
        expect(clampZoom(NaN)).toBeNaN(); // callers never pass NaN; the clamp does not hide it
    });

    test('panBy moves the view by the screen delta at any zoom', () => {
        const c = make({ x: 10, y: 20, zoom: 2 });
        c.panBy(40, -10); // dragging the board right and up
        expect([c.x, c.y]).toEqual([-10, 25]);
    });

    test('zoomTo sets an exact zoom', () => {
        const c = make({ x: 0, y: 0, zoom: 3 });
        c.zoomTo(100, 100, 1);
        expect(c.zoom).toBeCloseTo(1);
    });

    test('fit frames the bounds inside the screen with the margin', () => {
        const c = new Camera();
        c.fit({ minX: 1000, minY: 2000, maxX: 3000, maxY: 2600 }, 800, 600, 50);
        for (const [wx, wy] of [[1000, 2000], [3000, 2600]] as const) {
            const [sx, sy] = c.toScreen(wx, wy);
            expect(sx).toBeGreaterThanOrEqual(49);
            expect(sx).toBeLessThanOrEqual(751);
            expect(sy).toBeGreaterThanOrEqual(49);
            expect(sy).toBeLessThanOrEqual(551);
        }
    });

    test('fit never zooms in past 150% for a tiny drawing', () => {
        const c = new Camera();
        c.fit({ minX: 0, minY: 0, maxX: 5, maxY: 5 }, 800, 600);
        expect(c.zoom).toBe(1.5);
    });

    test('view reports the world rectangle on screen', () => {
        const c = make({ x: -100, y: 50, zoom: 2 });
        expect(c.view(400, 200)).toEqual({ minX: -100, minY: 50, maxX: 100, maxY: 150 });
    });
});

test('rectangles intersect and union', () => {
    const a = { minX: 0, minY: 0, maxX: 10, maxY: 10 };
    expect(intersects(a, { minX: 10, minY: 10, maxX: 20, maxY: 20 })).toBe(true); // touching counts
    expect(intersects(a, { minX: 11, minY: 0, maxX: 20, maxY: 10 })).toBe(false);
    expect(union(null, a)).toEqual(a);
    expect(union(a, { minX: -5, minY: 3, maxX: 4, maxY: 30 })).toEqual({ minX: -5, minY: 0, maxX: 10, maxY: 30 });
});

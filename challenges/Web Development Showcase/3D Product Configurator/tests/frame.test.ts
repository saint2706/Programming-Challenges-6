import { describe, expect, test } from 'vitest';
import { screenFromDiagonal } from '../src/headcoupled/calibration.ts';
import { canvasOnScreen, eyeInCanvas, placement } from '../src/headcoupled/frame.ts';

const display = { width: 1920, height: 1080, diagonal: 24 };
const full = screenFromDiagonal(24, 1920 / 1080);

describe('where the canvas is on the screen, and how big', () => {
    test('a canvas that fills the screen is the screen, centred on it', () => {
        const p = placement(display, { left: 0, top: 0, width: 1920, height: 1080 });
        expect(p.size.width).toBeCloseTo(full.width, 9);
        expect(p.size.height).toBeCloseTo(full.height, 9);
        expect(p.offset.x).toBeCloseTo(0, 12);
        expect(p.offset.y).toBeCloseTo(0, 12);
    });

    test('half the width is half the metres, and a canvas in the right-hand half is to the right', () => {
        const p = placement(display, { left: 960, top: 0, width: 960, height: 1080 });
        expect(p.size.width).toBeCloseTo(full.width / 2, 9);
        expect(p.offset.x).toBeCloseTo(full.width / 4, 9);
        expect(p.offset.y).toBeCloseTo(0, 12);
    });

    test('a canvas in the top half is above the middle: y runs up', () => {
        const p = placement(display, { left: 0, top: 0, width: 1920, height: 540 });
        expect(p.offset.y).toBeCloseTo(full.height / 4, 9);
    });

    test('a wrong screen size scales everything alike: the picture is the same shape, only its metres change', () => {
        const a = placement(display, { left: 300, top: 100, width: 800, height: 600 });
        const b = placement({ ...display, diagonal: 27 }, { left: 300, top: 100, width: 800, height: 600 });
        expect(b.size.width / a.size.width).toBeCloseTo(27 / 24, 9);
        expect(b.size.width / b.size.height).toBeCloseTo(a.size.width / a.size.height, 9);
        expect(b.offset.x / a.offset.x).toBeCloseTo(27 / 24, 9);
    });
});

describe('the eye as seen from the canvas', () => {
    test('is the eye from the screen, moved by where the canvas is', () => {
        const e = eyeInCanvas({ x: 0.1, y: 0.05, z: 0.6 }, { x: 0.04, y: -0.02 });
        expect(e.x).toBeCloseTo(0.06, 12);
        expect(e.y).toBeCloseTo(0.07, 12);
        expect(e.z).toBe(0.6);
    });

    test('a head in front of the middle of the screen is off to the left of a canvas on the right-hand side', () => {
        const p = placement(display, { left: 960, top: 0, width: 960, height: 1080 });
        expect(eyeInCanvas({ x: 0, y: 0, z: 0.6 }, p.offset).x).toBeLessThan(0);
    });
});

describe('finding the canvas on the screen', () => {
    const rect = { left: 10, top: 50, width: 800, height: 600 } as DOMRect;

    test('a maximised window with a toolbar: the canvas starts below the toolbar and just inside the frame', () => {
        const r = canvasOnScreen(rect, { screenX: 0, screenY: 0, outerWidth: 1920, outerHeight: 1080, innerWidth: 1920, innerHeight: 1000 }, 0, 0);
        expect(r).toEqual({ left: 10, top: 130, width: 800, height: 600 });
    });

    test('full screen: window and screen are the same', () => {
        const r = canvasOnScreen({ left: 0, top: 0, width: 1920, height: 1080 } as DOMRect, { screenX: 0, screenY: 0, outerWidth: 1920, outerHeight: 1080, innerWidth: 1920, innerHeight: 1080 }, 0, 0);
        expect(r).toEqual({ left: 0, top: 0, width: 1920, height: 1080 });
    });

    test('a window on a second monitor is measured from that monitor', () => {
        const r = canvasOnScreen(rect, { screenX: 2000, screenY: 40, outerWidth: 1000, outerHeight: 800, innerWidth: 1000, innerHeight: 700 }, 1920, 0);
        expect(r.left).toBe(2000 - 1920 + 10);
    });
});

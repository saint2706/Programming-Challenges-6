import { type Eye, type ScreenSize } from './offaxis.ts';
import { screenFromDiagonal } from './calibration.ts';

/**
 * The page is a window in the screen, not the whole screen: its canvas can be any size and anywhere.
 * The scene is built for a window the size the canvas physically is, and the eye is measured from the
 * middle of that window, not of the screen. Everything here is metres, x to the right, y up.
 */
export interface Rect {
    left: number;
    top: number;
    width: number;
    height: number;
}

export interface Display {
    /** The whole screen in CSS pixels (`screen.width`, `screen.height`). */
    width: number;
    height: number;
    /** Its diagonal in inches, which is what a person can tell us (or we guess). */
    diagonal: number;
}

export interface Placement {
    /** Physical size of the canvas. */
    size: ScreenSize;
    /** From the middle of the screen to the middle of the canvas. */
    offset: { x: number; y: number };
    /** Metres per CSS pixel. */
    pitch: number;
}

/**
 * Where the canvas is on the screen and how big it is in metres. `canvas` is in screen CSS pixels
 * (see `canvasOnScreen`).
 */
export function placement(display: Display, canvas: Rect): Placement {
    const screen = screenFromDiagonal(display.diagonal, display.width / display.height);
    const pitch = screen.width / display.width;
    return {
        size: { width: canvas.width * pitch, height: canvas.height * pitch },
        offset: { x: (canvas.left + canvas.width / 2 - display.width / 2) * pitch, y: -(canvas.top + canvas.height / 2 - display.height / 2) * pitch },
        pitch,
    };
}

/** An eye measured from the middle of the screen, as seen from the middle of the canvas. */
export function eyeInCanvas(eye: Eye, offset: { x: number; y: number }): Eye {
    return { x: eye.x - offset.x, y: eye.y - offset.y, z: eye.z };
}

/** The canvas's rectangle in screen coordinates, from what the browser reports (a best guess: it does not say how thick the window's frame is). */
export function canvasOnScreen(rect: DOMRect, win: { screenX: number; screenY: number; outerWidth: number; outerHeight: number; innerWidth: number; innerHeight: number }, availLeft = 0, availTop = 0): Rect {
    const side = Math.max(0, (win.outerWidth - win.innerWidth) / 2);
    const above = Math.max(0, win.outerHeight - win.innerHeight - side);
    return { left: win.screenX - availLeft + side + rect.left, top: win.screenY - availTop + above + rect.top, width: rect.width, height: rect.height };
}

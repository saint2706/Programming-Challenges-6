import { Vector3 } from 'three';
import { describe, expect, test } from 'vitest';
import { OffAxisCamera } from '../src/headcoupled/camera.ts';
import { WEBGL_DEPTH, WEBGPU_DEPTH } from '../src/headcoupled/offaxis.ts';

const SCREEN = { width: 0.31, height: 0.17 };

/** Where a point lands in the camera's normalised screen: x and y in -1..1 across the window. */
const ndc = (camera: OffAxisCamera, p: [number, number, number]) => new Vector3(...p).project(camera);

describe('the head-coupled camera', () => {
    test('can be built before it has been told anything', () => {
        expect(() => new OffAxisCamera()).not.toThrow();
    });

    test.each([
        ['WebGL depth', WEBGL_DEPTH],
        ['WebGPU depth', WEBGPU_DEPTH],
    ])('puts the corners of the window at the corners of the picture from any eye, with %s', (_, system) => {
        const camera = new OffAxisCamera();
        camera.coordinateSystem = system as never;
        for (const eye of [{ x: 0, y: 0, z: 0.6 }, { x: 0.2, y: -0.05, z: 0.4 }, { x: -0.15, y: 0.1, z: 0.9 }]) {
            camera.place(eye, SCREEN);
            for (const [sx, sy] of [[1, 1], [-1, 1], [1, -1], [-1, -1]] as const) {
                const p = ndc(camera, [(sx * SCREEN.width) / 2, (sy * SCREEN.height) / 2, 0]);
                expect(p.x).toBeCloseTo(sx, 5);
                expect(p.y).toBeCloseTo(sy, 5);
            }
        }
    });

    test('keeps the off-axis picture when the renderer asks for the projection again, even in the other depth convention', () => {
        // The WebGPU renderer sets the camera's coordinate system and calls updateProjectionMatrix; an ordinary camera would turn into a plain perspective.
        const camera = new OffAxisCamera();
        camera.place({ x: 0.2, y: 0, z: 0.5 }, SCREEN);
        const before = camera.projectionMatrix.clone();
        camera.updateProjectionMatrix();
        expect(camera.projectionMatrix.equals(before)).toBe(true);
        camera.coordinateSystem = WEBGPU_DEPTH as never;
        camera.updateProjectionMatrix();
        expect(camera.projectionMatrix.equals(before)).toBe(false); // the depth row differs
        expect(camera.projectionMatrix.elements[8]).toBeCloseTo(before.elements[8]!, 9); // the sideways skew does not
        expect(camera.projectionMatrix.elements[8]).not.toBe(0);
    });

    test('a point behind the glass moves across the window when the head moves, one on the glass does not', () => {
        const camera = new OffAxisCamera();
        const at = (x: number, z: number) => {
            camera.place({ x, y: 0, z: 0.6 }, SCREEN);
            return ndc(camera, [0, 0, z]).x;
        };
        expect(at(0.1, 0) - at(-0.1, 0)).toBeCloseTo(0, 9);
        expect(Math.abs(at(0.1, -0.2) - at(-0.1, -0.2))).toBeGreaterThan(0.2);
    });

    test('an eye at the glass or behind it does not break the matrix', () => {
        const camera = new OffAxisCamera();
        camera.place({ x: 0, y: 0, z: -1 }, SCREEN);
        expect(camera.projectionMatrix.elements.every(Number.isFinite)).toBe(true);
        expect(camera.position.z).toBeGreaterThan(0);
    });
});

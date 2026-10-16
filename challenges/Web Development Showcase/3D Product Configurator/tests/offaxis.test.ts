import fc from 'fast-check';
import { Matrix4, PerspectiveCamera, Vector3, Vector4 } from 'three';
import { describe, expect, test } from 'vitest';
import { MIN_EYE_DISTANCE, WEBGL_DEPTH, WEBGPU_DEPTH, applyOffAxis, offAxisFrustum, projectionMatrix, screenXOf } from '../src/headcoupled/offaxis.ts';

const SCREEN = { width: 0.3, height: 0.19 }; // roughly a 14 inch laptop

/** Normalised device coordinates of a world point seen by a camera set up for `eye`. */
function ndc(eye: { x: number; y: number; z: number }, point: Vector3, coordinateSystem = WEBGL_DEPTH, screen = SCREEN, near = 0.05, far = 5) {
    const camera = new PerspectiveCamera();
    applyOffAxis(camera, screen, eye, near, far, coordinateSystem);
    const clip = new Vector4(point.x, point.y, point.z, 1).applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
    return new Vector3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
}

const eyes = fc.record({
    x: fc.double({ min: -0.5, max: 0.5, noNaN: true }),
    y: fc.double({ min: -0.4, max: 0.4, noNaN: true }),
    z: fc.double({ min: 0.25, max: 1.5, noNaN: true }),
});

describe('the screen is a window', () => {
    test('a point on the glass is drawn in the same place wherever the eye is: that is what makes it a window', () => {
        fc.assert(
            fc.property(eyes, fc.double({ min: -0.15, max: 0.15, noNaN: true }), fc.double({ min: -0.095, max: 0.095, noNaN: true }), (eye, x, y) => {
                const p = ndc(eye, new Vector3(x, y, 0));
                expect(p.x).toBeCloseTo((x / SCREEN.width) * 2, 6);
                expect(p.y).toBeCloseTo((y / SCREEN.height) * 2, 6);
            }),
        );
    });

    test('the four corners of the glass are the four corners of the picture', () => {
        fc.assert(
            fc.property(eyes, (eye) => {
                for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]] as const) {
                    const p = ndc(eye, new Vector3((sx * SCREEN.width) / 2, (sy * SCREEN.height) / 2, 0));
                    expect(p.x).toBeCloseTo(sx, 6);
                    expect(p.y).toBeCloseTo(sy, 6);
                }
            }),
        );
    });

    test('where a point behind the glass lands: the closed form and the matrix agree', () => {
        fc.assert(
            fc.property(eyes, fc.double({ min: 0.01, max: 0.3, noNaN: true }), fc.double({ min: -0.1, max: 0.1, noNaN: true }), (eye, depth, x) => {
                const p = ndc(eye, new Vector3(x, 0, -depth));
                expect(p.x).toBeCloseTo(screenXOf(eye, { x, z: -depth }, SCREEN), 6);
            }),
        );
    });

    test('the nearer to the glass, the less a point moves when the head does', () => {
        const eye = { x: 0.2, y: 0, z: 0.6 };
        const shifts = [0, 0.02, 0.05, 0.1, 0.2].map((d) => Math.abs(ndc(eye, new Vector3(0, 0, -d)).x));
        expect(shifts[0]).toBeCloseTo(0, 9);
        for (let i = 1; i < shifts.length; i++) expect(shifts[i]!).toBeGreaterThan(shifts[i - 1]!);
    });

    test('a point in front of the glass goes the opposite way to one behind it: with the head to the right, what is behind drifts right with it and what floats in front drifts left', () => {
        const eye = { x: 0.2, y: 0, z: 0.6 };
        const behind = ndc(eye, new Vector3(0, 0, -0.1)).x;
        const before = ndc(eye, new Vector3(0, 0, 0.1)).x;
        expect(Math.sign(behind)).toBe(-Math.sign(before));
    });
});

describe('the frustum', () => {
    test('is symmetric when the eye is in front of the middle of the screen', () => {
        const f = offAxisFrustum(SCREEN, { x: 0, y: 0, z: 0.5 }, 0.05, 5);
        expect(f.left).toBeCloseTo(-f.right, 12);
        expect(f.bottom).toBeCloseTo(-f.top, 12);
    });

    test('leans towards the side the eye has moved to', () => {
        const f = offAxisFrustum(SCREEN, { x: 0.1, y: 0, z: 0.5 }, 0.05, 5);
        expect(f.left).toBeLessThan(-f.right); // more of the left of the window is in view
        expect(Math.abs(f.left)).toBeGreaterThan(Math.abs(f.right));
    });

    test('narrows as the eye moves back (the window looks smaller from far away)', () => {
        const near = offAxisFrustum(SCREEN, { x: 0, y: 0, z: 0.3 }, 0.05, 5);
        const far = offAxisFrustum(SCREEN, { x: 0, y: 0, z: 1.2 }, 0.05, 5);
        expect(far.right).toBeLessThan(near.right);
    });

    test('an eye at or behind the glass is held at a minimum distance instead of dividing by zero', () => {
        for (const z of [0, -1, 1e-12]) {
            const f = offAxisFrustum(SCREEN, { x: 0, y: 0, z }, 0.05, 5);
            expect(Number.isFinite(f.right) && Number.isFinite(f.top)).toBe(true);
            expect(f.right).toBeCloseTo((SCREEN.width / 2) * (0.05 / MIN_EYE_DISTANCE), 9);
        }
    });
});

describe('depth conventions', () => {
    test('WebGL maps the near plane to -1 and WebGPU to 0; the far plane to 1 in both', () => {
        const near = 0.05;
        const far = 5;
        const eye = { x: 0.05, y: 0.02, z: 0.6 };
        // A point on the camera's axis at distance near / far in front of the eye.
        const at = (distance: number) => new Vector3(eye.x, eye.y, eye.z - distance);
        expect(ndc(eye, at(near), WEBGL_DEPTH).z).toBeCloseTo(-1, 6);
        expect(ndc(eye, at(far), WEBGL_DEPTH).z).toBeCloseTo(1, 6);
        expect(ndc(eye, at(near), WEBGPU_DEPTH).z).toBeCloseTo(0, 6);
        expect(ndc(eye, at(far), WEBGPU_DEPTH).z).toBeCloseTo(1, 6);
    });

    test('x and y are the same in both conventions', () => {
        const eye = { x: 0.1, y: -0.05, z: 0.5 };
        const p = new Vector3(0.03, 0.02, -0.08);
        const a = ndc(eye, p, WEBGL_DEPTH);
        const b = ndc(eye, p, WEBGPU_DEPTH);
        expect(a.x).toBeCloseTo(b.x, 9);
        expect(a.y).toBeCloseTo(b.y, 9);
    });
});

describe('applyOffAxis', () => {
    test('moves the camera to the eye, leaves it facing the screen, and keeps the inverse in step', () => {
        const camera = new PerspectiveCamera();
        const eye = { x: 0.07, y: 0.03, z: 0.55 };
        applyOffAxis(camera, SCREEN, eye, 0.05, 5);
        expect(camera.position.toArray()).toEqual([eye.x, eye.y, eye.z]);
        expect(camera.quaternion.toArray()).toEqual([0, 0, 0, 1]);
        const product = new Matrix4().multiplyMatrices(camera.projectionMatrix, camera.projectionMatrixInverse);
        for (let i = 0; i < 16; i++) expect(product.elements[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 9);
    });

    test('projectionMatrix writes into the matrix it is given', () => {
        const out = new Matrix4();
        expect(projectionMatrix(offAxisFrustum(SCREEN, { x: 0, y: 0, z: 0.5 }, 0.05, 5), WEBGL_DEPTH, out)).toBe(out);
    });
});

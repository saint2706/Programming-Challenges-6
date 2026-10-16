import { describe, expect, test } from 'vitest';
import { cameraAbove, defaults, screenFromDiagonal } from '../src/headcoupled/calibration.ts';
import { EYE_Z_RANGE, HeadPipeline, REST_EYE } from '../src/headcoupled/pipeline.ts';
import { CAMERA, faceLandmarks } from './helpers.ts';

const settings = () => defaults(1440, 900);

/** Landmarks of a head whose eyes are at `eye` in the screen's frame, as the camera on the screen's top edge sees it. */
function seen(eye: { x: number; y: number; z: number }, ipd = 0.063, s = settings()) {
    const above = cameraAbove(screenFromDiagonal(s.diagonal, s.aspect));
    return faceLandmarks({ x: -eye.x, y: above - eye.y, z: eye.z }, ipd, CAMERA);
}

describe('from landmarks to the eye the picture is drawn for', () => {
    test('a head is found where it is, in the screen frame: left is left, up is up, and distance is distance', () => {
        const pipeline = new HeadPipeline(settings());
        for (const eye of [{ x: 0.12, y: 0.05, z: 0.55 }, { x: -0.2, y: -0.04, z: 0.8 }, { x: 0, y: 0, z: 0.4 }]) {
            pipeline.reset();
            const u = pipeline.update(seen(eye), CAMERA, 1);
            expect(u.found).toBe(true);
            expect(u.eye.x).toBeCloseTo(eye.x, 4);
            expect(u.eye.y).toBeCloseTo(eye.y, 4);
            expect(u.eye.z).toBeCloseTo(eye.z, 4);
        }
    });

    test('the depth setting scales sideways and vertical movement and leaves distance alone', () => {
        const s = { ...settings(), depth: 1.5 };
        const u = new HeadPipeline(s).update(seen({ x: 0.1, y: -0.06, z: 0.6 }, 0.063, s), CAMERA, 1);
        expect(u.eye.x).toBeCloseTo(0.15, 4);
        expect(u.eye.y).toBeCloseTo(-0.09, 4);
        expect(u.eye.z).toBeCloseTo(0.6, 4);
    });

    test('a distance that is not a person at a desk is held to a sensible range', () => {
        expect(new HeadPipeline(settings()).update(seen({ x: 0, y: 0, z: 0.1 }), CAMERA, 1).eye.z).toBe(EYE_Z_RANGE[0]);
        expect(new HeadPipeline(settings()).update(seen({ x: 0, y: 0, z: 3 }), CAMERA, 1).eye.z).toBe(EYE_Z_RANGE[1]);
    });

    test('it learns this person\'s interpupillary distance, which makes the distance right', () => {
        const pipeline = new HeadPipeline(settings());
        const eye = { x: 0.05, y: 0, z: 0.6 };
        const first = pipeline.update(seen(eye, 0.058), CAMERA, 0).eye.z;
        expect(Math.abs(first - 0.6) / 0.6).toBeGreaterThan(0.05); // 5 mm off the average: 8% off in distance
        let last = first;
        for (let i = 1; i <= 60; i++) last = pipeline.update(seen(eye, 0.058), CAMERA, i / 30).eye.z;
        expect(pipeline.ipd.learned).toBe(true);
        expect(Math.abs(last - 0.6) / 0.6).toBeLessThan(0.01);
    });

    test('a lost face stays put for a second, then drifts back to the middle and stays there', () => {
        const pipeline = new HeadPipeline(settings());
        const eye = { x: 0.2, y: 0.05, z: 0.9 };
        pipeline.update(seen(eye), CAMERA, 0);
        const held = pipeline.update(null, CAMERA, 0.5);
        expect(held.found).toBe(false);
        expect(held.eye.x).toBeCloseTo(eye.x, 6);
        expect(held.eye.z).toBeCloseTo(eye.z, 6);
        let u = pipeline.update(null, CAMERA, 1.5);
        expect(u.found).toBe(false);
        expect(u.eye.x).toBeLessThan(eye.x); // on its way
        for (let t = 1.6; t < 8; t += 0.1) u = pipeline.update(null, CAMERA, t);
        expect(u.eye.x).toBeCloseTo(REST_EYE.x, 2);
        expect(u.eye.z).toBeCloseTo(REST_EYE.z, 2);
    });

    test('a face that comes back is followed from where it is, not from where the eye had drifted to', () => {
        const pipeline = new HeadPipeline(settings());
        pipeline.update(seen({ x: 0.2, y: 0, z: 0.9 }), CAMERA, 0);
        for (let t = 1; t < 8; t += 0.1) pipeline.update(null, CAMERA, t);
        const u = pipeline.update(seen({ x: -0.1, y: 0, z: 0.5 }), CAMERA, 8);
        expect(u.eye.x).toBeCloseTo(-0.1, 4);
        expect(u.eye.z).toBeCloseTo(0.5, 4);
    });

    test('movement is smoothed: a jitter of a few millimetres hardly moves the eye', () => {
        const pipeline = new HeadPipeline(settings());
        let low = Infinity;
        let high = -Infinity;
        for (let i = 0; i < 90; i++) {
            const jitter = (i % 2 ? 1 : -1) * 0.004;
            const u = pipeline.update(seen({ x: 0.1 + jitter, y: 0, z: 0.6 }), CAMERA, i / 30);
            if (i > 30) {
                low = Math.min(low, u.eye.x);
                high = Math.max(high, u.eye.x);
            }
        }
        expect(high - low).toBeLessThan(0.004); // the input swings 8 mm
    });
});

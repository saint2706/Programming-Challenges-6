import fc from 'fast-check';
import { describe, expect, test } from 'vitest';
import { BEZEL_M, COMMON_DIAGONALS, HFOV_PRIOR_DEG, cameraAbove, defaults, guessDiagonal, parallaxRatio, screenFromDiagonal } from '../src/headcoupled/calibration.ts';
import { screenXOf } from '../src/headcoupled/offaxis.ts';
import { DEFAULT_IPD_M, IpdEstimator, MAX_YAW_DEG, cameraToScreen, focalPx, measure } from '../src/headcoupled/pose.ts';
import { CAMERA, faceLandmarks } from './helpers.ts';

const head = fc.record({
    x: fc.double({ min: -0.3, max: 0.3, noNaN: true }),
    y: fc.double({ min: -0.2, max: 0.2, noNaN: true }),
    z: fc.double({ min: 0.3, max: 1.2, noNaN: true }),
});

describe('finding the eyes from landmarks', () => {
    test('a face square to the camera is found where it is, for any head and any interpupillary distance', () => {
        fc.assert(
            fc.property(head, fc.double({ min: 0.054, max: 0.074, noNaN: true }), (mid, ipd) => {
                const m = measure(faceLandmarks(mid, ipd, CAMERA), CAMERA, ipd)!;
                expect(m.camera.x).toBeCloseTo(mid.x, 4);
                expect(m.camera.y).toBeCloseTo(mid.y, 4);
                expect(m.camera.z).toBeCloseTo(mid.z, 4);
            }),
        );
    });

    test('a turned head is found exactly, and the turn is measured', () => {
        fc.assert(
            fc.property(head, fc.double({ min: -30, max: 30, noNaN: true }), (mid, yaw) => {
                const m = measure(faceLandmarks(mid, 0.063, CAMERA, yaw), CAMERA, 0.063)!;
                expect(Math.abs(m.camera.z - mid.z) / mid.z).toBeLessThan(1e-6); // exact: the distance comes from the full 3D eye-to-eye vector
                expect(Math.abs(m.yawDeg - Math.abs(yaw))).toBeLessThan(1e-4);
            }),
        );
    });

    test('a face turned too far to trust is refused', () => {
        expect(measure(faceLandmarks({ x: 0, y: 0, z: 0.6 }, 0.063, CAMERA, MAX_YAW_DEG + 8), CAMERA, 0.063)).toBeNull();
        expect(measure(faceLandmarks({ x: 0, y: 0, z: 0.6 }, 0.063, CAMERA, MAX_YAW_DEG - 3), CAMERA, 0.063)).not.toBeNull();
    });

    test('no face, a missing iris, or two eyes on the same spot give nothing', () => {
        expect(measure([], CAMERA, 0.063)).toBeNull();
        expect(measure(Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 })), CAMERA, 0.063)).toBeNull();
    });

    test('a wrong field of view changes only the distance, not the sideways or vertical position', () => {
        // x = pixels * ipd / eye-to-eye pixels: the focal length cancels. z = f * ipd / pixels: it does not.
        const lm = faceLandmarks({ x: 0.1, y: 0.05, z: 0.6 }, 0.063, CAMERA);
        const wide = measure(lm, { ...CAMERA, hfovDeg: 90 }, 0.063)!;
        const base = measure(lm, CAMERA, 0.063)!;
        const k = focalPx({ ...CAMERA, hfovDeg: 90 }) / focalPx(CAMERA);
        expect(wide.camera.z / base.camera.z).toBeCloseTo(k, 9);
        expect(wide.camera.x).toBeCloseTo(base.camera.x, 9);
        expect(wide.camera.y).toBeCloseTo(base.camera.y, 9);
    });

    test('a bigger interpupillary distance scales every coordinate in proportion', () => {
        const lm = faceLandmarks({ x: 0.1, y: 0.05, z: 0.6 }, 0.063, CAMERA);
        const a = measure(lm, CAMERA, 0.07)!.camera;
        const b = measure(lm, CAMERA, 0.063)!.camera;
        for (const axis of ['x', 'y', 'z'] as const) expect(a[axis] / b[axis]).toBeCloseTo(0.07 / 0.063, 9);
    });
});

describe('from the camera to the screen', () => {
    test('left and right are swapped (the camera looks back at you), and up is measured from the middle of the picture', () => {
        const above = 0.105;
        // Someone sitting to the right of the screen's middle appears on the left of the camera's picture.
        const eye = cameraToScreen({ x: -0.1, y: above, z: 0.6 }, above);
        expect(eye).toEqual({ x: 0.1, y: 0, z: 0.6 });
        // Someone with their eyes at the height of the lens is above the middle of the picture by the lens's height.
        expect(cameraToScreen({ x: 0, y: 0, z: 0.5 }, above).y).toBeCloseTo(above, 12);
    });

    test('a full round trip: a known eye position in front of a screen comes back out of the landmarks', () => {
        const screen = screenFromDiagonal(14, 1.6);
        const above = cameraAbove(screen);
        fc.assert(
            fc.property(
                fc.record({ x: fc.double({ min: -0.25, max: 0.25, noNaN: true }), y: fc.double({ min: -0.1, max: 0.2, noNaN: true }), z: fc.double({ min: 0.35, max: 0.9, noNaN: true }) }),
                (eye) => {
                    const inCamera = { x: -eye.x, y: above - eye.y, z: eye.z }; // the inverse of cameraToScreen
                    const found = cameraToScreen(measure(faceLandmarks(inCamera, 0.063, CAMERA), CAMERA, 0.063)!.camera, above);
                    expect(found.x).toBeCloseTo(eye.x, 4);
                    expect(found.y).toBeCloseTo(eye.y, 4);
                    expect(found.z).toBeCloseTo(eye.z, 4);
                },
            ),
        );
    });
});

describe('learning this person\'s interpupillary distance from their irises', () => {
    const frame = (ipd: number, yaw = 0, z = 0.6, irisDiameter?: number) => measure(faceLandmarks({ x: 0, y: 0, z }, ipd, CAMERA, yaw, irisDiameter), CAMERA, DEFAULT_IPD_M)!;

    test.each([0.056, 0.063, 0.071])('it settles on %f m without being told', (ipd) => {
        const estimator = new IpdEstimator();
        expect(estimator.value).toBe(DEFAULT_IPD_M); // the population average until it knows
        for (let i = 0; i < 40; i++) estimator.add(frame(ipd, 0, 0.5 + (i % 7) * 0.05));
        expect(estimator.learned).toBe(true);
        expect(estimator.value).toBeCloseTo(ipd, 4);
    });

    test('it does not matter how far away the person sits while it learns', () => {
        const estimator = new IpdEstimator();
        for (let i = 0; i < 40; i++) estimator.add(frame(0.06, 0, 0.35 + i * 0.02));
        expect(estimator.value).toBeCloseTo(0.06, 4);
    });

    test('turned heads and implausible frames do not count, and a few bad frames cannot move the median', () => {
        const estimator = new IpdEstimator();
        for (let i = 0; i < 40; i++) estimator.add(frame(0.066));
        for (let i = 0; i < 10; i++) estimator.add(frame(0.066, 25)); // turned: ignored
        for (let i = 0; i < 8; i++) estimator.add({ ...frame(0.066), irisPx: 1 }); // a blink, say
        for (let i = 0; i < 8; i++) estimator.add({ ...frame(0.066), irisPx: 500 });
        expect(estimator.value).toBeCloseTo(0.066, 4);
    });

    test('a face with an iris of a different size (the spread is a few percent) moves the estimate by that much, not more', () => {
        const estimator = new IpdEstimator();
        for (let i = 0; i < 40; i++) estimator.add(frame(0.063, 0, 0.6, 0.0117 * 1.04));
        expect(Math.abs(estimator.value - 0.063) / 0.063).toBeLessThan(0.045);
    });

    test('reset forgets what it learned', () => {
        const estimator = new IpdEstimator();
        for (let i = 0; i < 40; i++) estimator.add(frame(0.06));
        estimator.reset();
        expect(estimator.learned).toBe(false);
        expect(estimator.value).toBe(DEFAULT_IPD_M);
    });
});

describe('what can be known about the screen without asking', () => {
    test('a screen from its diagonal and shape', () => {
        const s = screenFromDiagonal(14, 1.6);
        expect(Math.hypot(s.width, s.height)).toBeCloseTo(14 * 0.0254, 9);
        expect(s.width / s.height).toBeCloseTo(1.6, 9);
        expect(s.width).toBeCloseTo(0.3015, 3);
    });

    test.each([
        [1440, 900, 15.6], // a 14 inch laptop at 200%: the guess is off, but only by a size
        [1920, 1080, 21.5],
        [1536, 864, 16], // 16.02 inches by the prior
        [2560, 1440, 27],
        [390, 844, 11.6], // a phone: the smallest size on offer
    ])('%i x %i css pixels is guessed as %f inches', (w, h, diagonal) => {
        expect(guessDiagonal(w, h)).toBe(diagonal);
    });

    test('defaults come from the screen\'s shape and the priors', () => {
        expect(defaults(1920, 1080)).toEqual({ diagonal: 21.5, aspect: 1920 / 1080, hfov: HFOV_PRIOR_DEG, depth: 1 });
        expect(COMMON_DIAGONALS).toContain(guessDiagonal(1280, 800));
    });

    test('the lens is a bezel above the top of the picture', () => {
        const s = screenFromDiagonal(14, 1.6);
        expect(cameraAbove(s)).toBeCloseTo(s.height / 2 + BEZEL_M, 12);
    });
});

describe('how wrong the priors can be before the picture suffers (the sensitivity analysis)', () => {
    test('exactly right means exactly the true effect', () => {
        for (const behind of [0.02, 0.1, 0.3]) expect(parallaxRatio(1, 1, 1, behind, 0.6)).toBeCloseTo(1, 12);
    });

    test('the closed form is what the projection matrices actually do', () => {
        fc.assert(
            fc.property(
                fc.double({ min: 0.9, max: 1.1, noNaN: true }),
                fc.double({ min: 0.8, max: 1.25, noNaN: true }),
                fc.double({ min: 0.8, max: 1.25, noNaN: true }),
                fc.double({ min: 0.02, max: 0.4, noNaN: true }),
                fc.double({ min: 0.4, max: 1.0, noNaN: true }),
                (ipd, fov, m, behind, distance) => {
                    const screen = screenFromDiagonal(14, 1.6);
                    const trueEye = { x: 0.12, y: 0, z: distance };
                    const believed = { width: screen.width * m, height: screen.height * m };
                    const ideal = screenXOf(trueEye, { x: 0, z: -behind }, screen);
                    // The renderer's idea of the eye: x scaled by the ipd error, z by the ipd and the focal length errors.
                    const drawn = screenXOf({ x: trueEye.x * ipd, y: 0, z: trueEye.z * ipd * fov }, { x: 0, z: -behind }, believed);
                    expect(drawn / ideal).toBeCloseTo(parallaxRatio(ipd, fov, m, behind, distance), 9);
                },
            ),
        );
    });

    test('with the priors this app uses, the depth effect stays within about 30% of the truth for a shoe 15 cm behind the glass', () => {
        // Interpupillary distance learned from the irises: +/-5%. Field of view 66 +/- 8 degrees: focal length -12%..+14%.
        // Screen size guessed from the CSS size and snapped to a common diagonal: +/-20%.
        let over = 1;
        let under = 1;
        for (const ipd of [0.95, 1, 1.05]) for (const fov of [0.88, 1, 1.14]) for (const m of [0.85, 0.92, 1, 1.1, 1.2]) {
            const r = parallaxRatio(ipd, fov, m, 0.15, 0.55);
            over = Math.max(over, r);
            under = Math.min(under, r);
        }
        expect(over).toBeLessThan(1.35);
        expect(under).toBeGreaterThan(0.72);
    });

    test('the screen size matters most, then the field of view, and the interpupillary distance hardly at all', () => {
        // A wrong interpupillary distance scales the whole eye position, and parallax depends on ratios of its
        // parts, so most of it cancels. A wrong field of view moves only the distance. A wrong screen size
        // rescales everything drawn on the glass.
        const swing = (f: (x: number) => number) => Math.abs(f(1.2) - f(0.85));
        const screenSwing = swing((m) => parallaxRatio(1, 1, m, 0.15, 0.55));
        const fovSwing = swing((k) => parallaxRatio(1, k, 1, 0.15, 0.55));
        const ipdSwing = swing((k) => parallaxRatio(k, 1, 1, 0.15, 0.55));
        expect(screenSwing).toBeGreaterThan(fovSwing);
        expect(fovSwing).toBeGreaterThan(ipdSwing * 3);
        expect(ipdSwing).toBeLessThan(0.1);
    });

    test('a point on the glass has no depth effect to get wrong', () => {
        // Only the screen-size factor remains, and it is the 1/m of how big the same movement looks.
        for (const [k, f, m] of [[0.9, 0.8, 1.3], [1.1, 1.25, 0.8]] as const) expect(parallaxRatio(k, f, m, 0, 0.6) * m).toBeCloseTo(k / (k * f), 12);
    });

    test('a wrong field of view is a bigger fraction of a shift near the glass, but the shift itself is tiny there, so the error in millimetres grows with depth', () => {
        const screen = screenFromDiagonal(14, 1.6);
        const eye = { x: 0.1, y: 0, z: 0.6 };
        const errorMm = (behind: number) => {
            const ideal = screenXOf(eye, { x: 0, z: -behind }, screen);
            const drawn = screenXOf({ x: eye.x, y: 0, z: eye.z * 1.3 }, { x: 0, z: -behind }, screen); // the field of view is wrong by 30%
            return Math.abs(drawn - ideal) * (screen.width / 2) * 1000;
        };
        expect(errorMm(0.05)).toBeLessThan(errorMm(0.15));
        expect(errorMm(0.15)).toBeLessThan(errorMm(0.3));
        expect(errorMm(0.05)).toBeLessThan(2); // under two millimetres on the glass for a point 5 cm behind it
        const fraction = (behind: number) => Math.abs(parallaxRatio(1, 1.3, 1, behind, 0.6) - 1);
        expect(fraction(0.05)).toBeGreaterThan(fraction(0.3));
    });
});

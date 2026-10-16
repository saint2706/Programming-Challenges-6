import type { Eye } from './offaxis.ts';

/** A face landmark as MediaPipe reports it: x and y as fractions of the image, z in the same scale as x. */
export interface Landmark {
    x: number;
    y: number;
    z: number;
}

/** The camera as far as this needs it: the frame it delivers, and how wide it sees. */
export interface CameraModel {
    width: number;
    height: number;
    /** Horizontal field of view in degrees. Browsers do not say; see calibration.ts for the prior. */
    hfovDeg: number;
}

/** Landmark numbers in the 478-point face mesh (the last ten are the irises). */
export const IRIS = {
    rightCenter: 468,
    rightRing: [469, 470, 471, 472],
    leftCenter: 473,
    leftRing: [474, 475, 476, 477],
} as const;

/** The visible iris is about 11.7 mm across in nearly everyone (the spread is a few percent). */
export const IRIS_DIAMETER_M = 0.0117;
/** The average adult interpupillary distance. People differ by several millimetres. */
export const DEFAULT_IPD_M = 0.063;
/** Beyond this the face is turned too far for the eye-to-eye distance to say how far away it is. */
export const MAX_YAW_DEG = 35;

export interface Measurement {
    /** Midpoint between the eyes in the camera's frame, metres: x to the image's right, y down, z away from the camera. */
    camera: { x: number; y: number; z: number };
    /** Pixel distance between the two iris centres, corrected for how far the face is turned. */
    ipdPx: number;
    /** Average iris diameter in pixels. */
    irisPx: number;
    /** How far the face is turned, in degrees. 0 is square to the camera. */
    yawDeg: number;
}

export function focalPx(camera: CameraModel): number {
    return camera.width / (2 * Math.tan((camera.hfovDeg * Math.PI) / 360));
}

const dist = (a: { x: number; y: number }, b: { x: number; y: number }) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Where the midpoint between the eyes is, from one frame's landmarks.
 *
 * Distance comes from the pinhole camera model and the one length that is known: the distance between
 * the pupils, `ipd`. Each eye is a ray through its pixel; MediaPipe also gives the eyes' depth
 * difference relative to the distance, `k`. With the midpoint at distance Z, the vector from one eye
 * to the other is Z * (A, B, k), where A and B come from the two pixel positions, so its length is
 * `Z * sqrt(A^2 + B^2 + k^2)` and setting that equal to `ipd` gives Z exactly. (The simpler `f * ipd /
 * pixels` is right only for a face square to the camera; measured on synthetic heads it is out by up
 * to 20% at 30 cm for a head turned 10 degrees, and 45% at 20 degrees.)
 *
 * Returns `null` when the face is turned too far, or the landmarks are not usable.
 */
export function measure(landmarks: readonly Landmark[], camera: CameraModel, ipd: number): Measurement | null {
    const right = landmarks[IRIS.rightCenter];
    const left = landmarks[IRIS.leftCenter];
    if (!right || !left) return null;
    const rp = { x: right.x * camera.width, y: right.y * camera.height };
    const lp = { x: left.x * camera.width, y: left.y * camera.height };
    const dzPx = (left.z - right.z) * camera.width;
    const flat = dist(rp, lp);
    const along = Math.hypot(flat, dzPx);
    if (!(along > 1)) return null;

    const f = focalPx(camera);
    const cx = camera.width / 2;
    const cy = camera.height / 2;
    const k = dzPx / f; // the eyes' depth difference as a fraction of the distance
    const ur = rp.x - cx;
    const ul = lp.x - cx;
    const vr = rp.y - cy;
    const vl = lp.y - cy;
    const a = (ul - ur + (k / 2) * (ul + ur)) / f;
    const b = (vl - vr + (k / 2) * (vl + vr)) / f;
    const z = ipd / Math.hypot(a, b, k);
    if (!(z > 0) || !Number.isFinite(z)) return null;
    // The eye-to-eye vector is `ipd` long and its depth part is k * z, so that is the turn of the head.
    const yaw = (Math.asin(Math.min(1, Math.abs(k) * z / ipd)) * 180) / Math.PI;
    if (yaw > MAX_YAW_DEG) return null;
    const midX = (rp.x + lp.x) / 2 - cx;
    const midY = (rp.y + lp.y) / 2 - cy;
    const ring = (indices: readonly number[]): number => {
        // The ring runs round the iris, so opposite points are two apart: (0,2) and (1,3).
        const p = indices.map((i) => landmarks[i]);
        if (p.some((q) => !q)) return NaN;
        const px = p.map((q) => ({ x: q!.x * camera.width, y: q!.y * camera.height }));
        return (dist(px[0]!, px[2]!) + dist(px[1]!, px[3]!)) / 2;
    };
    const irisPx = (ring(IRIS.rightRing) + ring(IRIS.leftRing)) / 2;
    return { camera: { x: (midX * z) / f, y: (midY * z) / f, z }, ipdPx: along, irisPx, yawDeg: yaw };
}

/**
 * The camera sits on the screen's top edge looking out at the viewer, so its picture is the screen's
 * view of the viewer: image-right is the viewer's left, which is the screen frame's -x. `cameraAbove`
 * is how far the lens is above the middle of the picture area.
 */
export function cameraToScreen(c: { x: number; y: number; z: number }, cameraAbove: number): Eye {
    return { x: -c.x, y: cameraAbove - c.y, z: c.z };
}

/**
 * Learns this person's own interpupillary distance. Distance needs it, and people differ by 5% or
 * so; but the iris is 11.7 mm in nearly everyone, so the ratio of pupil distance to iris size, seen
 * from the front, gives the distance in millimetres without asking. It is the median of many frames,
 * so a blink or a bad frame does not move it.
 */
export class IpdEstimator {
    private readonly samples: number[] = [];
    private readonly capacity: number;
    private readonly needed: number;

    constructor(capacity = 150, needed = 30) {
        this.capacity = capacity;
        this.needed = needed;
    }

    /** Offer one frame's measurement. Only frames square to the camera, with a sensible iris, count. */
    add(m: Measurement): void {
        if (m.yawDeg > 15 || !(m.irisPx >= 4 && m.irisPx <= 120)) return;
        const ipd = (IRIS_DIAMETER_M * m.ipdPx) / m.irisPx;
        if (!(ipd >= 0.05 && ipd <= 0.078)) return; // outside any adult face: a bad frame
        this.samples.push(ipd);
        if (this.samples.length > this.capacity) this.samples.shift();
    }

    /** The estimate, or the population average until there are enough good frames. */
    get value(): number {
        if (this.samples.length < this.needed) return DEFAULT_IPD_M;
        const s = [...this.samples].sort((a, b) => a - b);
        return s[Math.floor(s.length / 2)]!;
    }

    get learned(): boolean {
        return this.samples.length >= this.needed;
    }

    reset(): void {
        this.samples.length = 0;
    }
}

import { IRIS_DIAMETER_M, type CameraModel, type Landmark, focalPx } from '../src/headcoupled/pose.ts';

/**
 * Landmarks of a face as a camera would see them, built from a known head: the midpoint between the
 * eyes at `mid` (camera frame, metres: x to the image's right, y down, z away), eyes `ipd` apart, the
 * face turned `yawDeg`. The other 468 landmarks are zero; the code under test reads only the irises.
 */
export function faceLandmarks(mid: { x: number; y: number; z: number }, ipd: number, camera: CameraModel, yawDeg = 0, irisDiameter = IRIS_DIAMETER_M): Landmark[] {
    const f = focalPx(camera);
    const yaw = (yawDeg * Math.PI) / 180;
    const landmarks: Landmark[] = Array.from({ length: 478 }, () => ({ x: 0, y: 0, z: 0 }));
    const project = (p: { x: number; y: number; z: number }): Landmark => ({
        x: (p.x * f) / p.z / camera.width + 0.5,
        y: (p.y * f) / p.z / camera.height + 0.5,
        z: ((p.z - mid.z) * f) / mid.z / camera.width,
    });
    // "Right" is the person's right eye, which is on the image's left.
    const eye = (side: -1 | 1) => ({ x: mid.x + side * (ipd / 2) * Math.cos(yaw), y: mid.y, z: mid.z + side * (ipd / 2) * Math.sin(yaw) });
    const place = (centerIndex: number, ringIndices: readonly number[], side: -1 | 1) => {
        const c = eye(side);
        landmarks[centerIndex] = project(c);
        const r = irisDiameter / 2;
        // The iris is a disc facing the camera, so it narrows with the turn of the head.
        const offsets = [
            { x: r * Math.cos(yaw), y: 0 },
            { x: 0, y: -r },
            { x: -r * Math.cos(yaw), y: 0 },
            { x: 0, y: r },
        ];
        ringIndices.forEach((index, k) => (landmarks[index] = project({ x: c.x + offsets[k]!.x, y: c.y + offsets[k]!.y, z: c.z })));
    };
    place(468, [469, 470, 471, 472], -1);
    place(473, [474, 475, 476, 477], 1);
    return landmarks;
}

export const CAMERA: CameraModel = { width: 1280, height: 720, hfovDeg: 66 };

import { Matrix4 } from 'three';

/**
 * Head-coupled ("off-axis") projection, after Kooima, "Generalized Perspective Projection" (2009).
 *
 * The screen is a window onto a scene behind it. The viewer's eye is somewhere in front of the
 * window, and the picture that belongs on the window is whatever the eye would see through it: a
 * frustum whose far end is the scene and whose near end is the screen rectangle, which is lopsided
 * when the eye is off to one side. Everything is in metres, in a frame where the screen is the
 * rectangle `[-w/2, w/2] x [-h/2, h/2]` in the plane z = 0, x runs right, y runs up, and +z points
 * out of the screen towards the viewer. The scene lives at z < 0 (behind the glass), or a little in
 * front of it.
 */
export interface ScreenSize {
    /** Width and height of the picture area, in metres. */
    width: number;
    height: number;
}

export interface Eye {
    x: number;
    y: number;
    /** Distance in front of the screen. Must be positive. */
    z: number;
}

export interface Frustum {
    left: number;
    right: number;
    bottom: number;
    top: number;
    near: number;
    far: number;
}

/** Closest an eye is taken to be, so the maths never divides by something near zero. */
export const MIN_EYE_DISTANCE = 0.05;

/**
 * The frustum seen from `eye`, looking straight at the screen plane (the camera is not rotated, only
 * moved): the screen rectangle, scaled back from the eye onto the near plane.
 */
export function offAxisFrustum(screen: ScreenSize, eye: Eye, near: number, far: number): Frustum {
    const z = Math.max(eye.z, MIN_EYE_DISTANCE);
    const scale = near / z;
    return {
        left: (-screen.width / 2 - eye.x) * scale,
        right: (screen.width / 2 - eye.x) * scale,
        bottom: (-screen.height / 2 - eye.y) * scale,
        top: (screen.height / 2 - eye.y) * scale,
        near,
        far,
    };
}

/** Three.js's own constants: WebGL clips depth to [-1, 1], WebGPU to [0, 1]. */
export const WEBGL_DEPTH = 2000;
export const WEBGPU_DEPTH = 2001;

/** The projection matrix for a frustum, in the depth convention of the renderer in use. */
export function projectionMatrix(f: Frustum, coordinateSystem: number = WEBGL_DEPTH, out = new Matrix4()): Matrix4 {
    return out.makePerspective(f.left, f.right, f.top, f.bottom, f.near, f.far, coordinateSystem as never);
}

/** The pieces of a Three.js camera this needs, so tests can use a stand-in. */
export interface CameraLike {
    position: { set(x: number, y: number, z: number): unknown };
    projectionMatrix: Matrix4;
    projectionMatrixInverse: Matrix4;
    updateMatrixWorld(force?: boolean): void;
}

/**
 * Put a camera where the eye is and give it the off-axis projection. The camera keeps its default
 * orientation, looking down -z at the screen plane; only `position` and `projectionMatrix` change.
 */
export function applyOffAxis(camera: CameraLike, screen: ScreenSize, eye: Eye, near: number, far: number, coordinateSystem: number = WEBGL_DEPTH): Frustum {
    const frustum = offAxisFrustum(screen, eye, near, far);
    camera.position.set(eye.x, eye.y, Math.max(eye.z, MIN_EYE_DISTANCE));
    projectionMatrix(frustum, coordinateSystem, camera.projectionMatrix);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.updateMatrixWorld(true);
    return frustum;
}

/**
 * Where a point behind the screen lands on the screen, as a fraction of the half-width (so -1 is the
 * left edge): the line from the eye to the point, cut by the plane z = 0. This is the closed form the
 * matrix must agree with, and the reason the effect works: the nearer a point is to the glass, the
 * less it moves when the head does.
 */
export function screenXOf(eye: Eye, point: { x: number; z: number }, screen: ScreenSize): number {
    const t = eye.z / (eye.z - point.z); // how far along the eye-to-point line the glass is
    return ((eye.x + (point.x - eye.x) * t) / screen.width) * 2;
}

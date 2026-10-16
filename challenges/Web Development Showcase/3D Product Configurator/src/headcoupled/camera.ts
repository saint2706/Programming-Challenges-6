import { PerspectiveCamera } from 'three';
import { type Eye, type ScreenSize, MIN_EYE_DISTANCE, offAxisFrustum, projectionMatrix } from './offaxis.ts';

/**
 * A camera that sits at the viewer's eye and shows the scene through a window the size of the
 * canvas. Three.js (and the WebGPU renderer, whenever the depth convention of its backend differs
 * from the camera's) calls `updateProjectionMatrix`, so the off-axis matrix has to be what that
 * method makes, or the first frame would silently turn it back into an ordinary perspective.
 */
export class OffAxisCamera extends PerspectiveCamera {
    screen: ScreenSize | undefined;
    eye: Eye = { x: 0, y: 0, z: 0.6 };

    override updateProjectionMatrix(): void {
        // The base constructor calls this before the fields above exist.
        if (!this.screen) return super.updateProjectionMatrix();
        const frustum = offAxisFrustum(this.screen, this.eye, this.near, this.far);
        projectionMatrix(frustum, this.coordinateSystem, this.projectionMatrix);
        this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
    }

    /** Move the eye (metres, relative to the middle of the window) and say how big the window is. */
    place(eye: Eye, screen: ScreenSize): void {
        this.eye = { x: eye.x, y: eye.y, z: Math.max(eye.z, MIN_EYE_DISTANCE) };
        this.screen = screen;
        this.position.set(this.eye.x, this.eye.y, this.eye.z);
        this.updateProjectionMatrix();
        this.updateMatrixWorld(true);
    }
}

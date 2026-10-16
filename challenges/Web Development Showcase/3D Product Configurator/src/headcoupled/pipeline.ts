import { type Settings, cameraAbove, screenFromDiagonal } from './calibration.ts';
import { OneEuroVector3 } from './filter.ts';
import { type Eye } from './offaxis.ts';
import { type CameraModel, type Landmark, IpdEstimator, type Measurement, cameraToScreen, measure } from './pose.ts';

/** Where the head is taken to be before it has been seen, and after it has been lost for a while. */
export const REST_EYE: Eye = { x: 0, y: 0.02, z: 0.6 };
/** Nearer and further than this is not a person at a desk, it is a bad frame. */
export const EYE_Z_RANGE = [0.2, 1.6] as const;
const LOST_AFTER_S = 1;
const EASE_S = 0.5;

export interface Update {
    /** The eye in the screen's frame (metres from the middle of the picture area, z in front of it). */
    eye: Eye;
    /** The head was seen in this frame. */
    found: boolean;
    measurement: Measurement | null;
}

/**
 * From a frame's landmarks to the eye position the picture is drawn for, in the screen's frame:
 * measure it, learn this person's interpupillary distance, scale the sideways movement by the
 * person's depth setting, smooth it. When the head is lost it stays where it was for a second and
 * then drifts back to the middle, so a lost face never leaves the picture stuck at an angle.
 */
export class HeadPipeline {
    readonly ipd = new IpdEstimator();
    private readonly filter = new OneEuroVector3();
    private eye: Eye = { ...REST_EYE };
    private lastSeen = -Infinity;
    private lastTime = 0;
    settings: Settings;

    constructor(settings: Settings) {
        this.settings = settings;
    }

    /** `landmarks` is null when no face was found. `time` is in seconds and must not go backwards. */
    update(landmarks: readonly Landmark[] | null, camera: Pick<CameraModel, 'width' | 'height'>, time: number): Update {
        const dt = Math.max(0, time - this.lastTime);
        this.lastTime = time;
        const m = landmarks ? measure(landmarks, { ...camera, hfovDeg: this.settings.hfov }, this.ipd.value) : null;
        if (m) {
            this.ipd.add(m);
            const screen = screenFromDiagonal(this.settings.diagonal, this.settings.aspect);
            const e = cameraToScreen(m.camera, cameraAbove(screen));
            const z = Math.min(EYE_Z_RANGE[1], Math.max(EYE_Z_RANGE[0], e.z));
            this.eye = this.filter.filter({ x: e.x * this.settings.depth, y: e.y * this.settings.depth, z }, time);
            this.lastSeen = time;
            return { eye: this.eye, found: true, measurement: m };
        }
        if (time - this.lastSeen > LOST_AFTER_S) {
            const k = 1 - Math.exp(-dt / EASE_S);
            this.eye = { x: this.eye.x + (REST_EYE.x - this.eye.x) * k, y: this.eye.y + (REST_EYE.y - this.eye.y) * k, z: this.eye.z + (REST_EYE.z - this.eye.z) * k };
            this.filter.reset();
        }
        return { eye: this.eye, found: false, measurement: null };
    }

    reset(): void {
        this.eye = { ...REST_EYE };
        this.lastSeen = -Infinity;
        this.filter.reset();
        this.ipd.reset();
    }
}

import { type Settings, defaults, parseSettings } from './calibration.ts';
import { type Eye } from './offaxis.ts';
import { HeadPipeline, REST_EYE } from './pipeline.ts';
import { type Landmark } from './pose.ts';
import { type FaceTracker, createTracker } from './tracker.ts';

const STORAGE_KEY = 'configurator:head:v1';

export type HeadStatus = 'off' | 'starting' | 'tracking' | 'no-face' | 'denied' | 'unavailable' | 'error';

/** The page's own best guess about the screen: its size, and whether it is a Mac laptop's. */
export function readDisplay(): { width: number; height: number; macLaptop: boolean } {
    const platform = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform ?? navigator.platform ?? '';
    return { width: screen.width, height: screen.height, macLaptop: /mac/i.test(platform) && !/iphone|ipad/i.test(navigator.userAgent) && Math.max(screen.width, screen.height) <= 1800 };
}

function loadSettings(guess: Settings): Settings {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? parseSettings(JSON.parse(raw), guess) : guess;
    } catch {
        return guess;
    }
}

/**
 * The webcam, the face model, and what they say about where the head is. The camera runs only
 * between `start()` and `stop()` (and stops by itself when the tab is hidden), and nothing it sees
 * leaves the page: the frames go to a model that runs here, and only a few numbers come out.
 */
export class HeadSession {
    status = $state<HeadStatus>('off');
    note = $state('');
    delegate = $state<'' | 'GPU' | 'CPU'>('');
    fps = $state(0);
    inferenceMs = $state(0);
    distanceCm = $state(0);
    yawDeg = $state(0);
    ipdMm = $state(63);
    ipdLearned = $state(false);
    /** Whether the settings are the page's guesses, or have been chosen. */
    chosen = $state(false);
    settings: Settings = $state(defaults(1440, 900));

    /** The newest landmarks, for the preview's overlay. Not reactive: it changes every frame. */
    landmarks: Landmark[] | null = null;
    frameSize = { width: 0, height: 0 };
    /** The eye in the screen's frame, as filtered. Read every frame by the renderer. */
    eye: Eye = { ...REST_EYE };
    stream = $state.raw<MediaStream | null>(null);
    readonly video: HTMLVideoElement;

    private readonly pipeline: HeadPipeline;
    private tracker: FaceTracker | null = null;
    private loading: Promise<FaceTracker> | null = null;
    private frame = 0;
    private videoFrame = 0;
    /** Counts starts and stops: an answer that arrives for an older one (a permission given late, a model finished loading) is for a camera nobody wants any more. */
    private run = 0;
    private lastStamp = 0;
    private fpsAt = 0;
    private frames = 0;
    // The camera does not come back by itself: it was a click that turned it on.
    private readonly onVisibility = () => {
        if (document.hidden && (this.status === 'tracking' || this.status === 'no-face' || this.status === 'starting')) this.stop('The camera was turned off while this tab was in the background. Press Use my camera to carry on.');
    };

    constructor() {
        const display = typeof screen === 'undefined' ? { width: 1440, height: 900, macLaptop: false } : readDisplay();
        const guess = defaults(display.width, display.height, display.macLaptop);
        this.settings = loadSettings(guess);
        this.chosen = this.settings.diagonal !== guess.diagonal || this.settings.depth !== guess.depth || this.settings.hfov !== guess.hfov;
        this.pipeline = new HeadPipeline(this.settings);
        this.video = document.createElement('video');
        this.video.muted = true;
        this.video.playsInline = true;
        this.video.setAttribute('aria-hidden', 'true');
        this.video.className = 'head-video';
        document.addEventListener('visibilitychange', this.onVisibility);
    }

    /** Remember what the person chose. */
    choose(change: Partial<Pick<Settings, 'diagonal' | 'hfov' | 'depth'>>): void {
        Object.assign(this.settings, change);
        this.chosen = true;
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify({ diagonal: this.settings.diagonal, hfov: this.settings.hfov, depth: this.settings.depth }));
        } catch {
            // Not saved: the page still works.
        }
    }

    async start(): Promise<void> {
        if (this.status === 'starting' || this.status === 'tracking' || this.status === 'no-face') return;
        if (!navigator.mediaDevices?.getUserMedia) {
            this.status = 'unavailable';
            this.note = 'This browser cannot open a camera here (it needs https or localhost).';
            return;
        }
        const run = ++this.run;
        this.status = 'starting';
        this.note = 'Asking for the camera…';
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' }, audio: false });
            if (run !== this.run) {
                for (const t of stream.getTracks()) t.stop(); // turned off while it was asking, and permission came late
                return;
            }
            this.stream = stream;
            for (const t of stream.getVideoTracks()) t.addEventListener('ended', () => run === this.run && this.stop('The camera stopped (it was unplugged, or sharing was ended in the browser).'));
            this.video.srcObject = stream;
            this.note = 'Loading the face model…';
            this.loading ??= createTracker();
            const [tracker] = await Promise.all([this.loading, this.video.play()]);
            if (run !== this.run) return; // stop() has already let go of the stream
            this.tracker = tracker;
            this.delegate = tracker.delegate;
            this.status = 'no-face';
            this.note = 'Looking for a face…';
            this.frames = 0;
            this.fpsAt = performance.now();
            this.schedule();
        } catch (error) {
            if (run !== this.run) return; // an older attempt failing says nothing about the current one
            this.loading = null;
            this.release();
            const name = error instanceof DOMException ? error.name : '';
            if (name === 'NotAllowedError' || name === 'SecurityError') {
                this.status = 'denied';
                this.note = 'The camera was blocked. You can allow it from the address bar, or keep using the pointer.';
            } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
                this.status = 'unavailable';
                this.note = 'No camera was found.';
            } else {
                this.status = 'error';
                this.note = `The face tracker could not start: ${error instanceof Error ? error.message : String(error)}`;
            }
        }
    }

    stop(note = ''): void {
        this.run++;
        this.release();
        this.status = 'off';
        this.note = note;
        this.fps = 0;
        this.landmarks = null;
        this.eye = { ...REST_EYE };
    }

    dispose(): void {
        document.removeEventListener('visibilitychange', this.onVisibility);
        this.stop();
        this.tracker?.close();
        this.tracker = null;
        this.video.remove();
    }

    /** Feed one frame's landmarks (or `null` for no face) as if the camera had seen them. For tests and demos. */
    inject(landmarks: readonly Landmark[] | null, size = { width: 1280, height: 720 }): void {
        if (this.status === 'off') this.status = 'no-face';
        this.frameSize = size;
        this.handle(landmarks, size, performance.now() / 1000);
    }

    private release(): void {
        cancelAnimationFrame(this.frame);
        if (this.videoFrame) this.video.cancelVideoFrameCallback?.(this.videoFrame);
        this.videoFrame = 0;
        if (this.stream) for (const t of this.stream.getTracks()) t.stop();
        this.stream = null;
        this.video.pause();
        this.video.srcObject = null;
        this.frame = 0;
    }

    private schedule(): void {
        const run = this.run;
        const next = (loop: (now: number) => void) => {
            if (this.video.requestVideoFrameCallback) this.videoFrame = this.video.requestVideoFrameCallback(loop);
            else this.frame = requestAnimationFrame(loop);
        };
        const loop = (now: number) => {
            if (run !== this.run || (this.status !== 'tracking' && this.status !== 'no-face')) return;
            this.step(now);
            next(loop);
        };
        next(loop);
    }

    private step(now: number): void {
        const { video, tracker } = this;
        if (!tracker || video.readyState < 2 || !video.videoWidth) return;
        const stamp = Math.max(this.lastStamp + 1, Math.round(performance.now()));
        this.lastStamp = stamp;
        const started = performance.now();
        let landmarks: Landmark[] | null = null;
        try {
            landmarks = tracker.detect(video, stamp);
        } catch (error) {
            this.release();
            this.status = 'error';
            this.note = `The face tracker stopped: ${error instanceof Error ? error.message : String(error)}`;
            return;
        }
        this.inferenceMs = Math.round((performance.now() - started) * 10) / 10;
        this.delegate = tracker.delegate;
        this.frames++;
        if (now - this.fpsAt >= 1000) {
            this.fps = Math.round((this.frames * 1000) / (now - this.fpsAt));
            this.frames = 0;
            this.fpsAt = now;
        }
        const size = { width: video.videoWidth, height: video.videoHeight };
        this.frameSize = size;
        this.handle(landmarks, size, now / 1000);
    }

    private handle(landmarks: readonly Landmark[] | null, size: { width: number; height: number }, seconds: number): void {
        this.landmarks = landmarks ? [...landmarks] : null;
        const update = this.pipeline.update(landmarks, size, seconds);
        this.eye = update.eye;
        if (update.found && update.measurement) {
            this.status = 'tracking';
            this.note = '';
            this.distanceCm = Math.round(update.measurement.camera.z * 100);
            this.yawDeg = Math.round(update.measurement.yawDeg);
        } else {
            this.status = 'no-face';
            this.note = landmarks ? 'Your head is turned too far to measure.' : 'Looking for a face…';
        }
        this.ipdMm = Math.round(this.pipeline.ipd.value * 1000);
        this.ipdLearned = this.pipeline.ipd.learned;
    }
}

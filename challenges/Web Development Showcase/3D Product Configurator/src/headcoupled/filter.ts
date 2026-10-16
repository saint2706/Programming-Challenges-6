/**
 * The 1-euro filter (Casiez, Roussel and Vogel, CHI 2012): a low-pass filter whose cutoff rises with
 * speed. When the signal is still it smooths hard, so the picture does not shimmer with tracking
 * noise; when it moves it lets go, so there is almost no lag. A plain moving average cannot do both.
 */
export interface OneEuroOptions {
    /** Cutoff in hertz when the signal is still. Lower is smoother and slower to follow. */
    minCutoff: number;
    /** How strongly speed raises the cutoff. Higher means less lag in motion. */
    beta: number;
    /** Cutoff for the speed estimate itself, in hertz. */
    derivativeCutoff: number;
}

export const HEAD_FILTER: OneEuroOptions = { minCutoff: 1.2, beta: 8, derivativeCutoff: 1 };

/** Smoothing factor for a cutoff and a time step: how much of the new value to take. */
const alpha = (cutoff: number, dt: number): number => {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
};

export class OneEuroFilter {
    private readonly options: OneEuroOptions;
    private last: number | null = null;
    private lastRate = 0;
    private lastTime = 0;

    constructor(options: OneEuroOptions = HEAD_FILTER) {
        this.options = options;
    }

    /** Filter one sample taken at `time` (seconds). Samples out of order or at the same instant are passed through. */
    filter(value: number, time: number): number {
        if (this.last === null || !Number.isFinite(this.last)) {
            this.last = value;
            this.lastTime = time;
            this.lastRate = 0;
            return value;
        }
        const dt = time - this.lastTime;
        if (!(dt > 0)) return this.last;
        this.lastTime = time;
        const rate = (value - this.last) / dt;
        const smoothRate = this.lastRate + alpha(this.options.derivativeCutoff, dt) * (rate - this.lastRate);
        this.lastRate = smoothRate;
        const cutoff = this.options.minCutoff + this.options.beta * Math.abs(smoothRate);
        this.last = this.last + alpha(cutoff, dt) * (value - this.last);
        return this.last;
    }

    reset(): void {
        this.last = null;
    }
}

/** One filter per axis, for a position. */
export class OneEuroVector3 {
    private readonly axes: [OneEuroFilter, OneEuroFilter, OneEuroFilter];

    constructor(options: OneEuroOptions = HEAD_FILTER) {
        this.axes = [new OneEuroFilter(options), new OneEuroFilter(options), new OneEuroFilter(options)];
    }

    filter(v: { x: number; y: number; z: number }, time: number): { x: number; y: number; z: number } {
        return { x: this.axes[0].filter(v.x, time), y: this.axes[1].filter(v.y, time), z: this.axes[2].filter(v.z, time) };
    }

    reset(): void {
        for (const a of this.axes) a.reset();
    }
}

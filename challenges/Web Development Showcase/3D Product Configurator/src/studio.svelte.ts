import { type Design, type FinishId, type PartId, PRESETS, defaultDesign, normalizeHex, parseDesign } from './parts.ts';

const STORAGE_KEY = 'configurator:design:v1';

export type Mode = 'design' | 'window';

/** Which drawing path ended up in use (the page asks for WebGPU and takes WebGL 2 where there is none). */
export type Backend = 'WebGPU' | 'WebGL 2' | '';

function load(): Design {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return (raw && parseDesign(JSON.parse(raw))) || defaultDesign();
    } catch {
        return defaultDesign();
    }
}

/** What the person has chosen, and what is on screen. Everything the interface and the scene share. */
export class Studio {
    design = $state<Design>(load());
    selected = $state<PartId>('upper');
    mode = $state<Mode>('design');
    hud = $state(false);
    /** The shoe is loaded and has been drawn at least once. */
    ready = $state(false);
    error = $state('');
    backend = $state<Backend>('');
    /** Frames per second of the drawing, measured while the window preview runs. */
    fps = $state(0);
    /** How big the shoe is drawn in the window preview, as a fraction of life size. */
    shoeScale = $state(1);
    /** An eye position to use instead of the camera or the pointer (canvas frame, metres): for tests and demos. */
    eyeOverride: { x: number; y: number; z: number } | null = null;
    /** The eye the window preview was last drawn for, and the size of its window, for the readout. */
    seen = { eye: { x: 0, y: 0, z: 0.6 }, width: 0, height: 0, source: 'pointer' as 'camera' | 'pointer' | 'override' };
    /** What the renderer is holding on the graphics card, for the tests that look for leaks. */
    memory: () => { geometries: number; textures: number } = () => ({ geometries: 0, textures: 0 });
    /** Frames drawn since the page started. */
    frames = 0;
    /** Set by the camera rig: turn or zoom the view of the design, for the keyboard. */
    orbit: (azimuth: number, polar: number, zoom: number) => void = () => {};

    private save(): void {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify($state.snapshot(this.design)));
        } catch {
            // Not saved: the page still works.
        }
    }

    setColor(part: PartId, colour: string): void {
        const hex = normalizeHex(colour);
        if (!hex) return;
        this.design.colors[part] = hex;
        this.save();
    }

    setFinish(part: PartId, finish: FinishId): void {
        this.design.finishes[part] = finish;
        this.save();
    }

    applyPreset(id: string): void {
        const preset = PRESETS.find((p) => p.id === id);
        if (!preset) return;
        this.design.colors = { ...preset.colors };
        this.save();
    }

    reset(): void {
        this.design = defaultDesign();
        this.save();
    }
}

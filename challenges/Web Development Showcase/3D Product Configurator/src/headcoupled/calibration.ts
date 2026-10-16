import type { ScreenSize } from './offaxis.ts';

/**
 * What the page can know about the screen and the camera without asking, and how wrong it can be.
 *
 * Off-axis projection needs the screen's physical size and where the viewer's eyes are. A browser
 * can tell neither. Rather than make people hold a credit card up to the glass, this starts from
 * priors, and `parallaxRatio` below says how much the picture suffers when a prior is off: very
 * little, because the screen plane itself is exact by construction (see offaxis.ts), and only the
 * strength of the depth effect depends on the guesses.
 */

/** CSS pixels per inch of a typical display at its default OS scaling: Windows laptops 110-126, monitors 90-110. */
export const CSS_PPI_PRIOR = 110;
/** Apple's laptops run at 127 CSS pixels per inch whatever the model, which is the one place this is exact. */
export const MAC_LAPTOP_PPI = 127;

/** Screen diagonals that people recognise, in inches. */
export const COMMON_DIAGONALS = [11.6, 13.3, 14, 15.6, 16, 17.3, 21.5, 24, 27, 32] as const;

/** Webcams in laptops see about 66 degrees across (diagonal 70-80). */
export const HFOV_PRIOR_DEG = 66;

/** Between the lens and the top of the picture there is a bezel, about a centimetre. */
export const BEZEL_M = 0.01;

const INCH = 0.0254;

/** The picture area of a screen with this diagonal and shape. */
export function screenFromDiagonal(diagonalInches: number, aspect: number): ScreenSize {
    const diagonal = diagonalInches * INCH;
    const height = diagonal / Math.hypot(aspect, 1);
    return { width: height * aspect, height };
}

/**
 * The nearest common diagonal to what the CSS size of the screen suggests. `macLaptop` is true for a
 * Mac with a built-in-sized screen, where the pixel density is known.
 */
export function guessDiagonal(cssWidth: number, cssHeight: number, macLaptop = false): number {
    const raw = Math.hypot(cssWidth, cssHeight) / (macLaptop ? MAC_LAPTOP_PPI : CSS_PPI_PRIOR);
    return COMMON_DIAGONALS.reduce((best, d) => (Math.abs(d - raw) < Math.abs(best - raw) ? d : best), COMMON_DIAGONALS[0]);
}

/** How far above the middle of the picture area the lens is. */
export function cameraAbove(screen: ScreenSize, bezel = BEZEL_M): number {
    return screen.height / 2 + bezel;
}

export interface Settings {
    /** Screen diagonal in inches, and the shape (width / height) of the picture. */
    diagonal: number;
    aspect: number;
    /** The camera's horizontal field of view in degrees. */
    hfov: number;
    /**
     * A multiplier on how far the head is taken to have moved sideways and up or down: the one knob for
     * "the depth effect feels too weak or too strong", whatever the cause.
     */
    depth: number;
}

export function defaults(cssWidth: number, cssHeight: number, macLaptop = false): Settings {
    return { diagonal: guessDiagonal(cssWidth, cssHeight, macLaptop), aspect: cssWidth / cssHeight, hfov: HFOV_PRIOR_DEG, depth: 1 };
}

export const LIMITS = { diagonal: [7, 60], hfov: [40, 110], depth: [0.3, 3] } as const;

const within = (value: unknown, [min, max]: readonly [number, number], fallback: number): number => (typeof value === 'number' && Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback);

/** Saved settings (which come from storage, so are not to be trusted) laid over the guesses. The shape of the screen is never saved: it is read off the screen each time. */
export function parseSettings(raw: unknown, guess: Settings): Settings {
    if (typeof raw !== 'object' || raw === null) return guess;
    const r = raw as Record<string, unknown>;
    return { ...guess, diagonal: within(r.diagonal, LIMITS.diagonal, guess.diagonal), hfov: within(r.hfov, LIMITS.hfov, guess.hfov), depth: within(r.depth, LIMITS.depth, guess.depth) };
}

/**
 * How much stronger (above 1) or weaker (below 1) the rendered depth effect is than the true one, for
 * a point `behind` metres behind the screen when the viewer is `distance` metres in front of it.
 *
 * Three guesses can be off, and they do different things:
 *
 * - `ipdScale`: the interpupillary distance used divided by the true one. Position is worked out as
 *   (pixels x ipd / eye-to-eye pixels), so a wrong ipd scales x, y and z alike.
 * - `fovScale`: the focal length used divided by the true one. It cancels out of x and y (a pixel
 *   offset times ipd over eye-to-eye pixels) and scales only z, the distance.
 * - `screenScale`: the assumed screen width divided by the true one. The scene is drawn in metres, so
 *   a screen believed to be larger shows the same movement smaller.
 *
 * At 1, 1 and 1 the ratio is exactly 1.
 *
 * Derivation: the true shift on the glass of a point at depth d is e.x * d / (e.z + d). The renderer
 * believes the head is at (k*e.x, k*e.y, k*f*e.z) with k = ipdScale and f = fovScale, so it shifts the
 * point by k*e.x * d / (k*f*e.z + d), and shows it 1/m as large when the screen is believed m times wider.
 */
export function parallaxRatio(ipdScale: number, fovScale: number, screenScale: number, behind: number, distance: number): number {
    return ((ipdScale * (distance + behind)) / (ipdScale * fovScale * distance + behind)) / screenScale;
}

// "Warming stripes": one vertical stripe per year, coloured by its anomaly on a diverging scale.
import { linearScale } from './scale.js';

// ColorBrewer RdBu (11 classes), cold to warm. Neutral (0 °C anomaly) is the middle stop.
const RDBU = [
    [5, 48, 97],
    [33, 102, 172],
    [67, 147, 195],
    [146, 197, 222],
    [209, 229, 240],
    [247, 247, 247],
    [253, 219, 199],
    [244, 165, 130],
    [214, 96, 77],
    [178, 24, 43],
    [103, 0, 31],
];

/** Hex colour for an anomaly; `limit` is the anomaly that reaches the darkest stop (both signs). */
export function stripeColor(value, limit = 1.3) {
    const clamped = Math.min(limit, Math.max(-limit, value));
    const t = linearScale([-limit, limit], [0, RDBU.length - 1])(clamped);
    const i = Math.min(RDBU.length - 2, Math.floor(t));
    const f = t - i;
    const rgb = RDBU[i].map((c, k) => Math.round(c + (RDBU[i + 1][k] - c) * f));
    return `#${rgb.map((c) => c.toString(16).padStart(2, '0')).join('')}`;
}

/** The strip markup. `--stripes` lets the CSS size the strip for the pinned horizontal scroll. */
export function stripesMarkup(years) {
    const cells = years
        .map(
            ({ year, value }) =>
                `<span class="stripe" data-year="${year}" style="background:${stripeColor(value)}"></span>`,
        )
        .join('');
    return `<div class="strip" style="--stripes:${years.length}">${cells}</div>`;
}

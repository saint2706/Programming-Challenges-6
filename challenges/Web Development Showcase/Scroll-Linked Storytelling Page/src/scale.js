// Linear scales and SVG path strings. Pure, so the chart geometry is testable without a DOM.

/** Map [d0, d1] onto [r0, r1]. Values outside the domain extrapolate (callers clamp if needed). */
export function linearScale([d0, d1], [r0, r1]) {
    const k = (r1 - r0) / (d1 - d0);
    return (value) => r0 + (value - d0) * k;
}

/** "M x y L x y ..." with one decimal, which is plenty at 800 px wide and keeps the markup small. */
export function linePath(points) {
    return points
        .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`)
        .join(' ');
}

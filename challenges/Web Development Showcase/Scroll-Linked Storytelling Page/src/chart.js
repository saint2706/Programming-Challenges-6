// The line chart as SVG markup. The same string is written into index.html at build time (so the
// chart exists without JavaScript) and the highlight band and note are moved later by the story.
import { linePath, linearScale } from './scale.js';

export const CHART = {
    width: 800,
    height: 440,
    left: 56,
    right: 24,
    top: 24,
    bottom: 40,
    years: [1880, 2025],
    values: [-0.6, 1.4],
};

const xScale = linearScale(CHART.years, [CHART.left, CHART.width - CHART.right]);
const yScale = linearScale(CHART.values, [CHART.height - CHART.bottom, CHART.top]);

export const xOf = (year) => xScale(year);
export const yOf = (value) => yScale(value);

const Y_TICKS = [-0.5, 0, 0.5, 1];
const X_TICKS = [1880, 1900, 1920, 1940, 1960, 1980, 2000, 2020];

/**
 * @param {{ year: number, value: number }[]} years
 * @returns {string} an <svg> element. Only numbers and constants go into it, never user text.
 */
export function chartMarkup(years) {
    const grid = Y_TICKS.map(
        (t) =>
            `<line class="grid" x1="${CHART.left}" x2="${CHART.width - CHART.right}" y1="${yOf(t).toFixed(1)}" y2="${yOf(t).toFixed(1)}"/>` +
            `<text class="tick" x="${CHART.left - 8}" y="${(yOf(t) + 4).toFixed(1)}" text-anchor="end">${t > 0 ? '+' : t < 0 ? '−' : ''}${Math.abs(t).toFixed(1)}</text>`,
    ).join('');
    const xLabels = X_TICKS.map(
        (year) =>
            `<text class="tick" x="${xOf(year).toFixed(1)}" y="${CHART.height - 14}" text-anchor="middle">${year}</text>`,
    ).join('');
    const points = years.map(({ year, value }) => [xOf(year), yOf(value)]);
    const last = years.at(-1);
    const bandTop = CHART.top;
    const bandHeight = CHART.height - CHART.top - CHART.bottom;
    return (
        `<svg class="chart-svg" viewBox="0 0 ${CHART.width} ${CHART.height}" role="img" ` +
        `aria-labelledby="chart-svg-title chart-svg-desc">` +
        `<title id="chart-svg-title">Global mean temperature anomaly, ${years[0].year} to ${last.year}</title>` +
        `<desc id="chart-svg-desc">Annual anomaly in degrees Celsius against the 1951 to 1980 average. ` +
        `The same numbers are in the table below the chart.</desc>` +
        `<g class="grid-layer">${grid}</g>` +
        `<line class="zero" x1="${CHART.left}" x2="${CHART.width - CHART.right}" y1="${yOf(0).toFixed(1)}" y2="${yOf(0).toFixed(1)}"/>` +
        `<g class="axis-x">${xLabels}</g>` +
        `<rect class="band" x="${CHART.left}" y="${bandTop}" width="0" height="${bandHeight}"/>` +
        `<path class="chart-line" pathLength="1" d="${linePath(points)}"/>` +
        `<circle class="chart-end" cx="${xOf(last.year).toFixed(1)}" cy="${yOf(last.value).toFixed(1)}" r="5"/>` +
        `<text class="chart-note" x="${CHART.left + 12}" y="${CHART.top + 22}">Scroll to draw the line</text>` +
        `</svg>`
    );
}

/** The same numbers as a table, for screen readers, copy-paste and anyone who wants exact values. */
export function dataTableMarkup(years) {
    const rows = years
        .map(({ year, value }) => `<tr><th scope="row">${year}</th><td>${value.toFixed(2)}</td></tr>`)
        .join('');
    return (
        `<table class="data-table"><caption>Annual mean anomaly, degrees Celsius against 1951 to 1980</caption>` +
        `<thead><tr><th scope="col">Year</th><th scope="col">Anomaly (°C)</th></tr></thead>` +
        `<tbody>${rows}</tbody></table>`
    );
}

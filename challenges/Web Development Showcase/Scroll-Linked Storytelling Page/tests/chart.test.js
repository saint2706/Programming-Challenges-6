import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import data from '../src/data/gistemp-annual.json';
import { CHART, chartMarkup, dataTableMarkup, xOf, yOf } from '../src/chart.js';

describe('chart geometry', () => {
    it('x grows left to right across the plot, y grows upward (warmer is higher)', () => {
        expect(xOf(1880)).toBe(CHART.left);
        expect(xOf(2025)).toBe(CHART.width - CHART.right);
        expect(yOf(1.4)).toBeLessThan(yOf(-0.6));
    });

    it('property: xOf is strictly increasing in year', () => {
        fc.assert(
            fc.property(fc.integer({ min: 1880, max: 2024 }), (y) => {
                expect(xOf(y + 1)).toBeGreaterThan(xOf(y));
            }),
        );
    });
});

describe('chartMarkup', () => {
    it('draws the full series as one path with pathLength=1 so the dash can be scrubbed 0 to 1', () => {
        const svg = chartMarkup(data.years);
        expect(svg.match(/class="chart-line"/g)).toHaveLength(1);
        expect(svg).toContain('pathLength="1"');
        const d = svg.match(/class="chart-line" pathLength="1" d="([^"]+)"/)[1];
        expect(d.match(/[ML]/g)).toHaveLength(146);
    });

    it('starts with a band of zero width and an accessible name and description', () => {
        const svg = chartMarkup(data.years);
        expect(svg).toContain('class="band" x="56" y="24" width="0"');
        expect(svg).toContain('aria-labelledby="chart-svg-title chart-svg-desc"');
        expect(svg).toContain('Global mean temperature anomaly, 1880 to 2025');
    });

    it('property: for any short series the path has one segment per point', () => {
        fc.assert(
            fc.property(
                fc.array(fc.double({ min: -1, max: 2, noNaN: true }), { minLength: 2, maxLength: 60 }),
                (values) => {
                    const years = values.map((value, i) => ({ year: 1880 + i, value }));
                    const d = chartMarkup(years).match(/ d="([^"]+)"/)[1];
                    expect(d.match(/[ML]/g)).toHaveLength(values.length);
                },
            ),
        );
    });
});

describe('dataTableMarkup', () => {
    it('has one row per year plus the header, with two decimals', () => {
        const html = dataTableMarkup(data.years);
        expect(html.match(/<tr>/g)).toHaveLength(146 + 1);
        expect(html).toContain('<td>1.18</td>');
    });
});

// The build writes these markers into index.html; each must appear exactly once.
describe('index.html placeholders', () => {
    const html = readFileSync(resolve('index.html'), 'utf8');
    it.each(['<!--chart-svg-->', '<!--data-table-->', '<!--stripes-->'])('%s appears once', (marker) => {
        expect(html.split(marker)).toHaveLength(2);
    });
});

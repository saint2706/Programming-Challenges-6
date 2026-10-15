import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import data from '../src/data/gistemp-annual.json';
import { formatAnomaly, parseGistemp, rangeMean } from '../src/gistemp.js';

const raw = readFileSync(resolve('data/GLB.Ts+dSST.csv'), 'utf8');
const get = (year) => data.years.find((y) => y.year === year).value;

describe('parseGistemp', () => {
    it('reads the committed GISS file: 146 complete years, 1880 to 2025, no 2026 partial row', () => {
        const { years, title } = parseGistemp(raw);
        expect(title).toBe('Land-Ocean: Global Means');
        expect(years).toHaveLength(146);
        expect(years[0]).toEqual({ year: 1880, value: -0.17 });
        expect(years.at(-1)).toEqual({ year: 2025, value: 1.18 });
    });

    it('drops rows whose annual mean is missing (***) instead of zeroing them', () => {
        const csv = 'T\nYear,Jan,J-D\n2000,.1,.5\n2001,.1,***\n2002,.1,-.2\n';
        expect(parseGistemp(csv).years).toEqual([
            { year: 2000, value: 0.5 },
            { year: 2002, value: -0.2 },
        ]);
    });

    it('fails loudly on a file without the expected header or column', () => {
        expect(() => parseGistemp('nothing here')).toThrow(/header/);
        expect(() => parseGistemp('T\nYear,Jan,DJF\n2000,1\n')).toThrow(/J-D/);
    });

    it('the committed JSON is exactly what the parser gives for the committed CSV', () => {
        expect(data.years).toEqual(parseGistemp(raw).years);
    });
});

describe('narrative claims match the data', () => {
    // These are the numbers printed in index.html. If the data changes, this fails first.
    it('step 1: no year from 1880 to 1910 is above zero; mean -0.26', () => {
        const window = data.years.filter((y) => y.year <= 1910);
        expect(window).toHaveLength(31);
        expect(window.some((y) => y.value > 0)).toBe(false);
        expect(formatAnomaly(rangeMean(data.years, 1880, 1910))).toBe('−0.26 °C');
    });

    it('step 2: 1910 at -0.44 and 1940 at +0.12', () => {
        expect(get(1910)).toBe(-0.44);
        expect(get(1940)).toBe(0.12);
    });

    it('step 3: the 1940 to 1975 average is -0.01', () => {
        expect(formatAnomaly(rangeMean(data.years, 1940, 1975))).toBe('−0.01 °C');
    });

    it('step 4: 1976 at -0.10 and 2000 at +0.39', () => {
        expect(get(1976)).toBe(-0.1);
        expect(get(2000)).toBe(0.39);
    });

    it('step 5: 2001 to 2025 averages +0.80; the years above +1.0 are 2016, 2020, 2023, 2024, 2025', () => {
        expect(formatAnomaly(rangeMean(data.years, 2001, 2025))).toBe('+0.80 °C');
        const above = data.years.filter((y) => y.value > 1).map((y) => y.year);
        expect(above).toEqual([2016, 2020, 2023, 2024, 2025]);
    });

    it('2024 is the warmest year at +1.28, and the line ends at +1.18 in 2025', () => {
        const warmest = data.years.reduce((a, b) => (b.value > a.value ? b : a));
        expect(warmest).toEqual({ year: 2024, value: 1.28 });
        expect(get(2025)).toBe(1.18);
    });
});

describe('formatAnomaly and rangeMean', () => {
    it('uses a true minus sign, a plus for positives and no sign at zero', () => {
        expect(formatAnomaly(-0.256)).toBe('−0.26 °C');
        expect(formatAnomaly(1.2)).toBe('+1.20 °C');
        expect(formatAnomaly(0)).toBe('0.00 °C');
    });

    it('rangeMean is null for an empty range', () => {
        expect(rangeMean(data.years, 1000, 1100)).toBeNull();
    });

    it('property: formatAnomaly is always a signed two-decimal value in degrees', () => {
        fc.assert(
            fc.property(fc.double({ min: -5, max: 5, noNaN: true }), (v) => {
                expect(formatAnomaly(v)).toMatch(/^(−|\+)?\d+\.\d{2} °C$/);
            }),
        );
    });
});

// Parse NASA GISS GISTEMP v4 "Land-Ocean: Global Means" CSV into annual anomalies.
// Pure: no DOM, no I/O, so the same code runs in the build script, the browser and the tests.

const HEADER_PREFIX = 'Year,Jan';
const MISSING = '***';

/**
 * @param {string} text the raw CSV (GLB.Ts+dSST.csv)
 * @returns {{ title: string, years: { year: number, value: number }[] }}
 *   Only years whose annual mean (J-D) is present. Rows with *** (missing) are dropped, not zeroed.
 */
export function parseGistemp(text) {
    const lines = text.split(/\r?\n/).filter((line) => line.trim() !== '');
    const headerAt = lines.findIndex((line) => line.startsWith(HEADER_PREFIX));
    if (headerAt < 1) throw new Error('GISTEMP CSV: header row "Year,Jan,..." not found');
    const columns = lines[headerAt].split(',');
    const annual = columns.indexOf('J-D');
    if (annual < 0) throw new Error('GISTEMP CSV: no J-D column');

    const years = [];
    for (const line of lines.slice(headerAt + 1)) {
        const cells = line.split(',');
        const year = Number(cells[0]);
        if (!Number.isInteger(year)) continue;
        const raw = cells[annual]?.trim();
        if (raw === undefined || raw === MISSING || raw === '') continue;
        const value = Number(raw);
        if (!Number.isFinite(value)) throw new Error(`GISTEMP CSV: bad value "${raw}" for ${year}`);
        years.push({ year, value });
    }
    return { title: lines[0].trim(), years };
}

/** Mean of the annual values for from..to inclusive, or null when no year falls in the range. */
export function rangeMean(years, from, to) {
    const inRange = years.filter((y) => y.year >= from && y.year <= to);
    if (inRange.length === 0) return null;
    return inRange.reduce((sum, y) => sum + y.value, 0) / inRange.length;
}

/** "+1.28 °C" / "−0.17 °C" with a true minus sign, one decimal place by default. */
export function formatAnomaly(value, digits = 2) {
    const rounded = value.toFixed(digits);
    const sign = Number(rounded) > 0 ? '+' : Number(rounded) < 0 ? '−' : '';
    return `${sign}${Math.abs(Number(rounded)).toFixed(digits)} °C`;
}

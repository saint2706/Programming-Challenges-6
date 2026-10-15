import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import data from '../src/data/gistemp-annual.json';
import { stripeColor, stripesMarkup } from '../src/stripes.js';

describe('stripeColor', () => {
    it('neutral is the pale middle of the diverging scale; the ends are the darkest blue and red', () => {
        expect(stripeColor(0)).toBe('#f7f7f7');
        expect(stripeColor(-1.3)).toBe('#053061');
        expect(stripeColor(1.3)).toBe('#67001f');
    });

    it('property: always a #rrggbb colour, and out-of-range values clamp to the end colours', () => {
        fc.assert(
            fc.property(fc.double({ min: -50, max: 50, noNaN: true }), (v) => {
                expect(stripeColor(v)).toMatch(/^#[0-9a-f]{6}$/);
            }),
        );
        expect(stripeColor(40)).toBe(stripeColor(1.3));
        expect(stripeColor(-40)).toBe(stripeColor(-1.3));
    });
});

describe('stripesMarkup', () => {
    it('writes one stripe per year and sets --stripes for the horizontal pin', () => {
        const html = stripesMarkup(data.years);
        expect(html.match(/class="stripe"/g)).toHaveLength(146);
        expect(html).toContain('--stripes:146');
        expect(html).toContain('data-year="2025"');
    });
});

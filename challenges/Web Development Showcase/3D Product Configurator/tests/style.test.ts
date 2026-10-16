import { Color, MeshPhysicalMaterial } from 'three';
import { describe, expect, test } from 'vitest';
import { FINISHES } from '../src/parts.ts';
import { DETAIL_GAIN, styleMaterial } from '../src/shoe/style.ts';

describe('styling a part', () => {
    test('the colour is the colour picked times the gain that undoes the half-brightness detail map, in the working space', () => {
        const m = new MeshPhysicalMaterial();
        styleMaterial(m, '#336699', 'matte');
        const wanted = new Color('#336699').multiplyScalar(DETAIL_GAIN);
        expect(m.color.r).toBeCloseTo(wanted.r, 6);
        expect(m.color.g).toBeCloseTo(wanted.g, 6);
        expect(m.color.b).toBeCloseTo(wanted.b, 6);
    });

    test('white is not clamped away: the gain pushes it above 1 in linear light, which the half-brightness map brings back to white', () => {
        const m = new MeshPhysicalMaterial();
        styleMaterial(m, '#ffffff', 'matte');
        expect(m.color.r).toBeCloseTo(DETAIL_GAIN, 6);
    });

    test.each(Object.keys(FINISHES) as (keyof typeof FINISHES)[])('the %s finish sets roughness and clearcoat', (finish) => {
        const m = new MeshPhysicalMaterial();
        styleMaterial(m, '#808080', finish);
        expect(m.roughness).toBe(FINISHES[finish].roughness);
        expect(m.clearcoat).toBe(FINISHES[finish].clearcoat);
        expect(m.clearcoatRoughness).toBe(FINISHES[finish].clearcoatRoughness);
    });

    test('styling twice is the same as styling once (the gain is not applied on top of the last colour)', () => {
        const a = new MeshPhysicalMaterial();
        const b = new MeshPhysicalMaterial();
        styleMaterial(a, '#445566', 'satin');
        styleMaterial(b, '#445566', 'satin');
        styleMaterial(b, '#445566', 'satin');
        expect(b.color.equals(a.color)).toBe(true);
    });
});

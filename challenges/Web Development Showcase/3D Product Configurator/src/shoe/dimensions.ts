/** The shoe as prepare-shoe.ts leaves it: metres, toe towards +x, standing on y = 0, centred on x = z = 0. */
export const SHOE = { length: 0.298, width: 0.116, height: 0.153 } as const;

/** The size of the square-ish picture of the shoe's shadow on the floor, which the preparation script also draws. */
export const SHADOW = { width: 0.44, depth: 0.26, pixels: 256 } as const;

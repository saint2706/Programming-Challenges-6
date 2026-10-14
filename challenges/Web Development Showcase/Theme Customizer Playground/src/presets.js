import { defaults } from './schema.js';

const make = (name, patch) => ({ name, theme: { ...defaults(), ...patch } });

export const PRESETS = [
  make('Indigo', {}),
  make('Forest', { brandL: 0.5, brandLDark: 0.78, brandC: 0.12, brandH: 152, accentShift: 60, neutralH: 140, neutralC: 0.01, radius: 6, headingFont: 'serif' }),
  make('Sunset', { brandL: 0.54, brandLDark: 0.72, brandC: 0.16, brandH: 40, accentShift: -40, neutralH: 60, neutralC: 0.02, radius: 16, shadowAlpha: 0.18, headingWeight: 800, headingFont: 'rounded', bodyFont: 'rounded' }),
  make('Mono', { brandL: 0.4, brandLDark: 0.82, brandC: 0, brandH: 0, accentShift: 0, neutralH: 0, neutralC: 0, radius: 0, space: 5, shadowY: 0, shadowBlur: 0, shadowAlpha: 0, typeRatio: 1.2, bodyFont: 'mono', headingFont: 'mono' }),
];

/** The seven parts of the shoe a person can colour, the finishes, and the ready-made designs. */
export const PART_IDS = ['upper', 'overlays', 'laces', 'accents', 'lining', 'midsole', 'outsole'] as const;
export type PartId = (typeof PART_IDS)[number];

export const PART_LABELS: Record<PartId, string> = {
    upper: 'Upper',
    overlays: 'Overlays & heel',
    laces: 'Laces',
    accents: 'Eyelets & pull tabs',
    lining: 'Tongue & lining',
    midsole: 'Midsole',
    outsole: 'Outsole',
};

export const FINISH_IDS = ['matte', 'satin', 'gloss', 'patent'] as const;
export type FinishId = (typeof FINISH_IDS)[number];

export interface Finish {
    label: string;
    /** Multiplies the roughness the model's own map says each spot has (1 keeps the artist's values). */
    roughness: number;
    clearcoat: number;
    clearcoatRoughness: number;
}

export const FINISHES: Record<FinishId, Finish> = {
    matte: { label: 'Matte', roughness: 1, clearcoat: 0, clearcoatRoughness: 0.1 },
    satin: { label: 'Satin', roughness: 0.72, clearcoat: 0, clearcoatRoughness: 0.1 },
    gloss: { label: 'Gloss', roughness: 0.4, clearcoat: 0.25, clearcoatRoughness: 0.2 },
    patent: { label: 'Patent', roughness: 0.55, clearcoat: 1, clearcoatRoughness: 0.05 },
};

export interface Design {
    colors: Record<PartId, string>;
    finishes: Record<PartId, FinishId>;
}

const FINISH_DEFAULTS: Record<PartId, FinishId> = { upper: 'matte', overlays: 'satin', laces: 'matte', accents: 'satin', lining: 'matte', midsole: 'satin', outsole: 'matte' };

/**
 * The first three are the model's own three colourways (their colours are the averages the
 * preparation script measured on the artist's textures); the others are new.
 */
export const PRESETS: { id: string; label: string; colors: Record<PartId, string> }[] = [
    { id: 'midnight', label: 'Midnight', colors: { upper: '#136690', overlays: '#072639', laces: '#2e2e2e', accents: '#3682a8', lining: '#2a2927', midsole: '#dfdfdf', outsole: '#e5e5e5' } },
    { id: 'beach', label: 'Beach', colors: { upper: '#ba9696', overlays: '#6d4a4c', laces: '#2e2e2e', accents: '#c7a7a7', lining: '#2a2927', midsole: '#dfdfdf', outsole: '#e5e5e5' } },
    { id: 'street', label: 'Street', colors: { upper: '#2a2a2a', overlays: '#bc3641', laces: '#2e2e2e', accents: '#b7b7b7', lining: '#2a2927', midsole: '#dfdfdf', outsole: '#e5e5e5' } },
    { id: 'volt', label: 'Volt', colors: { upper: '#1f1f23', overlays: '#c6ff1a', laces: '#c6ff1a', accents: '#c6ff1a', lining: '#2a2927', midsole: '#f3f3f1', outsole: '#1f1f23' } },
    { id: 'ember', label: 'Ember', colors: { upper: '#e8502f', overlays: '#1b1b1d', laces: '#f2e8dc', accents: '#1b1b1d', lining: '#2a2927', midsole: '#f2e8dc', outsole: '#1b1b1d' } },
];

export const PALETTE: readonly { name: string; hex: string }[] = [
    { name: 'Bone', hex: '#f5f2ea' },
    { name: 'Sand', hex: '#d9d4c7' },
    { name: 'Ember', hex: '#e8502f' },
    { name: 'Amber', hex: '#f2b632' },
    { name: 'Volt', hex: '#c6ff1a' },
    { name: 'Pine', hex: '#2f9e6e' },
    { name: 'Ocean', hex: '#136690' },
    { name: 'Indigo', hex: '#3d4cc4' },
    { name: 'Violet', hex: '#7a4ea3' },
    { name: 'Rose', hex: '#d96b8b' },
    { name: 'Graphite', hex: '#2a2a2a' },
    { name: 'Ink', hex: '#0f0f10' },
];

export function defaultDesign(): Design {
    return { colors: { ...PRESETS[0]!.colors }, finishes: { ...FINISH_DEFAULTS } };
}

/** `#rgb` or `#rrggbb` in any case, with or without the hash, as `#rrggbb`; anything else is `null`. */
export function normalizeHex(input: unknown): string | null {
    if (typeof input !== 'string') return null;
    const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(input.trim());
    if (!m) return null;
    const h = m[1]!.toLowerCase();
    return `#${h.length === 3 ? [...h].map((c) => c + c).join('') : h}`;
}

/** Which preset (if any) a design's colours are exactly. */
export function presetOf(colors: Record<PartId, string>): string | null {
    return PRESETS.find((p) => PART_IDS.every((id) => p.colors[id] === colors[id]))?.id ?? null;
}

/** A saved design from storage, or `null`. Anything that is not exactly a design is thrown away, part by part. */
export function parseDesign(raw: unknown): Design | null {
    if (typeof raw !== 'object' || raw === null) return null;
    const { colors, finishes } = raw as { colors?: unknown; finishes?: unknown };
    if (typeof colors !== 'object' || colors === null) return null;
    const out = defaultDesign();
    for (const id of PART_IDS) {
        const c = normalizeHex((colors as Record<string, unknown>)[id]);
        if (c) out.colors[id] = c;
        const f = typeof finishes === 'object' && finishes !== null ? (finishes as Record<string, unknown>)[id] : undefined;
        if (typeof f === 'string' && Object.hasOwn(FINISHES, f)) out.finishes[id] = f as FinishId;
    }
    return out;
}

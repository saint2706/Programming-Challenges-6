import { z } from 'zod';
import { isSafeHref } from './links.js';

/**
 * The resume document: a JSON Resume (v1) subset plus an `x-layout` key for the builder's own
 * settings. Every field has a default, so a partial or hand-written file still loads. Unknown keys
 * are stripped (zod's default), which also makes `__proto__` / `constructor` keys in an imported
 * file inert. Array and string sizes are capped so a hostile import is rejected, not rendered.
 */

export const SECTION_IDS = [
  'summary',
  'work',
  'education',
  'projects',
  'skills',
  'certificates',
  'languages',
  'awards',
];
export const TEMPLATE_IDS = ['classic', 'sidebar', 'compact'];
export const PAGE_SIZES = ['a4', 'letter'];
export const FONT_PAIR_IDS = ['inter', 'serif', 'plex'];
export const SCALE_MIN = 0.82;
export const SCALE_MAX = 1.15;
export const LIMITS = { entries: 100, bullets: 50, keywords: 50, short: 500, long: 5000 };

const PARTIAL_DATE = /^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$/;

const short = (max = LIMITS.short) => z.string().max(max).default('');
const long = z.string().max(LIMITS.long).default('');
const date = z
  .union([z.literal(''), z.string().regex(PARTIAL_DATE, 'Use YYYY, YYYY-MM or YYYY-MM-DD')])
  .default('');
const link = z
  .union([
    z.literal(''),
    z.string().max(2048).refine(isSafeHref, 'Use an http(s), mailto or tel link'),
  ])
  .default('');
const bullets = z.array(z.string().max(LIMITS.long)).max(LIMITS.bullets).default([]);

const entries = (shape) => z.array(z.object(shape)).max(LIMITS.entries).default([]);

export const LAYOUT_DEFAULTS = Object.freeze({
  template: 'classic',
  pageSize: 'a4',
  accent: '#1d4ed8',
  fontPair: 'inter',
  scale: 1,
  sectionOrder: SECTION_IDS,
  hidden: [],
});

/** A layout with unknown/duplicate section ids dropped and missing ones appended. */
export function normalizeLayout(layout) {
  const base = { ...LAYOUT_DEFAULTS, ...(layout ?? {}) };
  const order = [];
  for (const id of [...(base.sectionOrder ?? []), ...SECTION_IDS]) {
    if (SECTION_IDS.includes(id) && !order.includes(id)) order.push(id);
  }
  const hidden = [];
  for (const id of base.hidden ?? []) {
    if (SECTION_IDS.includes(id) && !hidden.includes(id)) hidden.push(id);
  }
  return { ...base, sectionOrder: order, hidden };
}

const layoutSchema = z
  .object({
    template: z.enum(TEMPLATE_IDS).default(LAYOUT_DEFAULTS.template),
    pageSize: z.enum(PAGE_SIZES).default(LAYOUT_DEFAULTS.pageSize),
    accent: z
      .string()
      .regex(/^#[0-9a-fA-F]{6}$/, 'Use a #rrggbb colour')
      .default(LAYOUT_DEFAULTS.accent),
    fontPair: z.enum(FONT_PAIR_IDS).default(LAYOUT_DEFAULTS.fontPair),
    scale: z.number().min(SCALE_MIN).max(SCALE_MAX).default(LAYOUT_DEFAULTS.scale),
    sectionOrder: z.array(z.string().max(40)).max(40).default([...SECTION_IDS]),
    hidden: z.array(z.string().max(40)).max(40).default([]),
  })
  .transform(normalizeLayout);

const basics = z.object({
  name: short(),
  label: short(),
  email: z
    .union([
      z.literal(''),
      z
        .string()
        .max(254)
        .refine(
          (value) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value),
          'Use a valid email address'
        ),
    ])
    .default(''),
  phone: z
    .string()
    .max(20)
    .regex(/^[0-9\s+\-().]*$/, 'Use digits, spaces, +, -, (), or .')
    .default(''),
  url: link,
  summary: long,
  location: z
    .object({ city: short(100), region: short(100), countryCode: short(10) })
    .prefault({}),
  profiles: z
    .array(z.object({ network: short(100), username: short(100), url: link }))
    .max(LIMITS.entries)
    .default([]),
});

const resumeSchema = z.object({
  basics: basics.prefault({}),
  work: entries({
    name: short(),
    position: short(),
    url: link,
    startDate: date,
    endDate: date,
    summary: long,
    highlights: bullets,
  }),
  education: entries({
    institution: short(),
    url: link,
    area: short(),
    studyType: short(),
    startDate: date,
    endDate: date,
    score: short(100),
  }),
  projects: entries({
    name: short(),
    url: link,
    startDate: date,
    endDate: date,
    description: long,
    highlights: bullets,
  }),
  skills: entries({
    name: short(),
    level: short(100),
    keywords: z.array(z.string().max(100)).max(LIMITS.keywords).default([]),
  }),
  certificates: entries({ name: short(), date, issuer: short(), url: link }),
  languages: entries({ language: short(100), fluency: short(100) }),
  awards: entries({ title: short(), date, awarder: short(), summary: long }),
  'x-layout': layoutSchema.prefault({}),
});

/**
 * Validate and normalise untrusted input.
 * @returns {{ok: true, data: object} | {ok: false, errors: {path: string, message: string}[]}}
 */
export function parseResume(input) {
  if (input === null || typeof input !== 'object' || Array.isArray(input)) {
    return { ok: false, errors: [{ path: '', message: 'Expected a JSON object' }] };
  }
  const result = resumeSchema.safeParse(input);
  if (result.success) return { ok: true, data: result.data };
  return {
    ok: false,
    errors: result.error.issues.map((issue) => ({
      path: issue.path.join('.'),
      message: issue.message,
    })),
  };
}

/** Validate a (possibly partial) layout. Returns the normalised layout, or null if invalid. */
export function parseLayout(input) {
  const result = layoutSchema.safeParse(input);
  return result.success ? result.data : null;
}

/** A blank, valid resume with the default layout. */
export function emptyResume() {
  return resumeSchema.parse({});
}

/** Human-readable multi-line summary of `parseResume` errors (first 8). */
export function formatIssues(errors, max = 8) {
  const lines = errors.slice(0, max).map((e) => `${e.path || '(file)'}: ${e.message}`);
  if (errors.length > max) lines.push(`…and ${errors.length - max} more`);
  return lines.join('\n');
}

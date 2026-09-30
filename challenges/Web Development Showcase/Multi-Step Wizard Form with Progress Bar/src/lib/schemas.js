import { z } from 'zod';

export const MIN_AGE = 13;
export const PLANS = ['free', 'pro', 'team'];
export const FREQUENCIES = ['daily', 'weekly', 'monthly'];
export const INTERESTS = ['frontend', 'backend', 'data', 'ml', 'design'];

/** Luhn checksum over a digit string. */
export function luhnValid(digits) {
  if (!/^\d+$/.test(digits)) return false;
  let sum = 0;
  let double = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let d = Number(digits[i]);
    if (double) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    sum += d;
    double = !double;
  }
  return sum % 10 === 0;
}

/** Whole years between an ISO date (YYYY-MM-DD) and `today`; NaN if the date is not real. */
export function ageOn(iso, today = new Date()) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso ?? '');
  if (!m) return NaN;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const probe = new Date(Date.UTC(y, mo - 1, d));
  if (probe.getUTCFullYear() !== y || probe.getUTCMonth() !== mo - 1 || probe.getUTCDate() !== d) return NaN;
  let age = today.getUTCFullYear() - y;
  const beforeBirthday =
    today.getUTCMonth() < mo - 1 || (today.getUTCMonth() === mo - 1 && today.getUTCDate() < d);
  if (beforeBirthday) age--;
  return age;
}

/** A card is valid through the last day of its expiry month. */
export function expiryValid(value, today = new Date()) {
  const m = /^(0[1-9]|1[0-2])\/(\d{2})$/.exec(value ?? '');
  if (!m) return false;
  const year = 2000 + Number(m[2]);
  const month = Number(m[1]);
  const y = today.getUTCFullYear();
  const mo = today.getUTCMonth() + 1;
  return year > y || (year === y && month >= mo);
}

const required = (label) => z.string().trim().min(1, `${label} is required`);

export const accountSchema = z
  .object({
    email: required('Email').email('Enter a valid email address'),
    username: required('Username')
      .min(3, 'Username must be at least 3 characters')
      .max(20, 'Username must be at most 20 characters')
      .regex(/^[a-z0-9_]+$/i, 'Use letters, numbers and underscores only'),
    password: z
      .string()
      .min(1, 'Password is required')
      .min(10, 'Password must be at least 10 characters')
      .regex(/[a-z]/i, 'Include at least one letter')
      .regex(/\d/, 'Include at least one number'),
    confirmPassword: z.string().min(1, 'Confirm your password'),
  })
  .superRefine((v, ctx) => {
    if (v.confirmPassword && v.password !== v.confirmPassword) {
      ctx.addIssue({ code: 'custom', path: ['confirmPassword'], message: 'Passwords do not match' });
    }
  });

export const profileSchema = z.object({
  fullName: required('Full name').max(80, 'Full name is too long'),
  birthDate: z
    .string()
    .min(1, 'Date of birth is required')
    .superRefine((v, ctx) => {
      const age = ageOn(v);
      if (Number.isNaN(age)) ctx.addIssue({ code: 'custom', message: 'Enter a real date' });
      else if (age < 0) ctx.addIssue({ code: 'custom', message: 'Date of birth cannot be in the future' });
      else if (age < MIN_AGE) ctx.addIssue({ code: 'custom', message: `You must be at least ${MIN_AGE}` });
    }),
  country: required('Country'),
  phone: z
    .string()
    .trim()
    .refine((v) => v === '' || /^\+?[0-9 ()-]{7,20}$/.test(v), 'Enter a valid phone number, or leave blank'),
});

export const preferencesSchema = z
  .object({
    interests: z.array(z.enum(INTERESTS)).min(1, 'Pick at least one interest'),
    newsletter: z.boolean(),
    frequency: z.enum(['', ...FREQUENCIES]),
  })
  .superRefine((v, ctx) => {
    if (v.newsletter && v.frequency === '') {
      ctx.addIssue({ code: 'custom', path: ['frequency'], message: 'Choose how often you want emails' });
    }
  });

export const planSchema = z
  .object({
    plan: z.enum(PLANS, { error: 'Choose a plan' }),
    company: z.string().trim(),
    seats: z.string().trim(),
    cardNumber: z.string(),
    expiry: z.string().trim(),
    cvc: z.string().trim(),
  })
  .superRefine((v, ctx) => {
    const add = (path, message) => ctx.addIssue({ code: 'custom', path: [path], message });
    if (v.plan === 'free') return;
    if (v.plan === 'team') {
      if (!v.company) add('company', 'Company name is required for Team');
      const seats = Number(v.seats);
      if (v.seats === '') add('seats', 'Number of seats is required');
      else if (!Number.isInteger(seats) || seats < 2 || seats > 50) add('seats', 'Seats must be a whole number from 2 to 50');
    }
    const digits = v.cardNumber.replace(/[ -]/g, '');
    if (!digits) add('cardNumber', 'Card number is required');
    else if (digits.length < 13 || digits.length > 19 || !luhnValid(digits)) add('cardNumber', 'Enter a valid card number');
    if (!v.expiry) add('expiry', 'Expiry is required');
    else if (!expiryValid(v.expiry)) add('expiry', 'Use MM/YY, and the card must not be expired');
    if (!v.cvc) add('cvc', 'CVC is required');
    else if (!/^\d{3,4}$/.test(v.cvc)) add('cvc', 'CVC is 3 or 4 digits');
  });

export const reviewSchema = z.object({
  acceptTerms: z.boolean().refine((v) => v === true, 'You must accept the terms to continue'),
});

/**
 * Ordered wizard definition. `fields` lists every form-control name a step owns, so the
 * UI can touch/focus them and a step-scoped validation can pick its own errors out.
 */
export const STEPS = [
  { id: 'account', title: 'Account', schema: accountSchema, fields: ['email', 'username', 'password', 'confirmPassword'] },
  { id: 'profile', title: 'Profile', schema: profileSchema, fields: ['fullName', 'birthDate', 'country', 'phone'] },
  { id: 'preferences', title: 'Preferences', schema: preferencesSchema, fields: ['interests', 'newsletter', 'frequency'] },
  { id: 'plan', title: 'Plan', schema: planSchema, fields: ['plan', 'company', 'seats', 'cardNumber', 'expiry', 'cvc'] },
  { id: 'review', title: 'Review', schema: reviewSchema, fields: ['acceptTerms'] },
];

export function defaultData() {
  return {
    email: '',
    username: '',
    password: '',
    confirmPassword: '',
    fullName: '',
    birthDate: '',
    country: '',
    phone: '',
    interests: [],
    newsletter: false,
    frequency: '',
    plan: 'free',
    company: '',
    seats: '',
    cardNumber: '',
    expiry: '',
    cvc: '',
    acceptTerms: false,
  };
}

/** Fields that must never be written to localStorage. */
export const SENSITIVE_FIELDS = ['password', 'confirmPassword', 'cardNumber', 'expiry', 'cvc'];

/** Validate one step. Returns { ok, errors } with the first message per field. */
export function validateStep(index, data) {
  const step = STEPS[index];
  const result = step.schema.safeParse(data);
  if (result.success) return { ok: true, errors: {} };
  const errors = {};
  for (const issue of result.error.issues) {
    const key = String(issue.path[0] ?? step.fields[0]);
    if (!(key in errors)) errors[key] = issue.message;
  }
  return { ok: false, errors };
}

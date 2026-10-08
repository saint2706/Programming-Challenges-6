import { describe, expect, it } from 'vitest';
import { formatDate, formatRange } from '../src/lib/dates.js';

describe('formatDate', () => {
  it('formats partial ISO dates', () => {
    expect(formatDate('2020')).toBe('2020');
    expect(formatDate('2023-05')).toBe('May 2023');
    expect(formatDate('2023-12-09')).toBe('Dec 9, 2023');
  });

  it('returns an empty string for empty or invalid input', () => {
    expect(formatDate('')).toBe('');
    expect(formatDate(undefined)).toBe('');
    expect(formatDate('soon')).toBe('');
    expect(formatDate('2023-13')).toBe('');
  });
});

describe('formatRange', () => {
  it('joins start and end with an en dash', () => {
    expect(formatRange('2023-05', '2023-07')).toBe('May 2023 – Jul 2023');
    expect(formatRange('2020', '2024')).toBe('2020 – 2024');
  });

  it('treats a missing end as Present', () => {
    expect(formatRange('2025-06', '')).toBe('Jun 2025 – Present');
  });

  it('collapses an identical start and end, and handles one-sided ranges', () => {
    expect(formatRange('2023-01', '2023-01')).toBe('Jan 2023');
    expect(formatRange('', '2022')).toBe('2022');
    expect(formatRange('', '')).toBe('');
  });
});

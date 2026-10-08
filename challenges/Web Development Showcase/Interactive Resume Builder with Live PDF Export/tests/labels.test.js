import { describe, expect, it } from 'vitest';
import { fieldLabel } from '../src/lib/labels.js';

describe('fieldLabel', () => {
  it('names the header fields', () => {
    expect(fieldLabel('basics.name')).toBe('Name');
    expect(fieldLabel('basics.label')).toBe('Headline');
    expect(fieldLabel('basics.summary')).toBe('Summary');
  });

  it('names entry fields with their section and entry number', () => {
    expect(fieldLabel('work.1.position')).toBe('Position, Experience entry 2');
    expect(fieldLabel('education.0.institution')).toBe('Institution, Education entry 1');
    expect(fieldLabel('skills.2.keywords')).toBe('Skills (comma separated), Skills entry 3');
  });

  it('numbers bullets', () => {
    expect(fieldLabel('work.0.highlights.2')).toBe('Bullet 3, Experience entry 1');
    expect(fieldLabel('projects.4.highlights.0')).toBe('Bullet 1, Projects entry 5');
  });

  it('falls back to the raw path for anything unknown', () => {
    expect(fieldLabel('nope.1.x')).toBe('nope.1.x');
    expect(fieldLabel('work.0.nope')).toBe('work.0.nope');
  });
});

import { describe, expect, it } from 'vitest';
import { isSafeHref, safeHref } from '../src/lib/links.js';

describe('isSafeHref', () => {
  it.each([
    'https://github.com/saint2706',
    'http://example.com/a?b=c#d',
    'mailto:someone@example.com',
    'tel:+919999999999',
    '  https://example.com/padded  ',
  ])('allows %j', (url) => {
    expect(isSafeHref(url)).toBe(true);
  });

  it.each([
    'javascript:alert(1)',
    'JaVaScRiPt:alert(1)',
    '  javascript:alert(1)',
    'java\tscript:alert(1)',
    'java\nscript:alert(1)',
    'java\u0000script:alert(1)',
    '\u0001javascript:alert(1)',
    'vbscript:msgbox(1)',
    'data:text/html,<script>alert(1)</script>',
    'file:///c:/windows/win.ini',
    '//evil.example.com',
    '/relative/path',
    'ftp://example.com',
    'https://exa mple.com',
    '',
    '   ',
    null,
    undefined,
    42,
    {},
  ])('rejects %j', (url) => {
    expect(isSafeHref(url)).toBe(false);
  });

  it('rejects absurdly long urls', () => {
    expect(isSafeHref(`https://example.com/${'a'.repeat(3000)}`)).toBe(false);
  });
});

describe('safeHref', () => {
  it('returns the trimmed url when safe and null otherwise', () => {
    expect(safeHref('  https://example.com/x  ')).toBe('https://example.com/x');
    expect(safeHref('javascript:alert(1)')).toBeNull();
    expect(safeHref(undefined)).toBeNull();
  });
});

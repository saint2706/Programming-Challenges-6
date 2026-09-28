import { describe, it, expect } from 'vitest';
import {
  escapeHtml,
  isSafeHref,
  safeHref,
  renderBlock,
  serializeBlocks,
  serializeEmail,
} from '../src/lib/serialize.js';

describe('escapeHtml', () => {
  it('escapes all five HTML-significant characters', () => {
    expect(escapeHtml(`&<>"'`)).toBe('&amp;&lt;&gt;&quot;&#39;');
  });

  it('handles null/undefined as empty string', () => {
    expect(escapeHtml(null)).toBe('');
    expect(escapeHtml(undefined)).toBe('');
  });

  it('coerces non-string values', () => {
    expect(escapeHtml(42)).toBe('42');
  });
});

describe('isSafeHref / safeHref', () => {
  it('accepts ordinary http(s)/mailto links', () => {
    expect(isSafeHref('https://example.com')).toBe(true);
    expect(isSafeHref('http://example.com/path?x=1')).toBe(true);
    expect(isSafeHref('mailto:person@example.com')).toBe(true);
  });

  it('rejects javascript: URLs, including whitespace/case-obfuscated ones', () => {
    expect(isSafeHref('javascript:alert(1)')).toBe(false);
    expect(isSafeHref('JavaScript:alert(1)')).toBe(false);
    expect(isSafeHref('java\tscript:alert(1)')).toBe(false);
    expect(isSafeHref('  javascript:alert(1)')).toBe(false);
  });

  it('rejects vbscript: and data: URLs', () => {
    expect(isSafeHref('vbscript:msgbox(1)')).toBe(false);
    expect(isSafeHref('data:text/html,<script>alert(1)</script>')).toBe(false);
  });

  it('rejects empty/non-string hrefs', () => {
    expect(isSafeHref('')).toBe(false);
    expect(isSafeHref('   ')).toBe(false);
    expect(isSafeHref(undefined)).toBe(false);
  });

  it('safeHref falls back to "#" for unsafe input and escapes safe input', () => {
    expect(safeHref('javascript:alert(1)')).toBe('#');
    expect(safeHref('https://example.com/?a=1&b=2')).toBe('https://example.com/?a=1&amp;b=2');
  });
});

describe('renderBlock', () => {
  it('throws on an unknown block type', () => {
    expect(() => renderBlock({ type: 'nope', props: {} })).toThrow(/Unknown block type/);
  });

  it('renders a heading inside a table row with escaped text', () => {
    const out = renderBlock({ type: 'heading', props: { text: '<b>Hi</b>', align: 'center' } });
    expect(out).toContain('<tr>');
    expect(out).toContain('<h1');
    expect(out).toContain('&lt;b&gt;Hi&lt;/b&gt;');
    expect(out).not.toContain('<b>Hi</b>');
  });

  it('renders the bulletproof button as a table-wrapped anchor', () => {
    const out = renderBlock({
      type: 'button',
      props: { label: 'Go', href: 'https://example.com', bgColor: '#000', textColor: '#fff' },
    });
    expect(out).toMatch(/<table[^>]*role="presentation"/);
    expect(out).toContain('<a href="https://example.com"');
    expect(out).toContain('>Go</a>');
  });

  it('neutralizes a javascript: href on a button', () => {
    const out = renderBlock({
      type: 'button',
      props: { label: 'Go', href: 'javascript:alert(document.cookie)' },
    });
    expect(out).toContain('href="#"');
    expect(out).not.toContain('javascript:');
  });

  it('renders an image with escaped src/alt and explicit width', () => {
    const out = renderBlock({
      type: 'image',
      props: { src: 'https://example.com/a.png', alt: '"quoted" alt', width: 480 },
    });
    expect(out).toContain('src="https://example.com/a.png"');
    expect(out).toContain('width="480"');
    expect(out).toContain('&quot;quoted&quot; alt');
  });

  it('renders two columns as a nested 50/50 table', () => {
    const out = renderBlock({
      type: 'columns',
      props: { leftText: 'left', rightText: 'right' },
    });
    expect(out).toContain('width="50%"');
    expect((out.match(/width="50%"/g) || []).length).toBe(2);
    expect(out).toContain('left');
    expect(out).toContain('right');
  });

  it('renders divider and spacer as zero-content structural rows', () => {
    const divider = renderBlock({ type: 'divider', props: { color: '#000', height: 2 } });
    expect(divider).toContain('border-top:2px solid #000');

    const spacer = renderBlock({ type: 'spacer', props: { height: 40 } });
    expect(spacer).toContain('height:40px');
  });
});

describe('serializeBlocks', () => {
  it('concatenates multiple blocks in order', () => {
    const out = serializeBlocks([
      { type: 'heading', props: { text: 'First' } },
      { type: 'paragraph', props: { text: 'Second' } },
    ]);
    expect(out.indexOf('First')).toBeLessThan(out.indexOf('Second'));
  });

  it('returns an empty string for no blocks', () => {
    expect(serializeBlocks([])).toBe('');
  });
});

describe('serializeEmail', () => {
  it('wraps blocks in a full HTML document with the requested table width', () => {
    const html = serializeEmail([{ type: 'heading', props: { text: 'Hello' } }], { width: 480 });
    expect(html).toMatch(/^<!doctype html>/);
    expect(html).toContain('width="480"');
    expect(html).toContain('Hello');
    expect(html).not.toContain('<style>');
  });

  it('defaults to 600px width when not specified', () => {
    const html = serializeEmail([]);
    expect(html).toContain('width="600"');
  });

  it('escapes the preview text and never leaves raw markup unescaped anywhere', () => {
    const html = serializeEmail([{ type: 'paragraph', props: { text: 'ok' } }], {
      previewText: '<img src=x onerror=alert(1)>',
    });
    expect(html).not.toContain('<img src=x onerror=alert(1)>');
    expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');
  });

  it('prevents stored XSS via block text content end-to-end', () => {
    const html = serializeEmail([
      { type: 'heading', props: { text: '<script>alert(document.cookie)</script>' } },
    ]);
    expect(html).not.toContain('<script>alert(document.cookie)</script>');
    expect(html).toContain('&lt;script&gt;');
  });
});

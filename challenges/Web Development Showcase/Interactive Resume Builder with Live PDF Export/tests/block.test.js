import { render, screen } from '@testing-library/svelte';
import { describe, expect, it, vi } from 'vitest';
import Block from '../src/components/Block.svelte';
import { blockIndex, buildRegions } from '../src/lib/blocks.js';
import { setPath } from '../src/lib/model.js';
import { longResume } from '../src/lib/sample-long.js';
import { seedResume } from '../src/lib/seed.js';

const allBlocks = (doc) => [...blockIndex(buildRegions(doc)).values()];
const withTemplate = (doc, template) => ({ ...doc, 'x-layout': { ...doc['x-layout'], template } });
const classicSeed = () => withTemplate(seedResume(), 'classic');

/** The element structure of a render: tag + classes for every node, in order. */
const shape = (container) =>
  [...container.querySelectorAll('*')].map((el) => `${el.tagName.toLowerCase()}.${[...el.classList].join('.')}`);

function renderBlock(block, props = {}) {
  return render(Block, { props: { block, editable: false, onedit: vi.fn(), ...props } });
}

describe('Block', () => {
  it('renders the same elements and text whether or not it is editable', () => {
    const docs = [
      classicSeed(),
      withTemplate(classicSeed(), 'sidebar'),
      withTemplate(longResume(), 'compact'),
      withTemplate(longResume(), 'sidebar'),
    ];
    let checked = 0;
    for (const doc of docs) {
      for (const block of allBlocks(doc)) {
        const plain = renderBlock(block, { editable: false });
        const live = renderBlock(block, { editable: true });
        expect(shape(live.container), block.id).toEqual(shape(plain.container));
        expect(live.container.textContent, block.id).toBe(plain.container.textContent);
        plain.unmount();
        live.unmount();
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(100);
  });

  it('marks its root with the block id for measuring', () => {
    const block = blockIndex(buildRegions(longResume())).get('work.1.header');
    const { container } = renderBlock(block);
    expect(container.firstElementChild).toHaveAttribute('data-block', 'work.1.header');
  });

  it('renders the header as h1 with editable name and headline', () => {
    const block = blockIndex(buildRegions(classicSeed())).get('header');
    renderBlock(block, { editable: true });
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Rishabh Agrawal');
    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveAttribute('data-path', 'basics.name');
    expect(screen.getByRole('textbox', { name: 'Headline' })).toBeInTheDocument();
  });

  it('shows a placeholder for an empty name', () => {
    const doc = setPath(classicSeed(), 'basics.name', '');
    renderBlock(blockIndex(buildRegions(doc)).get('header'), { editable: true });
    expect(screen.getByRole('textbox', { name: 'Name' })).toHaveAttribute('data-placeholder', 'Your name');
  });

  it('renders headings as h2', () => {
    const block = blockIndex(buildRegions(classicSeed())).get('heading.work');
    renderBlock(block);
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent('Experience');
  });

  it('links only safe urls, with safe rel attributes', () => {
    const doc = classicSeed();
    renderBlock(blockIndex(buildRegions(doc)).get('header'));
    const github = screen.getByRole('link', { name: 'GitHub' });
    expect(github).toHaveAttribute('href', 'https://github.com/saint2706');
    expect(github).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('renders an unsafe profile link as plain text', () => {
    const doc = setPath(classicSeed(), 'basics.profiles', [
      { network: 'Evil', username: 'x', url: 'javascript:alert(1)' },
    ]);
    const { container } = renderBlock(blockIndex(buildRegions(doc)).get('header'));
    expect(container).toHaveTextContent('Evil');
    expect(screen.queryByRole('link', { name: 'Evil' })).toBeNull();
    expect(container.querySelector('[href^="javascript"]')).toBeNull();
    // the website in the same header is still a link
    expect(screen.getByRole('link', { name: 'saint2706.github.io' })).toBeInTheDocument();
  });

  it('renders only the requested rows of a bullet list', () => {
    const block = blockIndex(buildRegions(longResume())).get('work.0.bullets');
    renderBlock(block, { rowRange: [1, 3] });
    expect(screen.getAllByRole('listitem')).toHaveLength(2);
    expect(screen.getAllByRole('listitem')[0]).toHaveTextContent(block.rows[1].value);
  });

  it('gives every bullet an editable field named after its position', () => {
    const block = blockIndex(buildRegions(longResume())).get('work.0.bullets');
    renderBlock(block, { editable: true });
    expect(screen.getByRole('textbox', { name: 'Bullet 1, Experience entry 1' })).toHaveAttribute(
      'data-path',
      'work.0.highlights.0',
    );
    expect(screen.getAllByRole('textbox')).toHaveLength(5);
  });

  it('shows markup in resume text as text, in both modes', () => {
    const evil = '<img src=x onerror=alert(1)><script>alert(2)</script>';
    const doc = setPath(longResume(), 'work.0.highlights.0', evil);
    const block = blockIndex(buildRegions(doc)).get('work.0.bullets');
    for (const editable of [false, true]) {
      const { container, unmount } = renderBlock(block, { editable });
      expect(container.querySelector('img, script')).toBeNull();
      expect(container.textContent).toContain(evil);
      unmount();
    }
  });

  it('renders emoji, CJK and combining characters unchanged, in both modes', () => {
    const text = '日本語のテキスト 🚀 café e\u0301 Ελληνικά 😀';
    const doc = setPath(longResume(), 'work.0.highlights.0', text);
    const block = blockIndex(buildRegions(doc)).get('work.0.bullets');
    for (const editable of [false, true]) {
      const { container, unmount } = renderBlock(block, { editable });
      expect(container.textContent).toContain(text);
      unmount();
    }
  });

  it('renders skill keywords as a comma separated editable list', () => {
    const block = blockIndex(buildRegions(classicSeed())).get('skills.0');
    renderBlock(block, { editable: true });
    expect(screen.getByRole('textbox', { name: 'Skills (comma separated), Skills entry 1' })).toHaveTextContent(
      'Python, SQL, JavaScript, Java, C++, R, HTML/CSS',
    );
  });
});

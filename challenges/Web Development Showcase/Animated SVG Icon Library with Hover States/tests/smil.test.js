import { describe, expect, it } from 'vitest';
import { countSmilAnimations, stripSmilAnimations } from '../scripts/smil.js';

describe('stripSmilAnimations', () => {
  it('removes a self-closing <animate> tag', () => {
    const markup = '<svg><circle r="2"><animate attributeName="r" values="2;4;2" dur="1s"/></circle></svg>';
    expect(stripSmilAnimations(markup)).toBe('<svg><circle r="2"></circle></svg>');
  });

  it('removes a self-closing <animateTransform> tag', () => {
    const markup = '<line><animateTransform attributeName="transform" type="rotate" values="0;10;0" dur="1s"/></line>';
    expect(stripSmilAnimations(markup)).toBe('<line></line>');
  });

  it('removes a paired <animateMotion> block including its nested <mpath>', () => {
    const markup = '<svg><path id="p"/><circle><animateMotion dur="1s"><mpath href="#p"/></animateMotion></circle></svg>';
    expect(stripSmilAnimations(markup)).toBe('<svg><path id="p"/><circle></circle></svg>');
  });

  it('removes multiple animate elements from the same markup', () => {
    const markup =
      '<svg><circle><animate attributeName="r" values="1;2" dur="1s"/><animate attributeName="opacity" values="1;0" dur="1s"/></circle></svg>';
    expect(stripSmilAnimations(markup)).toBe('<svg><circle></circle></svg>');
  });

  it('leaves markup with no animation elements untouched', () => {
    const markup = '<svg><rect x="0" y="0" width="10" height="10"/></svg>';
    expect(stripSmilAnimations(markup)).toBe(markup);
  });

  it('does not touch unrelated attributes containing the word "animate"', () => {
    const markup = '<svg data-animate-hint="none"><rect/></svg>';
    expect(stripSmilAnimations(markup)).toBe(markup);
  });
});

describe('countSmilAnimations', () => {
  it('returns 0 for markup with no animation elements', () => {
    expect(countSmilAnimations('<svg><rect/></svg>')).toBe(0);
  });

  it('counts animate, animateTransform, and animateMotion tags together', () => {
    const markup =
      '<svg><animate attributeName="r"/><animateTransform attributeName="transform"/><animateMotion></animateMotion></svg>';
    expect(countSmilAnimations(markup)).toBe(3);
  });

  it('is case-insensitive to tag casing', () => {
    expect(countSmilAnimations('<ANIMATE attributeName="r"/>')).toBe(1);
  });
});

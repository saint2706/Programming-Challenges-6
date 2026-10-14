import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ImgCompare } from '../src/index.js';

const RECT = { left: 0, top: 0, width: 200, height: 100, right: 200, bottom: 100, x: 0, y: 0 };

function make(attrs = {}) {
  const el = document.createElement('img-compare');
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  document.body.append(el);
  return el;
}
const shadow = (el) => el.shadowRoot;
const root = (el) => shadow(el).querySelector('.root');
const handle = (el) => shadow(el).querySelector('.handle');
const img = (el, which) => shadow(el).querySelector(`.${which} img`);

/** jsdom has no layout: give the slider a 200x100 box so pointer maths has something to work with. */
function sized(el) {
  root(el).getBoundingClientRect = () => RECT;
  return el;
}

function pointer(el, type, init = {}) {
  const event = new MouseEvent(type, { bubbles: true, composed: true, cancelable: true, button: 0, ...init });
  Object.defineProperty(event, 'pointerId', { value: init.pointerId ?? 1 });
  Object.defineProperty(event, 'pointerType', { value: init.pointerType ?? 'mouse' });
  // Dispatch on the handle's parent layer when asked, so composedPath() sees the real target.
  (init.target ?? root(el)).dispatchEvent(event);
}

function key(el, k, init = {}) {
  const event = new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true, ...init });
  handle(el).dispatchEvent(event);
  return event;
}

function record(el, ...types) {
  const log = [];
  for (const type of types) el.addEventListener(type, (e) => log.push([type, e.detail?.position]));
  return log;
}

beforeEach(() => {
  document.body.replaceChildren();
});

describe('defaults and accessibility', () => {
  it('registers the element and exposes a slider with a name and a value', () => {
    const el = make();
    expect(customElements.get('img-compare')).toBe(ImgCompare);
    const h = handle(el);
    expect(h.getAttribute('role')).toBe('slider');
    expect(h.tabIndex).toBe(0);
    expect(h.getAttribute('aria-label')).toBe('Image comparison');
    expect(h.getAttribute('aria-valuemin')).toBeNull(); // 0..100 are the ARIA defaults
    expect(h.getAttribute('aria-valuenow')).toBe('50');
    expect(h.getAttribute('aria-valuetext')).toBe('50% Before, 50% After');
    expect(h.getAttribute('aria-orientation')).toBe('horizontal');
  });

  it('follows position, labels and orientation into ARIA', () => {
    const el = make({ position: '30', 'before-label': 'Gray', 'after-label': 'Color', label: 'Colour check', orientation: 'vertical' });
    const h = handle(el);
    expect(h.getAttribute('aria-valuenow')).toBe('30');
    expect(h.getAttribute('aria-valuetext')).toBe('30% Gray, 70% Color');
    expect(h.getAttribute('aria-label')).toBe('Colour check');
    expect(h.getAttribute('aria-orientation')).toBe('vertical');
    expect(root(el).style.getPropertyValue('--ic-pos')).toBe('30');
  });

  it('is correct when the element is upgraded from markup that already has attributes', () => {
    document.body.innerHTML = '<img-compare position="25" before-label="Old"></img-compare>';
    const el = document.querySelector('img-compare');
    expect(handle(el).getAttribute('aria-valuetext')).toBe('25% Old, 75% After');
  });
});

describe('attribute and property reflection', () => {
  it('position: property -> attribute, clamped, and bad values keep the old one', () => {
    const el = make();
    el.position = 70;
    expect(el.getAttribute('position')).toBe('70');
    el.position = 500;
    expect(el.position).toBe(100);
    el.position = 'nonsense';
    expect(el.position).toBe(100);
    el.setAttribute('position', '-20');
    expect(el.position).toBe(0);
    el.setAttribute('position', 'junk');
    expect(el.position).toBe(50);
  });

  it('mirrors every other property onto its attribute and back', () => {
    const el = make();
    el.beforeSrc = 'a.webp';
    el.afterSrc = 'b.webp';
    el.beforeAlt = 'A';
    el.afterAlt = 'B';
    el.label = 'L';
    el.beforeLabel = 'x';
    el.afterLabel = 'y';
    el.step = 2;
    el.bigStep = 20;
    el.ratio = '3/2';
    el.orientation = 'vertical';
    el.fit = 'contain';
    el.hover = true;
    expect(Object.fromEntries([...el.attributes].map((a) => [a.name, a.value]))).toMatchObject({
      'before-src': 'a.webp', 'after-src': 'b.webp', 'before-alt': 'A', 'after-alt': 'B', label: 'L',
      'before-label': 'x', 'after-label': 'y', step: '2', 'big-step': '20', ratio: '3/2', orientation: 'vertical', fit: 'contain', hover: '',
    });
    el.removeAttribute('hover');
    expect(el.hover).toBe(false);
    el.setAttribute('orientation', 'diagonal');
    expect(el.orientation).toBe('horizontal');
    el.setAttribute('step', '-3');
    expect(el.step).toBe(1);
  });

  it('keeps a property that was set on the instance before it connected (upgrade case)', () => {
    const el = new ImgCompare();
    Object.defineProperty(el, 'position', { value: 12, configurable: true, writable: true, enumerable: true });
    document.body.append(el);
    expect(el.position).toBe(12);
    expect(el.getAttribute('position')).toBe('12');
  });

  it('puts alt text on the images', () => {
    const el = make({ 'before-alt': 'dark', 'after-alt': 'light' });
    expect(img(el, 'before').alt).toBe('dark');
    expect(img(el, 'after').alt).toBe('light');
  });
});

describe('keyboard', () => {
  it('moves, emits input then change per press, and swallows only its own keys', () => {
    const el = make({ position: '50' });
    const log = record(el, 'input', 'change');
    const arrow = key(el, 'ArrowRight');
    expect(arrow.defaultPrevented).toBe(true);
    expect(el.position).toBe(51);
    expect(log).toEqual([['input', 51], ['change', 51]]);
    expect(key(el, 'Tab').defaultPrevented).toBe(false);
    expect(key(el, 'ArrowRight', { ctrlKey: true }).defaultPrevented).toBe(false);
    expect(el.position).toBe(51);
  });

  it('supports Shift, PageUp/PageDown, Home and End with custom steps', () => {
    const el = make({ step: '5', 'big-step': '25' });
    key(el, 'ArrowRight');
    expect(el.position).toBe(55);
    key(el, 'ArrowRight', { shiftKey: true });
    expect(el.position).toBe(80);
    key(el, 'PageDown');
    expect(el.position).toBe(55);
    key(el, 'End');
    expect(el.position).toBe(100);
    key(el, 'Home');
    expect(el.position).toBe(0);
  });

  it('emits nothing when the value cannot change', () => {
    const el = make({ position: '100' });
    const log = record(el, 'input', 'change');
    key(el, 'End');
    key(el, 'ArrowRight');
    expect(log).toEqual([]);
  });

  it('uses the vertical mapping when vertical', () => {
    const el = make({ orientation: 'vertical' });
    key(el, 'ArrowDown');
    expect(el.position).toBe(51);
    key(el, 'ArrowUp');
    key(el, 'ArrowUp');
    expect(el.position).toBe(49);
  });

  it('flips left/right under direction: rtl', () => {
    const el = make();
    root(el).style.direction = 'rtl';
    key(el, 'ArrowRight');
    expect(el.position).toBe(49);
  });
});

describe('pointer', () => {
  it('mouse: press jumps to the point, moving drags, release commits one change', () => {
    const el = sized(make());
    const log = record(el, 'input', 'change');
    pointer(el, 'pointerdown', { clientX: 50, clientY: 10 });
    expect(el.position).toBe(25);
    expect(root(el).hasAttribute('data-dragging')).toBe(true);
    pointer(el, 'pointermove', { clientX: 150, clientY: 10 });
    pointer(el, 'pointermove', { clientX: 900, clientY: 10 });
    expect(el.position).toBe(100);
    pointer(el, 'pointerup', { clientX: 900, clientY: 10 });
    expect(root(el).hasAttribute('data-dragging')).toBe(false);
    expect(log.map(([t]) => t)).toEqual(['input', 'input', 'input', 'change']);
    expect(log.at(-1)).toEqual(['change', 100]);
  });

  it('ignores moves with no press, a second pointer, and the right mouse button', () => {
    const el = sized(make());
    pointer(el, 'pointermove', { clientX: 10, clientY: 0 });
    expect(el.position).toBe(50);
    pointer(el, 'pointerdown', { clientX: 100, clientY: 0, button: 2 });
    expect(el.position).toBe(50);
    pointer(el, 'pointerdown', { clientX: 20, clientY: 0, pointerId: 1 });
    pointer(el, 'pointermove', { clientX: 180, clientY: 0, pointerId: 2 });
    expect(el.position).toBe(10);
  });

  it('a cancelled gesture ends the drag and still reports the change', () => {
    const el = sized(make());
    const log = record(el, 'change');
    pointer(el, 'pointerdown', { clientX: 40, clientY: 0 });
    pointer(el, 'pointercancel', {});
    expect(root(el).hasAttribute('data-dragging')).toBe(false);
    expect(log).toEqual([['change', 20]]);
    pointer(el, 'pointermove', { clientX: 180, clientY: 0 });
    expect(el.position).toBe(20);
  });

  it('vertical orientation reads y', () => {
    const el = sized(make({ orientation: 'vertical' }));
    pointer(el, 'pointerdown', { clientX: 5, clientY: 25 });
    expect(el.position).toBe(25);
  });

  it('touch off the handle waits for horizontal travel, so a vertical scroll never moves it', () => {
    const el = sized(make());
    const log = record(el, 'input', 'change');
    pointer(el, 'pointerdown', { clientX: 100, clientY: 50, pointerType: 'touch' });
    expect(el.position).toBe(50);
    pointer(el, 'pointermove', { clientX: 102, clientY: 90, pointerType: 'touch' }); // mostly vertical
    pointer(el, 'pointermove', { clientX: 104, clientY: 20, pointerType: 'touch' });
    expect(el.position).toBe(50);
    pointer(el, 'pointercancel', { pointerType: 'touch' });
    expect(log).toEqual([]);
  });

  it('touch off the handle becomes a drag once it travels along the axis', () => {
    const el = sized(make());
    pointer(el, 'pointerdown', { clientX: 100, clientY: 50, pointerType: 'touch' });
    pointer(el, 'pointermove', { clientX: 120, clientY: 52, pointerType: 'touch' });
    expect(el.position).toBe(60);
    pointer(el, 'pointermove', { clientX: 20, clientY: 52, pointerType: 'touch' });
    expect(el.position).toBe(10);
  });

  it('a touch tap off the handle jumps there; a touch on the handle drags at once', () => {
    const el = sized(make());
    const log = record(el, 'change');
    pointer(el, 'pointerdown', { clientX: 40, clientY: 50, pointerType: 'touch' });
    pointer(el, 'pointerup', { clientX: 40, clientY: 50, pointerType: 'touch' });
    expect(el.position).toBe(20);
    expect(log).toEqual([['change', 20]]);
    pointer(el, 'pointerdown', { clientX: 40, clientY: 50, pointerType: 'touch', target: handle(el) });
    pointer(el, 'pointermove', { clientX: 160, clientY: 50, pointerType: 'touch' });
    expect(el.position).toBe(80);
  });

  it('RTL measures from the right', () => {
    const el = sized(make());
    root(el).style.direction = 'rtl';
    pointer(el, 'pointerdown', { clientX: 150, clientY: 0 });
    expect(el.position).toBe(25);
  });
});

describe('hover mode', () => {
  it('follows a hovering mouse without a press and reports change on leave', () => {
    const el = sized(make({ hover: '' }));
    const log = record(el, 'input', 'change');
    pointer(el, 'pointerenter', { clientX: 0, clientY: 0 });
    pointer(el, 'pointermove', { clientX: 120, clientY: 0 });
    pointer(el, 'pointermove', { clientX: 150, clientY: 0 });
    expect(el.position).toBe(75);
    pointer(el, 'pointerleave', {});
    expect(log.map(([t]) => t)).toEqual(['input', 'input', 'change']);
  });

  it('does nothing on hover when the option is off, and never reacts to touch hover', () => {
    const el = sized(make());
    pointer(el, 'pointermove', { clientX: 100, clientY: 0 });
    expect(el.position).toBe(50);
    el.hover = true;
    pointer(el, 'pointermove', { clientX: 10, clientY: 0, pointerType: 'touch' });
    expect(el.position).toBe(50);
  });
});

describe('images', () => {
  it('loads safe sources and reports ready once both have loaded', () => {
    const el = make({ 'before-src': 'a.webp', 'after-src': 'https://example.com/b.webp' });
    const ready = vi.fn();
    el.addEventListener('ready', ready);
    expect(el.status).toBe('loading');
    expect(el.getAttribute('data-status')).toBe('loading');
    img(el, 'before').dispatchEvent(new Event('load'));
    expect(ready).not.toHaveBeenCalled();
    img(el, 'after').dispatchEvent(new Event('load'));
    expect(ready).toHaveBeenCalledTimes(1);
    expect(el.status).toBe('ready');
    expect(img(el, 'before').getAttribute('src')).toBe('a.webp');
  });

  it('decodes off the main thread', () => {
    expect(img(make({ 'before-src': 'a.webp' }), 'before').getAttribute('decoding')).toBe('async');
  });

  describe('lazy loading', () => {
    let observers;
    beforeEach(() => {
      observers = [];
      vi.stubGlobal(
        'IntersectionObserver',
        class {
          constructor(callback, options) {
            this.callback = callback;
            this.options = options;
            this.disconnected = false;
            observers.push(this);
          }
          observe() {}
          disconnect() {
            this.disconnected = true;
          }
        },
      );
    });
    afterEach(() => vi.unstubAllGlobals());

    it('holds back the src until the slider is near the viewport, then sets both and stops watching', () => {
      const el = make({ 'before-src': 'a.webp', 'after-src': 'b.webp' });
      expect(img(el, 'before').hasAttribute('src')).toBe(false);
      expect(img(el, 'after').hasAttribute('src')).toBe(false);
      expect(el.status).toBe('loading');
      expect(observers).toHaveLength(1);
      expect(observers[0].options.rootMargin).toBe('300px');
      observers[0].callback([{ isIntersecting: false }]);
      expect(img(el, 'before').hasAttribute('src')).toBe(false);
      observers[0].callback([{ isIntersecting: true }]);
      expect(img(el, 'before').getAttribute('src')).toBe('a.webp');
      expect(img(el, 'after').getAttribute('src')).toBe('b.webp');
      expect(observers[0].disconnected).toBe(true);
      // Once visible, later source changes load straight away.
      el.beforeSrc = 'c.webp';
      expect(img(el, 'before').getAttribute('src')).toBe('c.webp');
      expect(observers).toHaveLength(1);
    });

    it('loading="eager" skips the observer, and switching to eager flushes pending images', () => {
      const eager = make({ 'before-src': 'a.webp', loading: 'eager' });
      expect(img(eager, 'before').getAttribute('src')).toBe('a.webp');
      expect(observers.every((o) => o.disconnected)).toBe(true); // the attribute order made one first
      const lazy = make({ 'after-src': 'b.webp' });
      expect(img(lazy, 'after').hasAttribute('src')).toBe(false);
      lazy.loading = 'eager';
      expect(img(lazy, 'after').getAttribute('src')).toBe('b.webp');
    });
  });

  it('a failed image sets the error status and an image-error event naming the side', () => {
    const el = make({ 'before-src': 'a.webp', 'after-src': 'b.webp' });
    const errors = [];
    el.addEventListener('image-error', (e) => errors.push(e.detail.which));
    img(el, 'after').dispatchEvent(new Event('error'));
    expect(errors).toEqual(['after']);
    expect(el.status).toBe('error');
    expect(root(el).dataset.status).toBe('error');
    img(el, 'before').dispatchEvent(new Event('load'));
    expect(el.status).toBe('error');
    // Fixing the source clears the error.
    el.afterSrc = 'c.webp';
    expect(el.status).toBe('loading');
  });

  it('with no src at all (slotted content) there is nothing to wait for', () => {
    expect(make().status).toBe('ready');
  });

  it('reserves the declared ratio, ignoring junk', () => {
    const el = make({ ratio: '3/2' });
    expect(root(el).style.getPropertyValue('--ic-ratio')).toBe('3 / 2');
    el.ratio = 'url(javascript:alert(1))';
    expect(root(el).style.getPropertyValue('--ic-ratio')).toBe('');
  });

  it('takes the ratio from the loaded image when none is declared', () => {
    const el = make({ 'after-src': 'b.webp' });
    Object.defineProperty(img(el, 'after'), 'naturalWidth', { value: 800 });
    Object.defineProperty(img(el, 'after'), 'naturalHeight', { value: 534 });
    img(el, 'after').dispatchEvent(new Event('load'));
    expect(root(el).style.getPropertyValue('--ic-ratio')).toBe('800 / 534');
  });
});

describe('untrusted attribute values', () => {
  const payload = '"><img src=x onerror="window.__pwned=1"><script>window.__pwned=1</script>';

  it('never turns alt, label or src values into markup', () => {
    const el = make({
      'before-alt': payload,
      'after-alt': payload,
      label: payload,
      'before-label': payload,
      'after-label': payload,
      'before-src': payload,
      'after-src': 'javascript:window.__pwned=1',
    });
    const all = shadow(el).querySelectorAll('*');
    expect([...all].some((n) => n.tagName === 'SCRIPT')).toBe(false);
    expect(shadow(el).querySelectorAll('img')).toHaveLength(2); // only the two we built
    expect(img(el, 'before').alt).toBe(payload);
    expect(shadow(el).querySelector('.tag-before .text').textContent).toBe(payload);
    expect(handle(el).getAttribute('aria-label')).toBe(payload);
    expect(window.__pwned).toBeUndefined();
  });

  it('refuses javascript: and non-image data: sources and reports an error instead', () => {
    const el = make();
    const errors = [];
    el.addEventListener('image-error', (e) => errors.push(e.detail.which));
    el.afterSrc = 'javascript:window.__pwned=1';
    el.beforeSrc = 'data:text/html,<script>window.__pwned=1</script>';
    expect(errors).toEqual(['after', 'before']);
    expect(img(el, 'after').hasAttribute('src')).toBe(false);
    expect(img(el, 'before').hasAttribute('src')).toBe(false);
    expect(el.status).toBe('error');
  });
});

describe('slots and parts', () => {
  it('exposes named parts for styling and slots for customisation', () => {
    const el = make();
    const parts = [...shadow(el).querySelectorAll('[part]')].flatMap((n) => n.getAttribute('part').split(' '));
    for (const part of ['root', 'before', 'after', 'before-image', 'after-image', 'handle', 'line', 'knob', 'label', 'label-before', 'label-after', 'track', 'error']) {
      expect(parts).toContain(part);
    }
    const slots = [...shadow(el).querySelectorAll('slot')].map((s) => s.name);
    expect(slots).toEqual(expect.arrayContaining(['before', 'after', 'handle', 'label-before', 'label-after', 'error']));
  });

  it('labels are hidden from assistive tech (aria-valuetext already says it)', () => {
    const el = make();
    for (const tag of shadow(el).querySelectorAll('.tag')) expect(tag.getAttribute('aria-hidden')).toBe('true');
  });
});

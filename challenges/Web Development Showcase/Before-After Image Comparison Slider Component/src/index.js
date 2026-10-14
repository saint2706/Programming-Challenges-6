import { CSS } from './styles.js';
import { isSafeImageUrl, keyTarget, parsePosition, parseRatio, positionFromPoint, valueText } from './position.js';

export { isSafeImageUrl, keyTarget, parsePosition, parseRatio, positionFromPoint, valueText };

// Importing this module on a server (no DOM) must not throw, so nothing here touches
// HTMLElement, customElements or document until a component is actually constructed.
const Base = /** @type {typeof HTMLElement} */ (typeof HTMLElement === 'undefined' ? class {} : HTMLElement);

/** Static markup only; no user value is ever interpolated into it. */
const TEMPLATE = `
<div class="root" part="root">
  <div class="layer after" part="after"><slot name="after"><img part="after-image" draggable="false" decoding="async"></slot>
    <span class="tag tag-after" part="label label-after" aria-hidden="true"><slot name="label-after"><span class="text"></span></slot></span>
  </div>
  <div class="layer before" part="before"><slot name="before"><img part="before-image" draggable="false" decoding="async"></slot>
    <span class="tag tag-before" part="label label-before" aria-hidden="true"><slot name="label-before"><span class="text"></span></slot></span>
  </div>
  <div class="track" part="track"><div class="handle" part="handle" role="slider" tabindex="0">
    <span class="line" part="line"></span>
    <span class="knob" part="knob"><slot name="handle"><svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M9 6l-6 6 6 6M15 6l6 6-6 6" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg></slot></span>
  </div></div>
  <div class="error" part="error" role="alert"><slot name="error">Image failed to load</slot></div>
</div>`;

const ATTRIBUTES = [
  'before-src', 'after-src', 'before-alt', 'after-alt', 'position', 'orientation', 'hover', 'fit',
  'label', 'before-label', 'after-label', 'step', 'big-step', 'ratio', 'loading',
];

/** How far outside the viewport (px) a lazy slider starts fetching its images. */
const LAZY_MARGIN = 300;

/** Pixels a touch must travel along the slider axis before it counts as a drag rather than a tap. */
const TOUCH_SLOP = 6;

let sheet = null;
function adoptStyles(root) {
  if (typeof CSSStyleSheet !== 'undefined' && 'adoptedStyleSheets' in root && CSSStyleSheet.prototype.replaceSync) {
    sheet ??= new CSSStyleSheet();
    if (!sheet.cssRules.length) sheet.replaceSync(CSS);
    root.adoptedStyleSheets = [sheet];
  } else {
    const style = document.createElement('style');
    style.textContent = CSS;
    root.append(style);
  }
}

/**
 * `<img-compare before-src="a.webp" after-src="b.webp">`: a before/after slider.
 *
 * `position` is the percentage of the box, from its start edge, that shows the "before" image.
 * Events: `input` (while moving), `change` (when a move ends), `ready` (both images loaded),
 * `image-error` (one failed; `detail.which`). `input`/`change` carry `detail.position`.
 */
export class ImgCompare extends Base {
  static observedAttributes = ATTRIBUTES;

  #root;
  #handle;
  #images;
  #texts;
  #position = 50;
  // A side with no `*-src` is filled through its slot, so there is nothing to wait for.
  #loaded = { before: true, after: true };
  #failed = { before: false, after: false };
  /** @type {{id:number, x:number, y:number, pending:boolean, rtl:boolean, start:number} | null} */
  #drag = null;
  /** Hover-follow bookkeeping: the position and text direction when the pointer entered. */
  #hoverStart = null;
  #hoverRtl = false;
  // Why not `<img loading="lazy">`: Chrome treats an image inside a fully clipped `clip-path` as
  // not visible, so at position 0 the "before" image would never load until the first drag.
  #sources = { before: '', after: '' };
  #visible = false;
  /** @type {IntersectionObserver | null} */
  #observer = null;

  constructor() {
    super();
    const shadow = this.attachShadow({ mode: 'open' });
    adoptStyles(shadow);
    const template = document.createElement('template');
    template.innerHTML = TEMPLATE;
    shadow.append(template.content);

    this.#root = shadow.querySelector('.root');
    this.#handle = shadow.querySelector('.handle');
    this.#images = { before: shadow.querySelector('.before img'), after: shadow.querySelector('.after img') };
    this.#texts = { before: shadow.querySelector('.tag-before .text'), after: shadow.querySelector('.tag-after .text') };

    for (const [which, img] of Object.entries(this.#images)) {
      img.addEventListener('load', () => this.#onLoad(which));
      img.addEventListener('error', () => this.#onError(which));
    }
    this.#root.addEventListener('pointerdown', (e) => this.#onPointerDown(e));
    this.#root.addEventListener('pointermove', (e) => this.#onPointerMove(e));
    this.#root.addEventListener('pointerup', (e) => this.#onPointerEnd(e));
    this.#root.addEventListener('pointercancel', (e) => this.#onPointerEnd(e));
    this.#root.addEventListener('lostpointercapture', (e) => this.#onPointerEnd(e));
    this.#root.addEventListener('pointerenter', (e) => this.#onPointerEnter(e));
    this.#root.addEventListener('pointerleave', (e) => this.#onPointerLeave(e));
    this.#handle.addEventListener('keydown', (e) => this.#onKeyDown(e));

    for (const name of ATTRIBUTES) if (!name.endsWith('-src')) this.#sync(name, this.getAttribute(name));
  }

  connectedCallback() {
    // Properties set before the element was upgraded would otherwise shadow the accessors.
    for (const name of ['position', 'orientation', 'hover', 'fit', 'beforeSrc', 'afterSrc', 'beforeAlt', 'afterAlt', 'label', 'beforeLabel', 'afterLabel', 'step', 'bigStep', 'ratio', 'loading']) {
      if (Object.hasOwn(this, name)) {
        const value = this[name];
        delete this[name];
        this[name] = value;
      }
    }
    this.#renderStatus();
  }

  attributeChangedCallback(name, _old, value) {
    this.#sync(name, value);
  }

  // ---- properties (each one reflects to its attribute) -------------------------------------

  /** Percentage (0 to 100) of the box, from the start edge, showing the "before" image. */
  get position() {
    return this.#position;
  }
  set position(value) {
    this.setAttribute('position', String(parsePosition(value, this.#position)));
  }

  /** @returns {'horizontal' | 'vertical'} */
  get orientation() {
    return this.getAttribute('orientation') === 'vertical' ? 'vertical' : 'horizontal';
  }
  set orientation(value) {
    this.setAttribute('orientation', value === 'vertical' ? 'vertical' : 'horizontal');
  }

  /** Follow a hovering mouse/pen without pressing. Touch still drags. */
  get hover() {
    return this.hasAttribute('hover');
  }
  set hover(value) {
    this.toggleAttribute('hover', Boolean(value));
  }

  /** @returns {'cover' | 'contain'} */
  get fit() {
    return this.getAttribute('fit') === 'contain' ? 'contain' : 'cover';
  }
  set fit(value) {
    this.setAttribute('fit', value === 'contain' ? 'contain' : 'cover');
  }

  get beforeSrc() { return this.getAttribute('before-src') ?? ''; }
  set beforeSrc(v) { this.setAttribute('before-src', String(v)); }
  get afterSrc() { return this.getAttribute('after-src') ?? ''; }
  set afterSrc(v) { this.setAttribute('after-src', String(v)); }
  get beforeAlt() { return this.getAttribute('before-alt') ?? ''; }
  set beforeAlt(v) { this.setAttribute('before-alt', String(v)); }
  get afterAlt() { return this.getAttribute('after-alt') ?? ''; }
  set afterAlt(v) { this.setAttribute('after-alt', String(v)); }
  /** Accessible name of the slider. */
  get label() { return this.getAttribute('label') ?? 'Image comparison'; }
  set label(v) { this.setAttribute('label', String(v)); }
  get beforeLabel() { return this.getAttribute('before-label') ?? 'Before'; }
  set beforeLabel(v) { this.setAttribute('before-label', String(v)); }
  get afterLabel() { return this.getAttribute('after-label') ?? 'After'; }
  set afterLabel(v) { this.setAttribute('after-label', String(v)); }
  /** Keyboard step, in percentage points. */
  get step() { return positiveNumber(this.getAttribute('step'), 1); }
  set step(v) { this.setAttribute('step', String(v)); }
  /** Shift+arrow and PageUp/PageDown step. */
  get bigStep() { return positiveNumber(this.getAttribute('big-step'), 10); }
  set bigStep(v) { this.setAttribute('big-step', String(v)); }
  /** Width/height reserved before the images load, e.g. "3/2". Otherwise taken from the images. */
  get ratio() { return this.getAttribute('ratio') ?? ''; }
  set ratio(v) { this.setAttribute('ratio', String(v)); }

  /** `lazy` (default): fetch the images once the slider is near the viewport. `eager`: right away. */
  get loading() { return this.getAttribute('loading') === 'eager' ? 'eager' : 'lazy'; }
  set loading(v) { this.setAttribute('loading', v === 'eager' ? 'eager' : 'lazy'); }

  /** @returns {'loading' | 'ready' | 'error'} */
  get status() {
    if (this.#failed.before || this.#failed.after) return 'error';
    return this.#loaded.before && this.#loaded.after ? 'ready' : 'loading';
  }

  // ---- rendering ---------------------------------------------------------------------------

  #sync(name, value) {
    switch (name) {
      case 'position':
        this.#position = parsePosition(value, 50);
        this.#renderPosition();
        break;
      case 'before-src':
      case 'after-src':
        this.#setSource(name.startsWith('before') ? 'before' : 'after', value);
        break;
      case 'before-alt':
      case 'after-alt':
        this.#images[name.startsWith('before') ? 'before' : 'after'].alt = value ?? '';
        break;
      case 'orientation':
        this.#root.dataset.orientation = this.orientation;
        this.#handle.setAttribute('aria-orientation', this.orientation);
        break;
      case 'fit':
        this.#root.dataset.fit = this.fit;
        break;
      case 'hover':
        this.#root.toggleAttribute('data-follow', this.hover);
        break;
      case 'label':
        this.#handle.setAttribute('aria-label', this.label);
        break;
      case 'before-label':
      case 'after-label':
        this.#texts.before.textContent = this.beforeLabel;
        this.#texts.after.textContent = this.afterLabel;
        this.#renderPosition();
        break;
      case 'ratio':
        this.#renderRatio();
        break;
      case 'loading':
        this.#attach('before');
        this.#attach('after');
        break;
      default:
        break;
    }
  }

  #renderPosition() {
    const p = this.#position;
    this.#root.style.setProperty('--ic-pos', String(p));
    this.#handle.setAttribute('aria-valuenow', String(Math.round(p * 10) / 10));
    this.#handle.setAttribute('aria-valuetext', valueText(p, this.beforeLabel, this.afterLabel));
  }

  #renderRatio() {
    const declared = parseRatio(this.getAttribute('ratio'));
    const natural = [this.#images.after, this.#images.before].find((img) => img.naturalWidth > 0 && img.naturalHeight > 0);
    const ratio = declared ?? (natural ? `${natural.naturalWidth} / ${natural.naturalHeight}` : null);
    if (ratio) this.#root.style.setProperty('--ic-ratio', ratio);
    else this.#root.style.removeProperty('--ic-ratio');
  }

  #renderStatus() {
    const status = this.status;
    this.#root.dataset.status = status;
    this.setAttribute('data-status', status);
    return status;
  }

  #setSource(which, value) {
    this.#loaded[which] = false;
    this.#failed[which] = false;
    const img = this.#images[which];
    this.#sources[which] = '';
    img.removeAttribute('src');
    if (!value) {
      // No src: the side is provided through the slot (or not at all), so there is nothing to wait for.
      this.#loaded[which] = true;
    } else if (isSafeImageUrl(value, this.baseURI)) {
      this.#sources[which] = value;
      this.#attach(which);
    } else {
      this.#onError(which);
      return;
    }
    this.#renderStatus();
  }

  /** Give the <img> its src now, or once the slider is near the viewport when lazy. */
  #attach(which) {
    const value = this.#sources[which];
    if (!value) return;
    if (this.loading === 'lazy' && !this.#visible && typeof IntersectionObserver !== 'undefined') {
      this.#observer ??= new IntersectionObserver(
        (entries) => {
          if (!entries.some((entry) => entry.isIntersecting)) return;
          this.#visible = true;
          this.#observer.disconnect();
          this.#observer = null;
          this.#attach('before');
          this.#attach('after');
        },
        { rootMargin: `${LAZY_MARGIN}px` },
      );
      this.#observer.observe(this);
      return;
    }
    // Eager (possibly just switched to): nothing left to wait for.
    this.#observer?.disconnect();
    this.#observer = null;
    const img = this.#images[which];
    if (img.getAttribute('src') !== value) img.src = value;
  }

  #onLoad(which) {
    this.#loaded[which] = true;
    this.#failed[which] = false;
    this.#renderRatio();
    if (this.#renderStatus() === 'ready') this.dispatchEvent(new CustomEvent('ready'));
  }

  #onError(which) {
    this.#failed[which] = true;
    this.#renderStatus();
    this.dispatchEvent(new CustomEvent('image-error', { detail: { which } }));
  }

  // ---- value changes -----------------------------------------------------------------------

  #setPosition(value) {
    const next = parsePosition(value, this.#position);
    if (next === this.#position) return false;
    this.position = next;
    this.#emit('input');
    return true;
  }

  #emit(type) {
    this.dispatchEvent(new CustomEvent(type, { bubbles: true, composed: true, detail: { position: this.#position } }));
  }

  #direction() {
    return getComputedStyle(this.#root).direction === 'rtl';
  }

  #pointToPosition(e, rtl) {
    const rect = this.#root.getBoundingClientRect();
    return positionFromPoint(rect, e.clientX, e.clientY, { vertical: this.orientation === 'vertical', rtl });
  }

  // ---- keyboard ----------------------------------------------------------------------------

  #onKeyDown(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const next = keyTarget(e.key, this.#position, {
      vertical: this.orientation === 'vertical',
      rtl: this.#direction(),
      step: this.step,
      bigStep: this.bigStep,
      shift: e.shiftKey,
    });
    if (next === null) return;
    e.preventDefault();
    if (this.#setPosition(next)) this.#emit('change');
  }

  // ---- pointer -----------------------------------------------------------------------------

  #onPointerDown(e) {
    if (this.#drag || (e.pointerType === 'mouse' && e.button !== 0)) return;
    const onHandle = e.composedPath().includes(this.#handle);
    // A touch that lands off the handle might be the start of a page scroll, so it only becomes a
    // drag after it moves along the slider axis (touch-action hands vertical pans to the browser).
    const pending = e.pointerType === 'touch' && !onHandle;
    this.#drag = { id: e.pointerId, x: e.clientX, y: e.clientY, pending, rtl: this.#direction(), start: this.#position };
    try {
      this.#root.setPointerCapture(e.pointerId);
    } catch {
      // A synthetic or already-gone pointer id: dragging still works while the pointer is over us.
    }
    if (!pending) this.#beginDrag(e);
  }

  #beginDrag(e) {
    this.#drag.pending = false;
    this.#root.toggleAttribute('data-dragging', true);
    this.#handle.focus({ preventScroll: true });
    this.#setPosition(this.#pointToPosition(e, this.#drag.rtl));
  }

  #onPointerMove(e) {
    const drag = this.#drag;
    if (!drag) {
      if (this.hover && e.pointerType !== 'touch') this.#setPosition(this.#pointToPosition(e, this.#hoverRtl));
      return;
    }
    if (e.pointerId !== drag.id) return;
    if (drag.pending) {
      const vertical = this.orientation === 'vertical';
      const along = Math.abs(vertical ? e.clientY - drag.y : e.clientX - drag.x);
      const across = Math.abs(vertical ? e.clientX - drag.x : e.clientY - drag.y);
      if (along < TOUCH_SLOP || along < across) return;
      this.#beginDrag(e);
      return;
    }
    this.#setPosition(this.#pointToPosition(e, drag.rtl));
  }

  #onPointerEnd(e) {
    const drag = this.#drag;
    if (!drag || e.pointerId !== drag.id) return;
    this.#drag = null;
    this.#root.removeAttribute('data-dragging');
    if (e.type === 'pointerup' && drag.pending) {
      // A tap: jump there.
      this.#handle.focus({ preventScroll: true });
      this.#setPosition(this.#pointToPosition(e, drag.rtl));
    }
    if (this.#position !== drag.start) this.#emit('change');
  }

  #onPointerEnter(e) {
    if (!this.hover || e.pointerType === 'touch') return;
    this.#hoverRtl = this.#direction();
    this.#hoverStart = this.#position;
  }

  #onPointerLeave(e) {
    if (this.#hoverStart === null || e.pointerType === 'touch') return;
    if (this.#position !== this.#hoverStart) this.#emit('change');
    this.#hoverStart = null;
  }
}

function positiveNumber(value, fallback) {
  const n = parseFloat(value);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

/** Register the element (idempotent, and a no-op without a DOM). Returns whether it is defined. */
export function define(tag = 'img-compare') {
  if (typeof customElements === 'undefined') return false;
  if (!customElements.get(tag)) customElements.define(tag, ImgCompare);
  return true;
}

define();

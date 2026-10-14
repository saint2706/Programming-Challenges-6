// The stylesheet is a string so the package is one JS file with no CSS to import.
// Everything themeable is a CSS custom property or a ::part, documented in the README.
export const CSS = `
:host {
  --ic-accent: #fff;
  --ic-line-width: 2px;
  --ic-line-color: var(--ic-accent);
  --ic-knob-size: 44px;
  --ic-knob-bg: rgb(0 0 0 / 55%);
  --ic-knob-color: #fff;
  --ic-focus-ring: #4d9fff;
  --ic-label-bg: rgb(0 0 0 / 70%);
  --ic-label-color: #fff;
  --ic-radius: 0px;
  --ic-duration: 140ms;
  display: block;
  position: relative;
}
:host([hidden]) { display: none; }
.root {
  position: relative;
  width: 100%;
  height: 100%;
  aspect-ratio: var(--ic-ratio, 16 / 9);
  overflow: hidden;
  border-radius: var(--ic-radius);
  background: #1114;
  touch-action: pan-y;
  user-select: none;
  -webkit-user-select: none;
  cursor: ew-resize;
}
.root[data-orientation="vertical"] { touch-action: pan-x; cursor: ns-resize; }
.layer { position: absolute; inset: 0; }
.layer img, .layer > slot::slotted(*) {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  pointer-events: none;
  -webkit-user-drag: none;
}
.root[data-fit="contain"] .layer img, :host([fit="contain"]) .layer > slot::slotted(*) { object-fit: contain; }
.before {
  clip-path: inset(0 calc(100% - var(--ic-pos) * 1%) 0 0);
  transition: clip-path var(--ic-duration) ease-out;
}
:host(:dir(rtl)) .before { clip-path: inset(0 0 0 calc(100% - var(--ic-pos) * 1%)); }
.root[data-orientation="vertical"] .before,
:host(:dir(rtl)) .root[data-orientation="vertical"] .before { clip-path: inset(0 0 calc(100% - var(--ic-pos) * 1%) 0); }

.tag {
  position: absolute;
  inset-block-start: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  background: var(--ic-label-bg);
  color: var(--ic-label-color);
  font: 600 13px/1.3 system-ui, sans-serif;
  pointer-events: none;
}
.tag-before { inset-inline-start: 12px; }
.tag-after { inset-inline-end: 12px; }
.root[data-orientation="vertical"] .tag-after { inset-block-start: auto; inset-block-end: 12px; inset-inline-end: auto; inset-inline-start: 12px; }

.track {
  position: absolute;
  inset-block: 0;
  inset-inline-start: calc(var(--ic-pos) * 1%);
  width: 0;
  display: flex;
  justify-content: center;
  pointer-events: none;
  transition: inset-inline-start var(--ic-duration) ease-out;
}
.root[data-orientation="vertical"] .track {
  inset-block: auto;
  inset-inline: 0;
  inset-block-start: calc(var(--ic-pos) * 1%);
  width: auto;
  height: 0;
  align-items: center;
  transition: inset-block-start var(--ic-duration) ease-out;
}
.handle {
  position: relative;
  flex: none;
  width: var(--ic-knob-size);
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
  touch-action: none;
  cursor: inherit;
  outline: none;
}
.root[data-orientation="vertical"] .handle { width: 100%; height: var(--ic-knob-size); }
.line {
  position: absolute;
  inset-block: 0;
  left: 50%;
  width: var(--ic-line-width);
  translate: -50% 0;
  background: var(--ic-line-color);
  box-shadow: 0 0 6px rgb(0 0 0 / 40%);
}
.root[data-orientation="vertical"] .line {
  inset-block: auto;
  inset-inline: 0;
  left: 0;
  top: 50%;
  width: auto;
  height: var(--ic-line-width);
  translate: 0 -50%;
}
.knob {
  position: relative;
  display: grid;
  place-items: center;
  width: var(--ic-knob-size);
  height: var(--ic-knob-size);
  border-radius: 50%;
  background: var(--ic-knob-bg);
  color: var(--ic-knob-color);
  box-shadow: 0 0 0 2px var(--ic-line-color), 0 2px 8px rgb(0 0 0 / 35%);
}
.knob svg { width: 60%; height: 60%; }
.root[data-orientation="vertical"] .knob svg { rotate: 90deg; }
.handle:focus-visible .knob { outline: 3px solid var(--ic-focus-ring); outline-offset: 3px; }

.root[data-dragging] .before, .root[data-dragging] .track,
.root[data-follow] .before, .root[data-follow] .track { transition: none; }
.error {
  position: absolute;
  inset: 0;
  display: none;
  place-items: center;
  background: rgb(20 20 20 / 85%);
  color: #fff;
  font: 14px system-ui, sans-serif;
  text-align: center;
  padding: 1em;
}
.root[data-status="error"] .error { display: grid; }
@media (prefers-reduced-motion: reduce) {
  .before, .track { transition: none; }
}
`;

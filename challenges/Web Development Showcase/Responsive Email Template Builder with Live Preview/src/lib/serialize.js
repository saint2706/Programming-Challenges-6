// Pure, DOM-free block -> email-safe-HTML serialization. No Svelte/browser
// dependency so this can be unit-tested directly, same pattern this repo's
// other Web Development Showcase challenges use for their math/logic layer.

const ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (ch) => ESCAPE_MAP[ch]);
}

const UNSAFE_SCHEMES = ['javascript:', 'vbscript:', 'data:'];

export function isSafeHref(href) {
  if (typeof href !== 'string' || href.trim() === '') return false;
  // Strip whitespace and ASCII control characters before the scheme check —
  // browsers ignore them when resolving a URL scheme, so `java\tscript:` is
  // just as dangerous as `javascript:` and must not slip past a naive check.
  const normalized = href.replace(/[\s\u0000-\u001f]/g, '').toLowerCase();
  return !UNSAFE_SCHEMES.some((scheme) => normalized.startsWith(scheme));
}

export function safeHref(href, fallback = '#') {
  return isSafeHref(href) ? escapeHtml(href) : fallback;
}

function renderHeading(props) {
  const size = Number(props.fontSize) || 24;
  return `
      <tr>
        <td style="padding:16px 24px 0 24px; text-align:${escapeHtml(props.align || 'left')};">
          <h1 style="margin:0; font-family:Arial,Helvetica,sans-serif; font-size:${size}px; line-height:1.3; color:${escapeHtml(props.color || '#111827')}; font-weight:bold;">${escapeHtml(props.text)}</h1>
        </td>
      </tr>`;
}

function renderParagraph(props) {
  const size = Number(props.fontSize) || 14;
  return `
      <tr>
        <td style="padding:8px 24px; text-align:${escapeHtml(props.align || 'left')};">
          <p style="margin:0; font-family:Arial,Helvetica,sans-serif; font-size:${size}px; line-height:1.6; color:${escapeHtml(props.color || '#374151')};">${escapeHtml(props.text)}</p>
        </td>
      </tr>`;
}

function renderImage(props) {
  const width = Number(props.width) || 600;
  return `
      <tr>
        <td style="padding:8px 24px;">
          <img src="${safeHref(props.src)}" alt="${escapeHtml(props.alt || '')}" width="${width}" style="display:block; width:100%; max-width:${width}px; height:auto; border:0;" />
        </td>
      </tr>`;
}

// The "bulletproof button" technique: a table-wrapped anchor styled as a
// button. Real email clients (notably Outlook, which renders via Word's HTML
// engine) unreliably apply padding directly on an <a>, so the padding lives
// on the anchor while the clickable/colored surface is guaranteed by the
// wrapping <td> background instead.
function renderButton(props) {
  const align = props.align === 'left' || props.align === 'right' ? props.align : 'center';
  const margin = align === 'center' ? '0 auto' : '0';
  return `
      <tr>
        <td style="padding:16px 24px; text-align:${align};">
          <table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin:${margin};">
            <tr>
              <td style="border-radius:4px; background-color:${escapeHtml(props.bgColor || '#2563eb')};">
                <a href="${safeHref(props.href)}" target="_blank" rel="noopener noreferrer" style="display:inline-block; padding:12px 28px; font-family:Arial,Helvetica,sans-serif; font-size:14px; font-weight:bold; color:${escapeHtml(props.textColor || '#ffffff')}; text-decoration:none; border-radius:4px;">${escapeHtml(props.label)}</a>
              </td>
            </tr>
          </table>
        </td>
      </tr>`;
}

function renderDivider(props) {
  const height = Number(props.height) || 1;
  return `
      <tr>
        <td style="padding:16px 24px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr><td style="font-size:0; line-height:0; border-top:${height}px solid ${escapeHtml(props.color || '#e5e7eb')};">&nbsp;</td></tr>
          </table>
        </td>
      </tr>`;
}

function renderSpacer(props) {
  const height = Number(props.height) || 24;
  return `
      <tr>
        <td style="font-size:0; line-height:0; height:${height}px;" height="${height}">&nbsp;</td>
      </tr>`;
}

function renderColumns(props) {
  return `
      <tr>
        <td style="padding:8px 24px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0">
            <tr>
              <td width="50%" valign="top" style="padding:0 8px 0 0; font-family:Arial,Helvetica,sans-serif; font-size:14px; line-height:1.6; color:#374151;">${escapeHtml(props.leftText)}</td>
              <td width="50%" valign="top" style="padding:0 0 0 8px; font-family:Arial,Helvetica,sans-serif; font-size:14px; line-height:1.6; color:#374151;">${escapeHtml(props.rightText)}</td>
            </tr>
          </table>
        </td>
      </tr>`;
}

const RENDERERS = {
  heading: renderHeading,
  paragraph: renderParagraph,
  image: renderImage,
  button: renderButton,
  divider: renderDivider,
  spacer: renderSpacer,
  columns: renderColumns,
};

export function renderBlock(block) {
  const renderer = RENDERERS[block.type];
  if (!renderer) throw new Error(`Unknown block type: ${block.type}`);
  return renderer(block.props || {});
}

export function serializeBlocks(blocks) {
  return blocks.map(renderBlock).join('');
}

// Produces a full, standalone email HTML document: inline styles only (no
// <style> block — many clients strip <head>), nested role="presentation"
// tables for layout, suitable both for the sandboxed iframe preview and as
// the final exported/copied HTML.
export function serializeEmail(blocks, { width = 600, previewText = '' } = {}) {
  const rows = serializeBlocks(blocks);
  const safePreview = escapeHtml(previewText);
  return `<!doctype html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<meta http-equiv="X-UA-Compatible" content="IE=edge" />
<title>Email</title>
</head>
<body style="margin:0; padding:0; background-color:#f3f4f6;">
${safePreview ? `<div style="display:none; max-height:0; overflow:hidden; mso-hide:all;">${safePreview}</div>` : ''}
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color:#f3f4f6;">
  <tr>
    <td align="center" style="padding:24px 12px;">
      <table role="presentation" width="${width}" cellpadding="0" cellspacing="0" border="0" style="width:${width}px; max-width:100%; background-color:#ffffff;">${rows}
      </table>
    </td>
  </tr>
</table>
</body>
</html>`;
}

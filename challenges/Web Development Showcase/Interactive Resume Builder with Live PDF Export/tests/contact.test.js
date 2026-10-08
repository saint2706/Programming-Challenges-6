import { fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../src/App.svelte';
import Preview from '../src/components/Preview.svelte';
import { buildRegions } from '../src/lib/blocks.js';
import { exportJson, importJson } from '../src/lib/io.js';
import { isEmail, isSafeHref, mailHref, telHref } from '../src/lib/links.js';
import { setLayout, setPath } from '../src/lib/model.js';
import { parseResume } from '../src/lib/schema.js';
import { longResume } from '../src/lib/sample-long.js';
import { seedResume } from '../src/lib/seed.js';
import { STORAGE_KEY } from '../src/lib/storage.js';

// jsdom has no layout engine: give every block and list row a fixed height.
let rectSpy;
beforeEach(() => {
  localStorage.clear();
  rectSpy = vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function rect() {
    const height = this.hasAttribute('data-row') ? 20 : this.hasAttribute('data-block') ? 60 : 0;
    return { height, width: 0, top: 0, left: 0, right: 0, bottom: height, x: 0, y: 0, toJSON() {} };
  });
});
afterEach(() => {
  rectSpy.mockRestore();
  vi.restoreAllMocks();
  localStorage.clear();
});

const EMAIL = 'ada.lovelace@example.com';
const PHONE = '+1 (555) 123-4567';
const PHONE_HREF = 'tel:+15551234567';

const withContact = (doc, template, email = EMAIL, phone = PHONE) =>
  setLayout(setPath(setPath(doc, 'basics.email', email), 'basics.phone', phone), { template });

const HOSTILE_EMAILS = [
  'ada@example.com?bcc=eve@example.net',
  'ada@example.com?subject=hi',
  'ada@example.com#fragment',
  'ada@exa%6Dple.com',
  'ada@example.com,eve@example.net',
  'ada@example.com;eve@example.net',
  '<ada@example.com>',
  '"ada"@example.com',
  'javascript:alert(1)//@example.com',
  'ada@example.com\nBcc: eve@example.net',
  'ada\u0000@example.com',
  'ada @example.com',
  `${'a'.repeat(250)}@example.com`,
];

describe('isEmail', () => {
  it.each([
    'ada@example.com',
    'first.last+tag@sub.example.co.uk',
    "o'brien@example.com",
    'josé@exämple.com',
    'ada@例え.jp',
  ])('accepts %j', (value) => {
    expect(isEmail(value)).toBe(true);
  });

  it.each(['', 'ada', 'ada@', '@example.com', 'ada@example', 'ada@@example.com', 'ada@.com', 'ada@example..com', ' ada@example.com', 'ada@example.com ', null, 42, undefined])(
    'rejects %j',
    (value) => {
      expect(isEmail(value)).toBe(false);
    },
  );

  it.each(HOSTILE_EMAILS)('rejects the hostile address %j', (value) => {
    expect(isEmail(value)).toBe(false);
  });
});

describe('mailHref and telHref', () => {
  it('builds a safe mailto link and refuses anything that is not an address', () => {
    expect(mailHref(' ada@example.com ')).toBe('mailto:ada@example.com');
    expect(isSafeHref(mailHref('ada@example.com'))).toBe(true);
    for (const value of HOSTILE_EMAILS) expect(mailHref(value)).toBeNull();
  });

  it('normalises a phone number to a dialable tel link and refuses one with no digits', () => {
    expect(telHref(PHONE)).toBe(PHONE_HREF);
    expect(telHref('555.0100')).toBe('tel:5550100');
    expect(telHref('+49 30 1234567')).toBe('tel:+49301234567');
    expect(isSafeHref(telHref(PHONE))).toBe(true);
    for (const value of ['', '+', '()', '12', ' ', 'abc', 'tel:+15551234567', null, undefined]) {
      expect(telHref(value)).toBeNull();
    }
  });
});

describe('email and phone in the schema and in imports', () => {
  it.each(HOSTILE_EMAILS)('parseResume rejects the hostile address %j', (email) => {
    const result = parseResume({ basics: { email } });
    expect(result.ok).toBe(false);
    expect(result.errors.some((e) => e.path.includes('email'))).toBe(true);
  });

  it.each(['tel:+15551234567', '555-0100 ext 5', '+1 555 0100;ext=1', '+1 555 0100\n', '+1 555 0100 <x>'])(
    'parseResume rejects the phone %j',
    (phone) => {
      expect(parseResume({ basics: { phone } }).ok).toBe(false);
    },
  );

  it('rejects a whole import that carries one hostile address, applying none of it', () => {
    const text = JSON.stringify({
      basics: { name: 'Ada', email: 'ada@example.com?subject=hi' },
      work: [{ name: 'Engines', position: 'Analyst' }],
    });
    const result = importJson(text);
    expect(result.ok).toBe(false);
    expect(result.doc).toBeUndefined();
  });

  it('keeps an address and a number through an export and an import, unchanged', () => {
    const doc = withContact(longResume(), 'classic', 'josé@exämple.com', '+49 30 1234567');
    const result = importJson(exportJson(doc));
    expect(result.ok).toBe(true);
    expect(result.doc.basics.email).toBe('josé@exämple.com');
    expect(result.doc.basics.phone).toBe('+49 30 1234567');
  });
});

describe('email and phone in the block builder', () => {
  // The sidebar template has a block of its own for the contact line; the others put it in the header.
  const contactOf = (doc) => {
    const blocks = buildRegions(doc).flatMap((r) => r.blocks);
    const standalone = blocks.find((b) => b.kind === 'contact');
    return standalone ? standalone.data : blocks.find((b) => b.kind === 'header').data.contact;
  };

  it.each(['classic', 'sidebar', 'compact'])('puts a mailto and a tel link in the %s template', (template) => {
    const contact = contactOf(withContact(seedResume(), template));
    expect(contact.email).toEqual({ label: EMAIL, href: `mailto:${EMAIL}` });
    expect(contact.phone).toEqual({ label: PHONE, href: PHONE_HREF });
  });

  it('shows a value that bypassed the schema as plain text, never as a link', () => {
    const doc = withContact(seedResume(), 'classic', 'ada@example.com?subject=hi', '()');
    const contact = contactOf(doc);
    expect(contact.email).toEqual({ label: 'ada@example.com?subject=hi', href: null });
    expect(contact.phone).toEqual({ label: '()', href: null });
  });

  it('leaves the contact line out when everything is blank', () => {
    const contact = buildRegions(withContact(seedResume(), 'classic', '', ''))[0].blocks[0].data.contact;
    expect(contact.email).toBeNull();
    expect(contact.phone).toBeNull();
  });
});

describe('email and phone on the page', () => {
  async function renderPreview(doc) {
    const utils = render(Preview, { props: { doc, onedit: vi.fn() } });
    await waitFor(() => expect(utils.container.querySelector('[data-ready="true"]')).not.toBeNull());
    return utils;
  }
  const anchors = (container, href) => container.querySelectorAll(`article.page a[href="${href}"]`);

  it.each(['classic', 'sidebar', 'compact'])('renders one mailto link and one tel link in the %s template', async (template) => {
    const { container } = await renderPreview(withContact(seedResume(), template));
    const mail = anchors(container, `mailto:${EMAIL}`);
    const tel = anchors(container, PHONE_HREF);
    expect(mail).toHaveLength(1);
    expect(tel).toHaveLength(1);
    expect(mail[0]).toHaveTextContent(EMAIL);
    expect(tel[0]).toHaveTextContent(PHONE);
    for (const a of [...mail, ...tel]) expect(a).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('does not make a link of a hostile value, but still shows it as text', async () => {
    const hostile = 'ada@example.com?subject=hi';
    const { container } = await renderPreview(withContact(seedResume(), 'classic', hostile, PHONE));
    expect(container.querySelector('article.page a[href^="mailto:"]')).toBeNull();
    expect(container.querySelector('article.page .contact')).toHaveTextContent(hostile);
  });

  it('keeps the contact entries out of in-place editing: they are edited in the side panel', async () => {
    const { container } = await renderPreview(withContact(seedResume(), 'classic'));
    expect(container.querySelector('article.page .contact')).not.toBeNull();
    expect(container.querySelectorAll('article.page .contact [contenteditable]')).toHaveLength(0);
    expect(container.querySelectorAll('article.page .contact [data-path]')).toHaveLength(0);
  });
});

describe('email and phone in the side panel', () => {
  async function renderApp() {
    const utils = render(App);
    await waitFor(() => expect(utils.container.querySelector('[data-ready="true"]')).not.toBeNull());
    return utils;
  }

  it('starts with neither, as the starting resume carries no contact details', async () => {
    await renderApp();
    expect(screen.getByLabelText('Email')).toHaveValue('');
    expect(screen.getByLabelText('Phone')).toHaveValue('');
    expect(document.querySelector('article.page a[href^="mailto:"], article.page a[href^="tel:"]')).toBeNull();
  });

  it('shows an address and a number on the page as you type them, and saves them', async () => {
    await renderApp();
    await fireEvent.input(screen.getByLabelText('Email'), { target: { value: EMAIL } });
    await fireEvent.input(screen.getByLabelText('Phone'), { target: { value: PHONE } });
    await waitFor(() => expect(document.querySelector(`article.page a[href="mailto:${EMAIL}"]`)).not.toBeNull());
    expect(document.querySelector(`article.page a[href="${PHONE_HREF}"]`)).toHaveTextContent(PHONE);

    window.dispatchEvent(new Event('pagehide'));
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.doc.basics.email).toBe(EMAIL);
    expect(saved.doc.basics.phone).toBe(PHONE);
  });

  it('keeps an invalid address in the box with an error and leaves the page and the saved resume alone', async () => {
    await renderApp();
    const email = screen.getByLabelText('Email');
    await fireEvent.input(email, { target: { value: 'ada@example.com?subject=hi' } });
    expect(email).toHaveValue('ada@example.com?subject=hi');
    expect(email).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent('Use a valid email address.');
    expect(document.querySelector('article.page a[href^="mailto:"]')).toBeNull();

    await fireEvent.input(email, { target: { value: EMAIL } });
    expect(screen.queryByRole('alert')).toBeNull();
    await waitFor(() => expect(document.querySelector(`article.page a[href="mailto:${EMAIL}"]`)).not.toBeNull());
  });

  it('refuses a phone number with letters or a scheme', async () => {
    await renderApp();
    const phone = screen.getByLabelText('Phone');
    await fireEvent.input(phone, { target: { value: 'tel:+15551234567' } });
    expect(phone).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('alert')).toHaveTextContent('Use digits, spaces, +, -, (), or .');
    expect(document.querySelector('article.page a[href^="tel:"]')).toBeNull();
  });
});

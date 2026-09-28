import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createApp } from '../src/server.js';

let dir;
let dataFile;
let app;

beforeEach(async () => {
  dir = await mkdtemp(path.join(os.tmpdir(), 'rsvp-server-test-'));
  dataFile = path.join(dir, 'rsvps.json');
  app = createApp({ dataFile });
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

function post(body) {
  return app.request('/api/rsvps', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

describe('GET /api/rsvps', () => {
  it('returns an empty array before any RSVP is submitted', async () => {
    const res = await app.request('/api/rsvps');
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual([]);
  });
});

describe('POST /api/rsvps', () => {
  it('accepts a valid RSVP, persists it, and stamps a submittedAt time', async () => {
    const res = await post({
      name: 'Ada Lovelace',
      email: 'ada@example.com',
      attending: 'yes',
      guestCount: 1,
      message: 'Excited!',
    });

    expect(res.status).toBe(201);
    const body = await res.json();
    expect(body.name).toBe('Ada Lovelace');
    expect(typeof body.submittedAt).toBe('string');
    expect(Number.isNaN(Date.parse(body.submittedAt))).toBe(false);

    const list = await (await app.request('/api/rsvps')).json();
    expect(list).toHaveLength(1);
  });

  it('rejects an RSVP with an invalid email and does not persist it', async () => {
    const res = await post({
      name: 'Bad Email',
      email: 'not-an-email',
      attending: 'yes',
      guestCount: 0,
    });

    expect(res.status).toBe(400);
    expect(await (await app.request('/api/rsvps')).json()).toEqual([]);
  });

  it('rejects a malformed JSON body', async () => {
    const res = await app.request('/api/rsvps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{not json',
    });
    expect(res.status).toBe(400);
  });

  it('rejects an invalid attending value', async () => {
    const res = await post({
      name: 'Undecided',
      email: 'x@example.com',
      attending: 'definitely',
      guestCount: 0,
    });
    expect(res.status).toBe(400);
  });

  it('stores a script-injecting name verbatim as data, never as executable HTML', async () => {
    // The API returns JSON, and the frontend (public/app.js) renders every
    // field via textContent - so the defense against XSS lives at render
    // time, not storage time. This test locks in that the server itself
    // never does any HTML templating of user input.
    const malicious = '<img src=x onerror=alert(1)>';
    const res = await post({
      name: malicious,
      email: 'x@example.com',
      attending: 'maybe',
      guestCount: 0,
    });

    expect(res.status).toBe(201);
    const body = await res.json();
    expect(body.name).toBe(malicious);
    expect(res.headers.get('content-type')).toContain('application/json');
  });
});

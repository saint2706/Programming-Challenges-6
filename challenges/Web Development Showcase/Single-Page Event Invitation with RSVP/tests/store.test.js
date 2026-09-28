import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { appendRsvp, readRsvps } from '../src/store.js';

let dir;
let dataFile;

beforeEach(async () => {
  dir = await mkdtemp(path.join(os.tmpdir(), 'rsvp-store-test-'));
  dataFile = path.join(dir, 'nested', 'rsvps.json');
});

afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

describe('readRsvps', () => {
  it('returns an empty array when the data file does not exist yet', async () => {
    expect(await readRsvps(dataFile)).toEqual([]);
  });
});

describe('appendRsvp', () => {
  it('creates the parent directory and writes the first RSVP', async () => {
    const rsvp = { name: 'Grace Hopper', attending: 'yes' };
    await appendRsvp(rsvp, dataFile);

    const onDisk = JSON.parse(await readFile(dataFile, 'utf-8'));
    expect(onDisk).toEqual([rsvp]);
  });

  it('appends to existing RSVPs without dropping prior entries', async () => {
    await appendRsvp({ name: 'Grace Hopper' }, dataFile);
    await appendRsvp({ name: 'Margaret Hamilton' }, dataFile);

    const all = await readRsvps(dataFile);
    expect(all.map((r) => r.name)).toEqual(['Grace Hopper', 'Margaret Hamilton']);
  });

  it('leaves no leftover temp files after a successful write', async () => {
    await appendRsvp({ name: 'Katherine Johnson' }, dataFile);

    const entries = await readdir(path.dirname(dataFile));
    expect(entries).toEqual(['rsvps.json']);
  });
});

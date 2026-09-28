import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_DATA_FILE = path.join(__dirname, '..', 'data', 'rsvps.json');

/** Reads all stored RSVPs. Returns [] if the data file doesn't exist yet. */
export async function readRsvps(dataFile = DEFAULT_DATA_FILE) {
  try {
    const raw = await readFile(dataFile, 'utf-8');
    return JSON.parse(raw);
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err;
  }
}

/**
 * Appends one RSVP and persists the full list atomically: write to a
 * uniquely-named temp file in the same directory, then rename() over the
 * real path. rename() is atomic on the same filesystem, so a crash or
 * concurrent read can never observe a partially-written data file - it
 * either sees the old complete file or the new complete file, never a
 * truncated/corrupt one.
 */
export async function appendRsvp(rsvp, dataFile = DEFAULT_DATA_FILE) {
  const dir = path.dirname(dataFile);
  await mkdir(dir, { recursive: true });

  const existing = await readRsvps(dataFile);
  existing.push(rsvp);

  const tmpFile = path.join(dir, `.${path.basename(dataFile)}.${randomUUID()}.tmp`);
  await writeFile(tmpFile, JSON.stringify(existing, null, 2), 'utf-8');
  await rename(tmpFile, dataFile);

  return existing;
}

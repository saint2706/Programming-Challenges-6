# Single-Page Event Invitation with RSVP

**Category:** Web Development Showcase
**Difficulty:** B

**Status:** Implemented (JavaScript)

A static event invitation page backed by a real "serverless-shaped" form
handler: submitting the RSVP form calls a small [Hono](https://hono.dev)
endpoint that validates the submission, persists it to a local JSON file,
and updates a live "who's coming" guest list — no third-party forms
service, no client-side-only fake.

## What it does

1. **Invitation page** (`public/index.html`) — event details plus an RSVP
   form (name, email, attending yes/no/maybe, extra guest count, optional
   message).
2. **RSVP API** (`src/server.js`) — `GET /api/rsvps` returns the guest
   list, `POST /api/rsvps` validates and stores a new RSVP.
3. **Live guest list** (`public/app.js`) — re-fetches and re-renders after
   every successful submission, entirely via `textContent`/DOM APIs.
4. **Persistence** (`src/store.js`) — RSVPs live in `data/rsvps.json`
   (gitignored — it's runtime data, not source), written atomically.

## Design notes

**Why Hono instead of Express:** the brief asks for a "serverless form
handler," and Hono's `app.fetch(request) => Promise<Response>` signature
*is* the standard shape a serverless platform expects — it's the same
interface Vercel Edge Functions, Cloudflare Workers, and AWS Lambda (via
`hono/aws-lambda`) call directly. `createApp()` in `src/server.js` builds
that handler; `src/dev-server.js` is the only piece that's specific to
running it locally, adapting it to a real Node socket via
`@hono/node-server` for `npm run dev`. Deploying this for real would mean
pointing a platform's function runtime at the exported `app.fetch` —
no handler code changes — and swapping `store.js`'s JSON file for that
platform's KV/database binding. An Express app couldn't demonstrate this;
its handler shape (`(req, res, next)`, mutated response object) has no
serverless equivalent without an adapter shim.

**Atomic writes, not just "write the file":** `appendRsvp` never writes
directly to `data/rsvps.json`. It writes the full updated list to a
uniquely-named temp file in the same directory, then `rename()`s it over
the real path. `rename()` is atomic on the same filesystem, so a reader
(or a crash mid-write) can only ever see the complete old file or the
complete new file — never a truncated/corrupt one. **Known limitation:**
this protects against corruption, not against a lost update under true
concurrent writes — two POSTs that both read the list before either
renames will both build their update from the same base list, and the
second `rename()` wins, silently dropping the first RSVP. Fine for a
single local demo server handling one submission at a time; a real
multi-instance deployment needs a real database (or the KV/database swap
mentioned above) rather than a shared JSON file.

**XSS defense lives at render time, not storage time:** the server stores
and returns guest-supplied strings (name, message) verbatim as JSON — it
does no HTML escaping, because it never generates HTML. The frontend
(`public/app.js`) is what touches the DOM, and every guest field is
inserted via `textContent`/`createElement`, never `innerHTML` with an
interpolated string, so a name like `<img src=x onerror=alert(1)>` renders
as inert visible text, not a script. `tests/server.test.js` locks in that
the API itself does no templating of user input, and server-side
validation (`src/validate.js`) independently rejects malformed email
addresses, over-length fields, and out-of-range guest counts regardless of
what the client-side `required`/`maxlength` attributes allow through.

## Run it

```bash
cd "challenges/Web Development Showcase/Single-Page Event Invitation with RSVP"
npm install

# Start the local server (serves the static page + the RSVP API on the
# same origin at http://localhost:3000):
npm run dev

# Run the test suite (validation, atomic-write persistence, and the
# HTTP handler — all exercised directly via Hono's app.request(), no
# real network socket needed):
npm test
```

`data/rsvps.json` is created on first submission and is gitignored —
delete it (or the whole `data/` folder) to reset the guest list.

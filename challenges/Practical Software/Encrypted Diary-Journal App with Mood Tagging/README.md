# Encrypted Diary/Journal App with Mood Tagging

**Category:** Practical Software
**Difficulty:** B (brief: "Local AES encryption, per-entry mood tag, simple trend chart.")

**Status:** Implemented (Python)

A real terminal journal app: passphrase-gated, AES-256-GCM encrypted at
rest, with a mood tag on every entry and an in-terminal mood-trend
sparkline. Built as a [Textual](https://textual.textualize.io/) TUI so it
feels like a standalone app you'd actually keep open, not "yet another
local web app."

## The real design tension: scrypt + AEAD, and a sentinel that can't lie

Two things had to be right, not just present:

**Key derivation.** The encryption key comes from the user's passphrase via
`scrypt`, not PBKDF2. PBKDF2's cost is pure CPU iteration count, which GPUs
and ASICs parallelize cheaply; scrypt is memory-hard, so brute-forcing a
stolen journal file is meaningfully more expensive per guess even at the
same wall-clock derivation time. `diary.SCRYPT_N/R/P` are tuned for roughly
half a second on ordinary hardware — expensive enough to matter against
brute force, cheap enough that unlocking your own diary isn't annoying.

**Authenticated encryption, not just confidentiality.** The journal is
encrypted with AES-256-**GCM** (an AEAD cipher), not plain AES-CBC. A wrong
passphrase or a tampered file both fail the same way — an
`InvalidTag`/`JournalUnlockError` — instead of CBC silently "decrypting" to
garbage bytes that might not even fail at the JSON-parsing stage. The
magic bytes are passed as GCM associated data, so even the file's format
marker is tamper-checked, not just the ciphertext.

**The mood-score sentinel bug that would've shipped unnoticed.** The
mood-trend sparkline needs a placeholder value for days with no entries
(Textual's `Sparkline` takes a plain `Sequence[float]`, no gap/`None`
support). The obvious choice — a valence scale from `-3` (angry) to `+3`
(happy) — makes `NEUTRAL` score exactly `0`, identical to the "no entries
that day" sentinel. A real neutral-mood day and an empty day would render
as the same flat point on the chart. `Mood.score` uses a `1..7` scale
instead specifically so `0` is never a valid mood score, and a real caught
test (`test_new_entry_appears_in_list_and_updates_trend`, which asserted
"the entry we just added shows up as a nonzero point") is what surfaced
this before it became a real "why is today missing" confusion in daily use.

## Design

- **`diary.py`** — all encryption, persistence, entry/mood/search/trend
  logic. Zero dependency on Textual, so every security-critical path is
  directly unit-testable without driving a terminal.
- **`app.py`** — the Textual UI only: an `UnlockScreen` modal (passphrase
  gate, doubles as first-run "create" flow), an `EntryEditorScreen` modal
  (mood picker + multi-line body), and a `MainScreen` with a searchable
  entry list, entry detail pane, and mood-trend sparkline + per-mood
  frequency counts.
- **Whole-journal encryption, not per-entry envelopes.** The entire entry
  list is serialized to one JSON blob and AES-GCM-encrypted as a single
  unit, re-encrypted in full on every save. At personal-diary scale
  (thousands of entries is still a tiny file) this is simpler to reason
  about than per-entry envelope encryption, and it means there's exactly
  one nonce/key pairing to get right, not one per entry. A fresh random
  96-bit nonce is drawn on *every* save — nonce reuse is catastrophic for
  GCM (it breaks both confidentiality and authentication), so a stable key
  reused across many saves only stays safe because the nonce never repeats.
- **Atomic writes.** `_atomic_write` writes the new ciphertext to a temp
  file in the same directory, `fsync`s it, then `os.replace`s it over the
  real journal file — atomic on both POSIX and Windows. A crash, killed
  process, or disk error mid-save leaves the original journal file exactly
  as it was; there's no window where it's half-written. Covered by a test
  that injects a real `OSError` mid-write and confirms the journal is both
  byte-identical to before *and* still fully openable afterward.
- **On-disk format:**
  `MAGIC(7) | VERSION(1) | SALT(16) | NONCE(12) | CIPHERTEXT+TAG(rest)`.
  The salt isn't secret (scrypt salts never are) and travels with the file
  so decryption is reproducible from the passphrase alone.
- **No "is this the right passphrase" oracle.** `JournalUnlockError` covers
  both a wrong passphrase and a corrupted/tampered file with the same
  exception and message — a caller can't distinguish "close, try again"
  from "this file is garbage" without actually being able to decrypt it,
  which needs the correct passphrase anyway.

## Usage

```bash
cd "challenges/Practical Software/Encrypted Diary-Journal App with Mood Tagging"

uv run --with textual --with cryptography python app.py
```

First run: there's no journal file yet, so the passphrase you enter
**creates** the journal (there is no recovery if you forget it — this is a
local encrypted file, not an account with a reset flow). Later runs: the
same screen unlocks the existing `journal.enc`.

Keybindings on the main screen: `n` new entry, `d` delete the highlighted
entry, `/` focus the search box, `Escape` clear the search. The search box
filters by full-text match against entry bodies; mood/date-range filtering
is available programmatically via `JournalStore.search()` but isn't wired
to a UI control yet (see below). The right pane shows a 30-day mood-trend
sparkline plus a per-mood frequency summary that update live as you add
entries.

```bash
uv run --with textual --with cryptography --with pytest --with pytest-asyncio pytest -q   # 28 tests
```

## Threat model / limitations

Protects the journal file at rest against someone with filesystem access
but no passphrase (a stolen laptop, a synced-but-not-trusted backup
destination, someone else's account on a shared machine). It does **not**
protect data once the app is unlocked and running — anyone with access to
that terminal session, or to process memory, can read it — and it is not a
substitute for full-disk encryption against an attacker with a live,
unlocked session. There is no passphrase-reset mechanism by design (there's
no server to reset it against); losing the passphrase means losing the
journal.

## What it deliberately doesn't do

- No date-range/mood filter controls in the UI (the search box is
  full-text only) — `JournalStore.search()` already supports mood and date
  filters for a future UI pass, but wiring up filter widgets wasn't core to
  the brief's "mood tagging + trend chart" ask.
- No entry editing after creation, only add/delete — keeps the
  append-and-prune model simple; a real edit feature would need to decide
  whether edits preserve the original timestamp for trend purposes.
- No multi-device sync — it's a single local encrypted file, on purpose.
- No auto-lock/session timeout — the app stays unlocked until you quit it.

## Tests

28 pytest cases across two files, all passing (`pytest -q`).
`test_diary.py` (23 cases, no terminal needed) covers: encrypt/decrypt
round trips, wrong-passphrase rejection, tampered-ciphertext rejection
(flips a byte in the GCM tag and confirms it's caught), garbage-data
rejection, nonce uniqueness across saves, the on-disk file never containing
a recognizable plaintext fragment, atomic writes leaving no leftover temp
file, a real injected `OSError` mid-write proving the original file
survives untouched, entry CRUD, text/mood/date-range search, and mood-trend
averaging (including the same-day-multiple-entries and gap-filling cases
that caught the sentinel-collision issue above). `test_app.py` (5 cases)
drives the real Textual app headlessly via `Pilot`: it starts on the
unlock screen, a fresh passphrase both creates and unlocks a new journal,
a wrong passphrase against an existing journal shows the error and stays
on the unlock screen, adding an entry through the real UI makes it appear
in the list and updates the trend sparkline, and the search box correctly
filters the entry list. A separate manual check (not part of the suite)
decrypted a real on-disk journal with the correct passphrase, confirmed a
wrong passphrase is rejected, and confirmed a known plaintext string never
appears in the raw file bytes.

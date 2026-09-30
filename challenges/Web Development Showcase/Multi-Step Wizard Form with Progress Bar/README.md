# Multi-Step Wizard Form with Progress Bar

**Category:** Web Development Showcase
**Difficulty:** I

**Status:** Implemented (JavaScript/Svelte)

A five-step sign-up wizard (Account, Profile, Preferences, Plan, Review) with
client-side validation on every step, a progress bar, and a draft that
survives a page reload, without ever writing a password or card detail to
disk.

## What it does

- **Per-step validation:** Next validates only the current step. On failure the
  step does not advance, an error summary appears and takes focus, each
  message is a button that jumps to its field, and every invalid control gets
  `aria-invalid` plus `aria-describedby` wired to its hint and error text.
  After a field is blurred once (or a failed Next), it revalidates live as you
  type.
- **Cross-field and conditional rules:** confirm-password match, age from the
  birth date (real calendar dates only, at least 13), newsletter frequency only
  when the newsletter box is ticked, and a Plan step that changes shape: Team
  asks for company and seats (2 to 50), any paid plan asks for a Luhn-checked
  card number, an unexpired `MM/YY`, and a CVC.
- **Progress bar and step list:** `role="progressbar"` with
  `aria-valuenow`/`aria-valuetext` ("Step 2 of 5: Profile. 20% complete"), plus
  an ordered step list with `aria-current="step"`, "(completed)" for
  screen readers, and steps you have not earned yet disabled. Progress counts
  verified steps, not the current position.
- **No skipping ahead:** a step is reachable only if every earlier step is
  verified. That holds for the step list, the URL hash (`#/step/4` on a fresh
  load lands on step 1), and browser Back/Forward. Going back keeps all data;
  editing a verified step voids its verification, so progress can drop.
- **Focus and announcements:** moving to a new step focuses its heading; a
  failed Next focuses the error summary instead; a polite live region says
  "Step N of 5: Title". Animations are switched off under
  `prefers-reduced-motion`.
- **Review step:** a read-only summary with an Edit link per section, a terms
  checkbox, and Submit. Submit revalidates *all* steps.
- **Resumable draft:** every change is saved to `localStorage`; reload and you
  are back on the same step with the same progress and a "Welcome back"
  notice (Dismiss / Discard draft). Submitting clears the draft.

## Design notes

**Why Svelte 5 runes.** The wizard is a single reactive object: form data,
which steps are verified, current errors, which fields have been touched.
That maps directly onto a class with `$state` fields
(`src/lib/wizard.svelte.js`), so the whole state machine is one testable unit
that components read from and write to, with no store plumbing and no context.
`$state.snapshot` gives plain data for zod and for persistence. Components stay
thin: `Field` owns the label/hint/error wiring once, `TextField` owns the input
plumbing, `Steps` is the per-step markup.

**Why zod.** Each step is a zod schema (`src/lib/schemas.js`), so the rules are
declarative, cross-field checks live next to the fields they concern
(`superRefine`), and the same `validateStep(i, data)` serves Next, live
revalidation, and the final all-steps check on Submit. The one helper that
flattens issues keeps the *first* message per field.

**Verified steps are the source of truth for navigation.** `navigation.js` is
pure: `furthestReachable(verified)` is the first unverified step, and every
route into a step (button, hash, Back) goes through `clampStep`. There is no
separate "how far have I got" counter to drift out of sync. Because Next moves
to the furthest reachable step, not just step + 1, a user who fixes one earlier
step is taken straight to the next step that still needs work instead of
clicking through steps that were already fine.

**Persistence is a security boundary, not a convenience.**
`SENSITIVE_FIELDS` (`password`, `confirmPassword`, `cardNumber`, `expiry`,
`cvc`) are stripped in `saveDraft` and refused again in `loadDraft`, so even a
hand-edited draft cannot inject them. `loadDraft` trusts nothing: it checks a
schema version (`wizard-draft:v1`), drops unknown keys, ignores any value whose
type differs from the default, clamps the step, and validates the verified
list. Unparseable JSON, a wrong version, or hostile shapes all fall back to a
fresh wizard. If `localStorage` throws (private mode, quota), the wizard just
runs without resume.

**The honest cost of not storing secrets.** A resumed draft can show a step as
"completed" even though its password or card is gone. That is why Submit
revalidates every step: it sends the user to the first failing step, un-verifies
only the steps that actually failed (the rest keep their progress), and the
Review step says "password not saved, re-enter to submit" up front rather than
surprising them on Submit.

**Hash routing without an SPA router.** The step lives in `#/step/N`, so Back
and Forward and deep links work. The first sync uses `replaceState` (no phantom
history entry), and incoming hash changes are clamped the same way as clicks;
a rejected hash is rewritten to the step you actually landed on.

**All rendering is `textContent`-safe.** Values are only ever bound through
Svelte's text interpolation and input `value`, never `{@html}`, so the review
summary and the confirmation message cannot inject markup from user input.

## Run it

```bash
cd "challenges/Web Development Showcase/Multi-Step Wizard Form with Progress Bar"
npm install

# Dev server (http://localhost:5173):
npm run dev

# Production build (outputs to dist/, gitignored):
npm run build

# Run the test suite (64 tests):
npm test
```

Try card `4242 4242 4242 4242` with any future expiry and a 3-digit CVC.

## Tests

64 tests across five files, all offline (jsdom):

| File                   | Covers                                                                                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schemas.test.js`      | Every rule per step, Luhn, real-date and age checks, expiry boundary, conditional Team/paid-plan fields                                                 |
| `navigation.test.js`   | `furthestReachable`, `clampStep`, hash parse/format, progress percentage                                                                                |
| `storage.test.js`      | Secrets never written or read back, versioning, corrupted JSON, hostile shapes, blocked storage                                                         |
| `wizard.test.js`       | The state machine: blocked Next, verification, edit-voids-verification, clamping, submit-after-resume, skipping ahead over still-verified steps          |
| `app.test.js`          | Real components with `@testing-library/svelte`: progressbar ARIA, error summary focus, disabled future steps, Back keeps data, full flow, resume/discard |

## Limitations

- **No real backend.** Submit is simulated; nothing leaves the browser.
- **Draft is per browser profile,** not per account, and is plain
  `localStorage` (not encrypted). That is fine for non-secret fields, which is
  exactly why the secret ones are excluded.
- **Progress can over-count after a resume,** since secrets are not restored;
  Submit corrects it (see above) rather than the bar hiding the fact.
- **Country list is a short fixed sample,** and the phone check is a loose
  format check, not carrier-grade validation.
- **Card validation is client-side theatre:** Luhn and expiry only, no
  network authorization.

<script>
  import { tick, untrack } from 'svelte';
  import ProgressBar from './components/ProgressBar.svelte';
  import ErrorSummary from './components/ErrorSummary.svelte';
  import Steps from './components/Steps.svelte';
  import { Wizard } from './lib/wizard.svelte.js';
  import { STEPS } from './lib/schemas.js';
  import { STEP_COUNT, formatHash, parseHash } from './lib/navigation.js';
  import { clearDraft, loadDraft, saveDraft } from './lib/storage.js';

  const store = (() => {
    try {
      return window.localStorage;
    } catch {
      return null; // blocked storage: run without resume
    }
  })();

  const draft = store ? loadDraft(store) : null;
  const w = new Wizard(draft);
  const requested = parseHash(location.hash);
  if (requested !== null) w.go(requested); // a deep link wins over the saved step, but is still clamped

  const resumed = !!draft && (draft.verified.length > 0 || draft.data.email !== '');
  let dismissedResume = $state(false);

  const labels = {
    email: 'Email', username: 'Username', password: 'Password', confirmPassword: 'Confirm password',
    fullName: 'Full name', birthDate: 'Date of birth', country: 'Country', phone: 'Phone',
    interests: 'Interests', newsletter: 'Newsletter', frequency: 'Frequency', plan: 'Plan',
    company: 'Company', seats: 'Seats', cardNumber: 'Card number', expiry: 'Expiry', cvc: 'CVC',
    acceptTerms: 'Terms',
  };

  let heading = $state();
  let summary = $state();
  let firstRun = true;

  // Keep the URL hash in step with the wizard. First sync replaces (no phantom history entry).
  $effect(() => {
    const target = formatHash(w.step);
    if (location.hash === target) return;
    if (firstRun) history.replaceState(null, '', target);
    else location.hash = target;
  });

  // Back/forward and manual hash edits go through the same clamp as everything else.
  function onhashchange() {
    const n = parseHash(location.hash);
    if (n === null) {
      history.replaceState(null, '', formatHash(w.step));
      return;
    }
    const landed = w.go(n);
    if (landed !== n) history.replaceState(null, '', formatHash(landed));
  }

  // Persist on every change (sensitive fields are stripped inside saveDraft).
  $effect(() => {
    const snap = w.snapshot();
    if (!store) return;
    if (w.submitted) clearDraft(store);
    else saveDraft(store, snap);
  });

  // Move focus to the new step's heading (unless errors want the summary to have it).
  $effect(() => {
    w.step;
    if (firstRun) return;
    untrack(() => {
      if (Object.keys(w.errors).length === 0) tick().then(() => heading?.focus());
    });
  });

  $effect(() => {
    firstRun = false;
  });

  async function focusSummary() {
    await tick();
    summary?.focus();
  }

  async function onsubmit(e) {
    e.preventDefault();
    if (w.isLast) {
      const result = w.submit();
      if (!result.ok) await focusSummary();
      return;
    }
    if (!w.next()) await focusSummary();
  }

  function startOver() {
    if (store) clearDraft(store);
    w.reset();
  }

  const announce = $derived(`Step ${w.step + 1} of ${STEP_COUNT}: ${w.current.title}`);
</script>

<main>
  <header>
    <h1>Create your account</h1>
    <p class="lede">Five short steps. Your progress is saved on this device (never passwords or card details).</p>
  </header>

  {#if w.submitted}
    <section class="success" aria-labelledby="done-title">
      <h2 id="done-title" tabindex="-1">You're all set, {w.data.fullName}!</h2>
      <p>Account <strong>{w.data.username}</strong> was created (simulated, nothing left your browser).</p>
      <button type="button" onclick={startOver}>Start over</button>
    </section>
  {:else}
    <ProgressBar step={w.step} verified={w.verified} maxStep={w.maxStep} percent={w.percent} onselect={(i) => w.go(i)} />

    {#if resumed && !dismissedResume}
      <p class="notice">
        Welcome back, your draft was restored. Passwords and card details are never stored, so re-enter them if asked.
        <button type="button" class="link" onclick={() => (dismissedResume = true)}>Dismiss</button>
        <button type="button" class="link" onclick={startOver}>Discard draft</button>
      </p>
    {/if}

    <div class="sr-only" aria-live="polite">{announce}</div>

    <form novalidate {onsubmit} aria-labelledby="step-title">
      <h2 id="step-title" tabindex="-1" bind:this={heading}>{w.current.title}</h2>
      <ErrorSummary errors={w.errors} {labels} bind:el={summary} />
      <Steps {w} />
      <div class="actions">
        <button type="button" class="secondary" disabled={w.step === 0} onclick={() => w.back()}>Back</button>
        <button type="submit">{w.isLast ? 'Submit' : 'Next'}</button>
      </div>
    </form>
  {/if}
</main>

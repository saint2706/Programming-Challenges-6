<script>
  import Field from './Field.svelte';
  import TextField from './TextField.svelte';
  import { INTERESTS, FREQUENCIES, STEPS } from '../lib/schemas.js';

  let { w } = $props();

  const COUNTRIES = ['Australia', 'Brazil', 'Canada', 'France', 'Germany', 'India', 'Japan', 'United Kingdom', 'United States'];
  const PLAN_INFO = {
    free: { name: 'Free', price: '$0', blurb: 'Personal projects, no card needed' },
    pro: { name: 'Pro', price: '$12/mo', blurb: 'Everything in Free plus priority support' },
    team: { name: 'Team', price: '$9/seat/mo', blurb: 'Shared workspace for 2-50 people' },
  };
  const INTEREST_LABEL = { frontend: 'Frontend', backend: 'Backend', data: 'Data', ml: 'Machine learning', design: 'Design' };
  const EDITABLE = STEPS.slice(0, -1);

  function toggleInterest(name, checked) {
    const set = new Set(w.data.interests);
    if (checked) set.add(name);
    else set.delete(name);
    w.data.interests = INTERESTS.filter((i) => set.has(i));
    w.edit('interests');
    w.touch('interests');
  }

  const cardLast4 = $derived(w.data.cardNumber.replace(/\D/g, '').slice(-4));
</script>

{#if w.step === 0}
  <TextField {w} name="email" label="Email" type="email" autocomplete="email" />
  <TextField {w} name="username" label="Username" autocomplete="username" hint="3-20 characters: letters, numbers, underscores." />
  <TextField {w} name="password" label="Password" type="password" autocomplete="new-password" hint="At least 10 characters with a letter and a number. Never saved to this device." />
  <TextField {w} name="confirmPassword" label="Confirm password" type="password" autocomplete="new-password" />
{:else if w.step === 1}
  <TextField {w} name="fullName" label="Full name" autocomplete="name" />
  <TextField {w} name="birthDate" label="Date of birth" type="date" autocomplete="bday" max={new Date().toISOString().slice(0, 10)} />
  <Field name="country" label="Country" error={w.errors.country}>
    {#snippet children({ id, describedby, invalid })}
      <select
        {id}
        name="country"
        value={w.data.country}
        aria-describedby={describedby}
        aria-invalid={invalid ? 'true' : undefined}
        aria-required="true"
        onchange={(e) => {
          w.data.country = e.currentTarget.value;
          w.edit('country');
        }}
        onblur={() => w.touch('country')}
      >
        <option value="">Select a country</option>
        {#each COUNTRIES as c}<option value={c}>{c}</option>{/each}
      </select>
    {/snippet}
  </Field>
  <TextField {w} name="phone" label="Phone" type="tel" optional autocomplete="tel" />
{:else if w.step === 2}
  <fieldset class:invalid={!!w.errors.interests} aria-describedby={w.errors.interests ? 'f-interests-err' : undefined}>
    <legend>Interests</legend>
    <div class="choices">
      {#each INTERESTS as name, i (name)}
        <label class="choice">
          <input
            type="checkbox"
            id={i === 0 ? 'f-interests' : undefined}
            checked={w.data.interests.includes(name)}
            aria-invalid={w.errors.interests ? 'true' : undefined}
            onchange={(e) => toggleInterest(name, e.currentTarget.checked)}
          />
          {INTEREST_LABEL[name] ?? name}
        </label>
      {/each}
    </div>
    {#if w.errors.interests}<p class="error" id="f-interests-err">{w.errors.interests}</p>{/if}
  </fieldset>

  <label class="choice standalone">
    <input
      type="checkbox"
      id="f-newsletter"
      checked={w.data.newsletter}
      onchange={(e) => {
        w.data.newsletter = e.currentTarget.checked;
        if (!e.currentTarget.checked) w.data.frequency = '';
        w.edit('newsletter');
      }}
    />
    Send me the newsletter
  </label>

  {#if w.data.newsletter}
    <Field name="frequency" label="How often?" error={w.errors.frequency}>
      {#snippet children({ id, describedby, invalid })}
        <select
          {id}
          name="frequency"
          value={w.data.frequency}
          aria-describedby={describedby}
          aria-invalid={invalid ? 'true' : undefined}
          aria-required="true"
          onchange={(e) => {
            w.data.frequency = e.currentTarget.value;
            w.edit('frequency');
          }}
          onblur={() => w.touch('frequency')}
        >
          <option value="">Choose…</option>
          {#each FREQUENCIES as f}<option value={f}>{f}</option>{/each}
        </select>
      {/snippet}
    </Field>
  {/if}
{:else if w.step === 3}
  <fieldset class:invalid={!!w.errors.plan}>
    <legend>Plan</legend>
    <div class="plans">
      {#each Object.entries(PLAN_INFO) as [key, info], i (key)}
        <label class="plan" class:selected={w.data.plan === key}>
          <input
            type="radio"
            name="plan"
            id={i === 0 ? 'f-plan' : undefined}
            value={key}
            checked={w.data.plan === key}
            onchange={() => {
              w.data.plan = key;
              w.edit('plan');
            }}
          />
          <span class="plan-name">{info.name}</span>
          <span class="plan-price">{info.price}</span>
          <span class="plan-blurb">{info.blurb}</span>
        </label>
      {/each}
    </div>
    {#if w.errors.plan}<p class="error">{w.errors.plan}</p>{/if}
  </fieldset>

  {#if w.data.plan === 'team'}
    <TextField {w} name="company" label="Company" autocomplete="organization" />
    <TextField {w} name="seats" label="Seats" type="text" inputmode="numeric" hint="2 to 50 people." />
  {/if}
  {#if w.data.plan !== 'free'}
    <TextField {w} name="cardNumber" label="Card number" inputmode="numeric" autocomplete="cc-number" hint="Try 4242 4242 4242 4242. Never saved to this device." />
    <div class="row">
      <TextField {w} name="expiry" label="Expiry (MM/YY)" autocomplete="cc-exp" placeholder="MM/YY" />
      <TextField {w} name="cvc" label="CVC" inputmode="numeric" autocomplete="cc-csc" />
    </div>
  {/if}
{:else}
  <dl class="review">
    {#each EDITABLE as s, i (s.id)}
      <div class="review-group">
        <dt>
          {s.title}
          <button type="button" class="link" onclick={() => w.go(i)} aria-label={`Edit ${s.title}`}>Edit</button>
        </dt>
        <dd>
          {#if i === 0}
            {w.data.username} · {w.data.email} · password {w.data.password ? '•'.repeat(8) : 'not saved, re-enter to submit'}
          {:else if i === 1}
            {w.data.fullName}, born {w.data.birthDate}, {w.data.country}{w.data.phone ? `, ${w.data.phone}` : ''}
          {:else if i === 2}
            {w.data.interests.map((i) => INTEREST_LABEL[i] ?? i).join(', ')} · newsletter {w.data.newsletter ? w.data.frequency : 'off'}
          {:else}
            {PLAN_INFO[w.data.plan].name}{w.data.plan === 'team' ? `, ${w.data.seats} seats for ${w.data.company}` : ''}
            {#if w.data.plan !== 'free'}
              · card {cardLast4 ? `ending ${cardLast4}` : 'not saved, re-enter to submit'}
            {/if}
          {/if}
        </dd>
      </div>
    {/each}
  </dl>

  <label class="choice standalone" class:invalid={!!w.errors.acceptTerms}>
    <input
      type="checkbox"
      id="f-acceptTerms"
      checked={w.data.acceptTerms}
      aria-invalid={w.errors.acceptTerms ? 'true' : undefined}
      aria-describedby={w.errors.acceptTerms ? 'f-acceptTerms-err' : undefined}
      onchange={(e) => {
        w.data.acceptTerms = e.currentTarget.checked;
        w.edit('acceptTerms');
        w.touch('acceptTerms');
      }}
    />
    I accept the terms of service
  </label>
  {#if w.errors.acceptTerms}<p class="error" id="f-acceptTerms-err">{w.errors.acceptTerms}</p>{/if}
{/if}

<script>
  /** Name, headline, website, location, summary and profile links. */
  import { announce } from '../lib/announcer.svelte.js';
  import { setPath } from '../lib/model.js';
  import { LIMITS } from '../lib/schema.js';
  import Field from './Field.svelte';

  let { basics, apply } = $props();

  const set = (path, value) => apply((doc) => setPath(doc, path, value));

  function addProfile() {
    apply((doc) => setPath(doc, ['basics', 'profiles'], [...doc.basics.profiles, { network: '', username: '', url: '' }]));
    announce('Profile link added.');
  }

  function removeProfile(i) {
    apply((doc) => setPath(doc, ['basics', 'profiles'], doc.basics.profiles.filter((_, j) => j !== i)));
    announce('Profile link removed.');
  }
</script>

<div class="form-grid">
  <Field label="Name" value={basics.name} oncommit={(v) => set(['basics', 'name'], v)} />
  <Field label="Headline" value={basics.label} oncommit={(v) => set(['basics', 'label'], v)} />
  <Field label="Email" type="email" autocomplete="email" value={basics.email} maxlength={254} oncommit={(v) => set(['basics', 'email'], v)} />
  <Field label="Phone" type="tel" autocomplete="tel" value={basics.phone} maxlength={20} oncommit={(v) => set(['basics', 'phone'], v)} />
  <Field label="Website" type="url" value={basics.url} oncommit={(v) => set(['basics', 'url'], v)} placeholder="https://" />
  <Field label="City" value={basics.location.city} maxlength={100} oncommit={(v) => set(['basics', 'location', 'city'], v)} />
  <Field label="Region" value={basics.location.region} maxlength={100} oncommit={(v) => set(['basics', 'location', 'region'], v)} />
  <Field label="Country code" value={basics.location.countryCode} maxlength={10} oncommit={(v) => set(['basics', 'location', 'countryCode'], v)} />
</div>
<Field label="Summary" type="textarea" rows={5} maxlength={LIMITS.long} value={basics.summary} oncommit={(v) => set(['basics', 'summary'], v)} />

<fieldset class="entry">
  <legend>Profile links</legend>
  {#each basics.profiles as profile, i (i)}
    <div class="profile-row">
      <Field label="Network" value={profile.network} maxlength={100} oncommit={(v) => set(['basics', 'profiles', i, 'network'], v)} />
      <Field label="Link" type="url" value={profile.url} placeholder="https://" oncommit={(v) => set(['basics', 'profiles', i, 'url'], v)} />
      <button type="button" class="icon-button danger" aria-label="Remove profile link {i + 1}" onclick={() => removeProfile(i)}>Remove</button>
    </div>
  {/each}
  <button type="button" class="button small" disabled={basics.profiles.length >= LIMITS.entries} onclick={addProfile}>Add profile link</button>
</fieldset>

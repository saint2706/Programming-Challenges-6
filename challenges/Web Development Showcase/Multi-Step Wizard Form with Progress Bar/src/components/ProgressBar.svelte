<script>
  import { STEPS } from '../lib/schemas.js';

  let { step, verified, maxStep, percent, onselect } = $props();
</script>

<nav aria-label="Wizard progress" class="progress">
  <div
    class="bar"
    role="progressbar"
    aria-label="Form completion"
    aria-valuemin="0"
    aria-valuemax="100"
    aria-valuenow={percent}
    aria-valuetext={`Step ${step + 1} of ${STEPS.length}: ${STEPS[step].title}. ${percent}% complete`}
  >
    <div class="fill" style:width="{percent}%"></div>
  </div>

  <ol class="steps">
    {#each STEPS as s, i (s.id)}
      {@const done = verified.includes(i)}
      <li aria-current={i === step ? 'step' : undefined} class:current={i === step} class:done>
        <button type="button" disabled={i > maxStep} onclick={() => onselect(i)}>
          <span class="dot" aria-hidden="true">{done && i !== step ? '✓' : i + 1}</span>
          <span class="label">{s.title}</span>
          {#if done}<span class="sr-only"> (completed)</span>{/if}
        </button>
      </li>
    {/each}
  </ol>
</nav>

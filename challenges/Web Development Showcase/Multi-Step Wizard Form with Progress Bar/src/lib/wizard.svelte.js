import { STEPS, defaultData, validateStep } from './schemas.js';
import { STEP_COUNT, clampStep, furthestReachable, progressPercent } from './navigation.js';

/** Reactive wizard controller: form data, per-step verification, errors and navigation. */
export class Wizard {
  data = $state(defaultData());
  step = $state(0);
  verified = $state([]);
  errors = $state({});
  touched = $state({});
  submitted = $state(false);

  constructor(draft = null) {
    if (draft) {
      Object.assign(this.data, draft.data);
      this.verified = [...draft.verified];
      this.step = clampStep(draft.step, this.verified);
    }
  }

  get current() {
    return STEPS[this.step];
  }
  get isLast() {
    return this.step === STEP_COUNT - 1;
  }
  get maxStep() {
    return furthestReachable(this.verified);
  }
  get percent() {
    return progressPercent(this.verified);
  }

  snapshot() {
    return { data: $state.snapshot(this.data), step: this.step, verified: [...this.verified] };
  }

  /** Move to a step, clamped so nobody can skip an unverified one. Returns the step landed on. */
  go(index) {
    const target = clampStep(index, this.verified);
    if (target !== this.step) {
      this.step = target;
      this.errors = {};
      this.touched = {};
    }
    return this.step;
  }

  back() {
    return this.go(this.step - 1);
  }

  #revalidate(names) {
    const { errors } = validateStep(this.step, $state.snapshot(this.data));
    const next = { ...this.errors };
    for (const name of names) {
      if (name in errors) next[name] = errors[name];
      else delete next[name];
    }
    this.errors = next;
  }

  /** Called on every edit: the step's verification is void; refresh errors already shown. */
  edit(name) {
    this.verified = this.verified.filter((i) => i !== this.step);
    if (this.touched[name] || name in this.errors) this.#revalidate(this.current.fields);
  }

  /** Called on blur: from now on this field reports errors live. */
  touch(name) {
    this.touched = { ...this.touched, [name]: true };
    this.#revalidate([name]);
  }

  /** Validate the current step; on success mark it verified and advance. */
  next() {
    const { ok, errors } = validateStep(this.step, $state.snapshot(this.data));
    if (!ok) {
      this.errors = errors;
      this.touched = Object.fromEntries(this.current.fields.map((f) => [f, true]));
      return false;
    }
    if (!this.verified.includes(this.step)) this.verified = [...this.verified, this.step];
    this.errors = {};
    // Skip over steps that are still verified (e.g. after a failed submit un-verified only some).
    if (!this.isLast) this.go(Math.max(this.step + 1, this.maxStep));
    return true;
  }

  /**
   * Validate everything (a resumed draft has no card/password, so earlier "verified"
   * steps can fail here). Only the failing steps lose their verification; the user is
   * sent to the first of them, or the wizard is marked submitted.
   */
  submit() {
    const data = $state.snapshot(this.data);
    const failing = [];
    let first = null;
    for (let i = 0; i < STEP_COUNT; i++) {
      const { ok, errors } = validateStep(i, data);
      if (!ok) {
        failing.push(i);
        first ??= { i, errors };
      }
    }
    if (first) {
      this.verified = this.verified.filter((v) => !failing.includes(v));
      this.step = first.i;
      this.errors = first.errors;
      this.touched = Object.fromEntries(STEPS[first.i].fields.map((f) => [f, true]));
      return { ok: false, failedStep: first.i };
    }
    this.verified = STEPS.map((_, i) => i);
    this.submitted = true;
    this.errors = {};
    return { ok: true };
  }

  reset() {
    Object.assign(this.data, defaultData());
    this.verified = [];
    this.step = 0;
    this.errors = {};
    this.touched = {};
    this.submitted = false;
  }
}

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/svelte';
import userEvent from '@testing-library/user-event';
import App from '../src/App.svelte';
import { STORAGE_KEY } from '../src/lib/storage.js';

const PASSWORD = 'correct horse 9';

function mountApp() {
  const user = userEvent.setup();
  const view = render(App);
  return { user, ...view };
}

const heading = () => screen.getByRole('heading', { level: 2 });
const next = (user) => user.click(screen.getByRole('button', { name: /^(next|submit)$/i }));

async function fillAccount(user) {
  await user.type(screen.getByLabelText(/^email/i), 'ada@example.com');
  await user.type(screen.getByLabelText(/^username/i), 'ada_l');
  await user.type(screen.getByLabelText(/^password/i), PASSWORD);
  await user.type(screen.getByLabelText(/confirm password/i), PASSWORD);
}

async function fillProfile(user) {
  await user.type(screen.getByLabelText(/full name/i), 'Ada Lovelace');
  await user.type(screen.getByLabelText(/date of birth/i), '1990-05-01');
  await user.selectOptions(screen.getByLabelText(/country/i), 'India');
}

async function reachReview(user) {
  await fillAccount(user);
  await next(user);
  await fillProfile(user);
  await next(user);
  await user.click(screen.getByLabelText('Data'));
  await next(user);
  await user.click(screen.getByLabelText(/^Free/));
  await next(user);
  await waitFor(() => expect(heading()).toHaveTextContent('Review'));
}

beforeEach(() => {
  cleanup();
  localStorage.clear();
  history.replaceState(null, '', '/');
});

describe('wizard UI', () => {
  it('exposes an accessible progress bar and marks the current step', () => {
    mountApp();
    const bar = screen.getByRole('progressbar');
    expect(bar).toHaveAttribute('aria-valuenow', '0');
    expect(bar).toHaveAttribute('aria-valuemin', '0');
    expect(bar).toHaveAttribute('aria-valuemax', '100');
    expect(bar.getAttribute('aria-valuetext')).toMatch(/Step 1 of 5: Account/);
    const current = document.querySelector('.steps li[aria-current="step"]');
    expect(current).toHaveTextContent('Account');
  });

  it('shows an error summary, focuses it, and does not advance on an invalid step', async () => {
    const { user } = mountApp();
    await next(user);
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/problems with this step/i);
    await waitFor(() => expect(alert).toHaveFocus());
    expect(heading()).toHaveTextContent('Account');
    expect(screen.getByLabelText(/^email/i)).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0');
  });

  it('error summary links move focus to the offending field', async () => {
    const { user } = mountApp();
    await next(user);
    await user.click(await screen.findByRole('button', { name: /^Email: / }));
    expect(screen.getByLabelText(/^email/i)).toHaveFocus();
  });

  it('advances on valid input, updates progress, and moves focus to the new heading', async () => {
    const { user } = mountApp();
    await fillAccount(user);
    await next(user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    await waitFor(() => expect(heading()).toHaveFocus());
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '20');
    expect(location.hash).toBe('#/step/2');
  });

  it('cannot skip ahead: later steps are disabled in the step list', async () => {
    const { user } = mountApp();
    const buttons = screen.getAllByRole('button', { name: /Account|Profile|Preferences|Plan|Review/ }).filter((b) => b.closest('.steps'));
    expect(buttons.map((b) => b.disabled)).toEqual([false, true, true, true, true]);
    await fillAccount(user);
    await next(user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    const after = screen.getAllByRole('button').filter((b) => b.closest('.steps'));
    expect(after.map((b) => b.disabled)).toEqual([false, false, true, true, true]);
  });

  it('keeps entered data when going back', async () => {
    const { user } = mountApp();
    await fillAccount(user);
    await next(user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    await user.click(screen.getByRole('button', { name: 'Back' }));
    await waitFor(() => expect(heading()).toHaveTextContent('Account'));
    expect(screen.getByLabelText(/^email/i)).toHaveValue('ada@example.com');
    expect(screen.getByLabelText(/^password/i)).toHaveValue(PASSWORD);
  });

  it('editing a verified step voids its verification (progress drops)', async () => {
    const { user } = mountApp();
    await fillAccount(user);
    await next(user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    await user.click(screen.getByRole('button', { name: 'Back' }));
    await waitFor(() => expect(heading()).toHaveTextContent('Account'));
    await user.type(screen.getByLabelText(/^username/i), 'x');
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0');
  });

  it('a deep link to an unreachable step is clamped to the first incomplete step', () => {
    history.replaceState(null, '', '/#/step/4');
    mountApp();
    expect(heading()).toHaveTextContent('Account');
    expect(location.hash).toBe('#/step/1');
  });

  it('completes the whole flow, shows the confirmation, and clears the draft', async () => {
    const { user } = mountApp();
    await reachReview(user);
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '80');
    await next(user);
    expect(await screen.findByRole('alert')).toHaveTextContent(/terms/i);
    await user.click(screen.getByLabelText(/accept the terms/i));
    await next(user);
    expect(await screen.findByText(/You're all set, Ada Lovelace/)).toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    await user.click(screen.getByRole('button', { name: 'Start over' }));
    expect(await screen.findByLabelText(/^email/i)).toHaveValue('');
  });
});

describe('resumable draft', () => {
  it('persists the draft but never passwords or card details', async () => {
    const { user } = mountApp();
    await fillAccount(user);
    await next(user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    const raw = localStorage.getItem(STORAGE_KEY);
    expect(raw).toBeTruthy();
    expect(raw).not.toContain(PASSWORD);
    const saved = JSON.parse(raw);
    expect(saved.data.email).toBe('ada@example.com');
    expect(saved.data).not.toHaveProperty('password');
    expect(saved.data).not.toHaveProperty('cardNumber');
    expect(saved.verified).toEqual([0]);
    expect(saved.step).toBe(1);
  });

  it('restores step, progress and fields after a remount, with the welcome-back notice', async () => {
    const first = mountApp();
    await fillAccount(first.user);
    await next(first.user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    await first.user.type(screen.getByLabelText(/full name/i), 'Ada');
    first.unmount();
    history.replaceState(null, '', '/');

    mountApp();
    expect(heading()).toHaveTextContent('Profile');
    expect(screen.getByLabelText(/full name/i)).toHaveValue('Ada');
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '20');
    expect(screen.getByText(/your draft was restored/i)).toBeInTheDocument();
  });

  it('"Discard draft" wipes storage and returns to a blank first step', async () => {
    const first = mountApp();
    await fillAccount(first.user);
    await next(first.user);
    await waitFor(() => expect(heading()).toHaveTextContent('Profile'));
    first.unmount();
    history.replaceState(null, '', '/');

    const { user } = mountApp();
    await user.click(screen.getByRole('button', { name: 'Discard draft' }));
    await waitFor(() => expect(heading()).toHaveTextContent('Account'));
    expect(screen.getByLabelText(/^email/i)).toHaveValue('');
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0');
  });

  it('submitting a resumed draft that lacks the password sends the user back to fix it', async () => {
    const first = mountApp();
    await reachReview(first.user);
    first.unmount();
    history.replaceState(null, '', '/');

    const { user } = mountApp();
    expect(heading()).toHaveTextContent('Review');
    expect(screen.getByText(/password not saved, re-enter to submit/i)).toBeInTheDocument();
    await user.click(screen.getByLabelText(/accept the terms/i));
    await next(user);
    await waitFor(() => expect(heading()).toHaveTextContent('Account'));
    expect(await screen.findByRole('alert')).toHaveTextContent(/password/i);
    expect(screen.queryByText(/You're all set/)).not.toBeInTheDocument();
  });

  it.each([
    ['not JSON', '{oops'],
    ['wrong version', JSON.stringify({ v: 99, step: 3, verified: [0, 1, 2], data: { email: 'x@y.z' } })],
    ['hostile shapes', JSON.stringify({ v: 1, step: 'zzz', verified: 'no', data: { email: { $evil: 1 }, interests: [1, 2], password: 'leak' } })],
  ])('starts fresh on a corrupted or stale draft (%s)', (_label, raw) => {
    localStorage.setItem(STORAGE_KEY, raw);
    mountApp();
    expect(heading()).toHaveTextContent('Account');
    expect(screen.getByLabelText(/^email/i)).toHaveValue('');
    expect(screen.getByLabelText(/^password/i)).toHaveValue('');
    expect(screen.queryByText(/your draft was restored/i)).not.toBeInTheDocument();
  });
});

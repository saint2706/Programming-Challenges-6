const form = document.querySelector('#rsvp-form');
const guestList = document.querySelector('#guest-list');
const formStatus = document.querySelector('#form-status');

function attendingLabel(attending, guestCount) {
  if (attending === 'yes') {
    return guestCount > 0 ? `Attending (+${guestCount})` : 'Attending';
  }
  if (attending === 'maybe') return 'Maybe attending';
  return 'Not attending';
}

// Every guest-supplied field is inserted via textContent (never innerHTML),
// so a malicious name/message can never execute as HTML/script here even
// though the server stores it verbatim - escaping happens at render time,
// not storage time.
function renderGuests(guests) {
  guestList.replaceChildren();

  if (guests.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'guest-list__empty';
    empty.textContent = 'No RSVPs yet - be the first!';
    guestList.append(empty);
    return;
  }

  for (const guest of guests) {
    const item = document.createElement('li');
    item.className = `guest-list__item guest-list__item--${guest.attending}`;

    const name = document.createElement('span');
    name.className = 'guest-list__name';
    name.textContent = guest.name;

    const status = document.createElement('span');
    status.className = 'guest-list__status';
    status.textContent = attendingLabel(guest.attending, guest.guestCount);

    item.append(name, status);

    if (guest.message) {
      const message = document.createElement('p');
      message.className = 'guest-list__message';
      message.textContent = guest.message;
      item.append(message);
    }

    guestList.append(item);
  }
}

async function loadGuests() {
  const res = await fetch('/api/rsvps');
  renderGuests(await res.json());
}

function setStatus(message, kind) {
  formStatus.textContent = message;
  formStatus.className = `form-status form-status--${kind}`;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setStatus('', '');

  const data = Object.fromEntries(new FormData(form).entries());
  const payload = {
    name: data.name,
    email: data.email,
    attending: data.attending,
    guestCount: Number(data.guestCount || 0),
    message: data.message || '',
  };

  try {
    const res = await fetch('/api/rsvps', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const result = await res.json();

    if (!res.ok) {
      setStatus(result.error || 'Something went wrong.', 'error');
      return;
    }

    setStatus(`Thanks, ${result.name}! Your RSVP is in.`, 'success');
    form.reset();
    await loadGuests();
  } catch {
    setStatus('Network error - please try again.', 'error');
  }
});

loadGuests();

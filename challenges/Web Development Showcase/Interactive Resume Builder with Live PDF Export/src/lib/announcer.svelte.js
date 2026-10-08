/**
 * A message for screen readers, shown in the polite live region (`Announcer.svelte`). The text is
 * cleared first so that announcing the same sentence twice in a row is still read out twice.
 */
export const announcer = $state({ message: '' });

export function announce(message) {
  announcer.message = '';
  queueMicrotask(() => {
    announcer.message = message;
  });
}

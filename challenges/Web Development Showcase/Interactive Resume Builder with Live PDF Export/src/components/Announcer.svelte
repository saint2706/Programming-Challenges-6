<script>
  /** A visually hidden polite live region; `announce()` from lib/announcer.svelte.js writes to it. */
  import { announcer } from '../lib/announcer.svelte.js';
</script>

<div class="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcer.message}</div>

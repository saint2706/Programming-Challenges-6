<script lang="ts">
    import { COMMON_DIAGONALS, LIMITS } from '../headcoupled/calibration.ts';
    import { type HeadSession } from '../headcoupled/session.svelte.ts';
    import { type Studio } from '../studio.svelte.ts';

    let { studio, head, fullscreen, onfullscreen }: { studio: Studio; head: HeadSession; fullscreen: boolean; onfullscreen: () => void } = $props();
    const sizes = $derived([...new Set([...COMMON_DIAGONALS, head.settings.diagonal])].sort((a, b) => a - b));
    const cameraOn = $derived(head.status === 'starting' || head.status === 'tracking' || head.status === 'no-face');
    const source = $derived(cameraOn ? 'your camera' : 'the pointer');
</script>

<section>
    <h2>Look through your screen</h2>
    <p>
        The screen becomes a window onto a small room with the shoe in it, drawn for where your eyes are, so it shifts as you move. Right now the picture follows {source}.
    </p>
    <div class="row">
        {#if cameraOn}
            <button class="btn" onclick={() => head.stop()}>Turn camera off</button>
        {:else}
            <button class="btn primary" onclick={() => head.start()}>Use my camera</button>
        {/if}
        <button class="btn" aria-pressed={fullscreen} onclick={onfullscreen}>Full screen (F)</button>
    </div>
    <p class="note" role="status" aria-live="polite">
        {#if head.note}{head.note}{:else if head.status === 'tracking'}Tracking your head. Nothing leaves this page: video is analysed here and discarded.{:else if head.status === 'off'}The camera is off. Move the pointer over the picture instead.{/if}
    </p>
</section>

<section>
    <h2>Your screen</h2>
    <p class="note">
        Your browser cannot tell how big the screen is, so this is a guess{head.chosen ? ' that you changed' : ''}. It decides how large the room is drawn.
    </p>
    <label class="field">
        <span>Screen size (diagonal)</span>
        <select value={head.settings.diagonal} onchange={(e) => head.choose({ diagonal: Number(e.currentTarget.value) })}>
            {#each sizes as d (d)}
                <option value={d}>{d}″</option>
            {/each}
        </select>
    </label>
    <label class="field">
        <span>Depth effect <output>{head.settings.depth.toFixed(2)}×</output></span>
        <input type="range" min={LIMITS.depth[0]} max={LIMITS.depth[1]} step="0.05" value={head.settings.depth} oninput={(e) => head.choose({ depth: Number(e.currentTarget.value) })} />
    </label>
    <details>
        <summary>Camera field of view</summary>
        <p class="note">Laptop webcams see about 66° across. Widen it if the shoe seems too far, narrow it if it seems too near.</p>
        <label class="field">
            <span>Field of view <output>{Math.round(head.settings.hfov)}°</output></span>
            <input type="range" min={LIMITS.hfov[0]} max={LIMITS.hfov[1]} step="1" value={head.settings.hfov} oninput={(e) => head.choose({ hfov: Number(e.currentTarget.value) })} />
        </label>
    </details>
</section>

<section>
    <h2>Readout</h2>
    <button class="btn" aria-pressed={studio.hud} onclick={() => (studio.hud = !studio.hud)}>Show numbers</button>
    {#if studio.shoeScale < 0.995}
        <p class="note">On this screen the window is smaller than the shoe, so it is drawn at {Math.round(studio.shoeScale * 100)}% of life size.</p>
    {:else}
        <p class="note">The shoe is drawn at life size (29.8 cm long).</p>
    {/if}
</section>

<style>
    section {
        margin-bottom: 20px;
    }
    h2 {
        margin: 0 0 8px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
    }
    p {
        margin: 0 0 12px;
    }
    .note {
        color: var(--muted);
        font-size: 0.9rem;
    }
    .row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .field {
        display: grid;
        gap: 4px;
        margin-bottom: 12px;
    }
    .field span {
        display: flex;
        justify-content: space-between;
    }
    select {
        min-height: 40px;
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 8px;
        padding: 0 8px;
    }
    input[type='range'] {
        width: 100%;
        accent-color: var(--accent);
    }
    summary {
        cursor: pointer;
        margin-bottom: 8px;
    }
</style>

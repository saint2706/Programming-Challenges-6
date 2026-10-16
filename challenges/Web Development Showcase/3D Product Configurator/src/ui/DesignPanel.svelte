<script lang="ts">
    import { FINISH_IDS, FINISHES, PALETTE, PART_IDS, PART_LABELS, PRESETS, presetOf } from '../parts.ts';
    import { type Studio } from '../studio.svelte.ts';

    let { studio }: { studio: Studio } = $props();
    const activePreset = $derived(presetOf(studio.design.colors));
    const colour = $derived(studio.design.colors[studio.selected]);

    // Radio groups move with the arrow keys, one stop in the group.
    function arrows(event: KeyboardEvent, ids: readonly string[], current: string, pick: (id: string) => void) {
        const step = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1 : event.key === 'ArrowLeft' || event.key === 'ArrowUp' ? -1 : 0;
        if (!step) return;
        event.preventDefault();
        const next = ids[(ids.indexOf(current) + step + ids.length) % ids.length]!;
        pick(next);
        (event.currentTarget as HTMLElement).querySelector<HTMLElement>(`[data-id="${next}"]`)?.focus();
    }
</script>

<section>
    <h2>Colourway</h2>
    <div class="chips">
        {#each PRESETS as preset (preset.id)}
            <button class="chip" aria-pressed={activePreset === preset.id} onclick={() => studio.applyPreset(preset.id)}>
                <span class="dots" aria-hidden="true">
                    {#each [preset.colors.upper, preset.colors.overlays, preset.colors.accents] as c}<i style:background={c}></i>{/each}
                </span>
                {preset.label}
            </button>
        {/each}
    </div>
</section>

<section>
    <h2 id="part-heading">Part</h2>
    <div class="parts" role="radiogroup" aria-labelledby="part-heading" tabindex="-1" onkeydown={(e) => arrows(e, PART_IDS, studio.selected, (id) => (studio.selected = id as typeof studio.selected))}>
        {#each PART_IDS as id (id)}
            <button class="part" role="radio" aria-checked={studio.selected === id} tabindex={studio.selected === id ? 0 : -1} data-id={id} onclick={() => (studio.selected = id)}>
                <i style:background={studio.design.colors[id]} aria-hidden="true"></i>
                {PART_LABELS[id]}
            </button>
        {/each}
    </div>
</section>

<section>
    <h2 id="colour-heading">Colour of {PART_LABELS[studio.selected].toLowerCase()}</h2>
    <div class="swatches" role="group" aria-labelledby="colour-heading">
        {#each PALETTE as swatch (swatch.hex)}
            <button class="swatch" style:background={swatch.hex} aria-label={swatch.name} aria-pressed={colour === swatch.hex} onclick={() => studio.setColor(studio.selected, swatch.hex)}></button>
        {/each}
    </div>
    <label class="custom">
        <span>Any colour</span>
        <input type="color" value={colour} oninput={(e) => studio.setColor(studio.selected, e.currentTarget.value)} />
        <output>{colour}</output>
    </label>
</section>

<section>
    <h2 id="finish-heading">Finish of {PART_LABELS[studio.selected].toLowerCase()}</h2>
    <div class="finishes" role="radiogroup" aria-labelledby="finish-heading" tabindex="-1" onkeydown={(e) => arrows(e, FINISH_IDS, studio.design.finishes[studio.selected], (id) => studio.setFinish(studio.selected, id as (typeof FINISH_IDS)[number]))}>
        {#each FINISH_IDS as id (id)}
            <button class="btn" role="radio" aria-checked={studio.design.finishes[studio.selected] === id} tabindex={studio.design.finishes[studio.selected] === id ? 0 : -1} data-id={id} onclick={() => studio.setFinish(studio.selected, id)}>{FINISHES[id].label}</button>
        {/each}
    </div>
</section>

<section class="footer">
    <button class="btn" onclick={() => studio.reset()}>Reset</button>
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
    .chips,
    .finishes {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid var(--line);
        background: var(--panel);
        border-radius: 999px;
        padding: 6px 12px 6px 8px;
        min-height: 40px;
    }
    .chip[aria-pressed='true'] {
        border-color: var(--accent);
        box-shadow: inset 0 0 0 1px var(--accent);
    }
    .dots {
        display: inline-flex;
    }
    .dots i {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        border: 1px solid var(--line);
        margin-left: -4px;
    }
    .dots i:first-child {
        margin-left: 0;
    }
    .parts {
        display: grid;
        gap: 4px;
    }
    .part {
        display: flex;
        align-items: center;
        gap: 10px;
        text-align: left;
        border: 1px solid transparent;
        background: transparent;
        border-radius: 10px;
        padding: 8px 10px;
        min-height: 44px;
    }
    .part:hover {
        background: color-mix(in srgb, var(--ink) 6%, transparent);
    }
    .part[aria-checked='true'] {
        border-color: var(--accent);
        background: color-mix(in srgb, var(--accent) 10%, transparent);
    }
    .part i {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        border: 1px solid var(--line);
        flex: none;
    }
    .swatches {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
    }
    .swatch {
        aspect-ratio: 1;
        min-width: 36px;
        border-radius: 50%;
        border: 1px solid var(--line);
    }
    .swatch[aria-pressed='true'] {
        outline: 3px solid var(--ink);
        outline-offset: 2px;
    }
    .custom {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 12px;
    }
    .custom input {
        width: 44px;
        height: 36px;
        padding: 0;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: none;
    }
    .custom output {
        font-family: ui-monospace, monospace;
        color: var(--muted);
    }
    .btn[aria-checked='true'] {
        background: var(--accent);
        border-color: var(--accent);
        color: var(--accent-ink);
    }
</style>

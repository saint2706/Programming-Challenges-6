<script>
  let { block, selected, isFirst, isLast, onSelect, onRemove, onMoveUp, onMoveDown, onChange } =
    $props();

  function handleInput(key, value) {
    onChange({ [key]: value });
  }

  function stop(fn) {
    return (e) => {
      e.stopPropagation();
      fn();
    };
  }
</script>

<div
  class="block-row"
  class:selected
  onclick={onSelect}
  onkeydown={(e) => e.key === 'Enter' && onSelect()}
  role="button"
  tabindex="0"
>
  <div class="block-header">
    <span class="type">{block.type}</span>
    <div class="controls">
      <button type="button" disabled={isFirst} onclick={stop(onMoveUp)} aria-label="Move up">↑</button>
      <button type="button" disabled={isLast} onclick={stop(onMoveDown)} aria-label="Move down">↓</button>
      <button type="button" onclick={stop(onRemove)} aria-label="Remove block">✕</button>
    </div>
  </div>

  {#if selected}
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <!-- svelte-ignore a11y_click_events_have_key_events -->
    <div class="fields" onclick={(e) => e.stopPropagation()}>
      {#if block.type === 'heading' || block.type === 'paragraph'}
        <label>
          Text
          <textarea value={block.props.text} oninput={(e) => handleInput('text', e.target.value)}
          ></textarea>
        </label>
        <label>
          Align
          <select value={block.props.align} onchange={(e) => handleInput('align', e.target.value)}>
            <option value="left">Left</option>
            <option value="center">Center</option>
            <option value="right">Right</option>
          </select>
        </label>
        <label>
          Color
          <input type="color" value={block.props.color} oninput={(e) => handleInput('color', e.target.value)} />
        </label>
      {/if}

      {#if block.type === 'image'}
        <label>
          Image URL
          <input type="text" value={block.props.src} oninput={(e) => handleInput('src', e.target.value)} />
        </label>
        <label>
          Alt text
          <input type="text" value={block.props.alt} oninput={(e) => handleInput('alt', e.target.value)} />
        </label>
        <label>
          Width (px)
          <input
            type="number"
            value={block.props.width}
            oninput={(e) => handleInput('width', Number(e.target.value))}
          />
        </label>
      {/if}

      {#if block.type === 'button'}
        <label>
          Label
          <input type="text" value={block.props.label} oninput={(e) => handleInput('label', e.target.value)} />
        </label>
        <label>
          Link URL
          <input type="text" value={block.props.href} oninput={(e) => handleInput('href', e.target.value)} />
        </label>
        <label>
          Background
          <input
            type="color"
            value={block.props.bgColor}
            oninput={(e) => handleInput('bgColor', e.target.value)}
          />
        </label>
        <label>
          Text color
          <input
            type="color"
            value={block.props.textColor}
            oninput={(e) => handleInput('textColor', e.target.value)}
          />
        </label>
      {/if}

      {#if block.type === 'divider'}
        <label>
          Color
          <input type="color" value={block.props.color} oninput={(e) => handleInput('color', e.target.value)} />
        </label>
        <label>
          Height (px)
          <input
            type="number"
            value={block.props.height}
            oninput={(e) => handleInput('height', Number(e.target.value))}
          />
        </label>
      {/if}

      {#if block.type === 'spacer'}
        <label>
          Height (px)
          <input
            type="number"
            value={block.props.height}
            oninput={(e) => handleInput('height', Number(e.target.value))}
          />
        </label>
      {/if}

      {#if block.type === 'columns'}
        <label>
          Left text
          <textarea
            value={block.props.leftText}
            oninput={(e) => handleInput('leftText', e.target.value)}
          ></textarea>
        </label>
        <label>
          Right text
          <textarea
            value={block.props.rightText}
            oninput={(e) => handleInput('rightText', e.target.value)}
          ></textarea>
        </label>
      {/if}
    </div>
  {/if}
</div>

<style>
  .block-row {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    margin-bottom: 8px;
    padding: 8px 10px;
    cursor: pointer;
    background: white;
  }
  .block-row.selected {
    border-color: #2563eb;
  }
  .block-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .type {
    text-transform: capitalize;
    font-weight: 600;
    font-size: 13px;
  }
  .controls button {
    margin-left: 4px;
    font-size: 12px;
    cursor: pointer;
  }
  .controls button:disabled {
    opacity: 0.35;
    cursor: default;
  }
  .fields {
    margin-top: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    cursor: default;
  }
  .fields label {
    display: flex;
    flex-direction: column;
    font-size: 12px;
    gap: 2px;
    color: #374151;
  }
  textarea {
    min-height: 50px;
    font-family: inherit;
    resize: vertical;
  }
</style>

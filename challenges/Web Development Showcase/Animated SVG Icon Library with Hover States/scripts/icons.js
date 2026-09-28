/**
 * Data/tech-themed SVG icon set for the gallery.
 *
 * Each icon's `markup` is a fully self-contained, copy-pasteable `<svg>`
 * string on a 24x24 viewBox using `stroke="currentColor"` so it inherits
 * color from its container. `technique` records whether the hover animation
 * is driven by CSS (`style.css`, keyed off `.icon-panel[data-icon="…"]:hover`)
 * or native SMIL (inline `<animate>`/`<animateTransform>`/`<animateMotion>`
 * elements wired to `begin="<svg-id>.mouseover"` / `end="<svg-id>.mouseout"`).
 *
 * Kept DOM-free and side-effect-free so the whole set can be validated by
 * plain unit tests without rendering anything.
 */

const STROKE = 'fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"';

function svg(id, body) {
  return `<svg id="${id}" class="icon-svg" viewBox="0 0 24 24" width="40" height="40" ${STROKE} aria-hidden="true">${body}</svg>`;
}

/** Transparent full-viewBox rect so SMIL `mouseover`/`mouseout` fire across the whole icon, not just its visible strokes. */
const HIT_AREA = '<rect x="0" y="0" width="24" height="24" class="hit-area" fill="transparent" stroke="none"/>';

export const CATEGORIES = [
  { id: 'data', label: 'Data' },
  { id: 'workflow', label: 'Workflow' },
  { id: 'infra', label: 'Infra' },
  { id: 'ml', label: 'ML' },
];

export const ICONS = [
  {
    id: 'database',
    name: 'Database',
    category: 'data',
    technique: 'css',
    description: 'Stacked disks that separate apart, staggered, on hover.',
    markup: svg(
      'database',
      `<ellipse class="db-disk" cx="12" cy="5" rx="7" ry="2.5"/>
<path class="db-disk db-disk-2" d="M5 5v5c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5V5"/>
<path class="db-disk db-disk-3" d="M5 10v5c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5v-5"/>
<path class="db-disk db-disk-4" d="M5 15v4c0 1.38 3.13 2.5 7 2.5s7-1.12 7-2.5v-4"/>`,
    ),
  },
  {
    id: 'bar-chart',
    name: 'Bar Chart',
    category: 'data',
    technique: 'css',
    description: 'Three bars grow up from the baseline with a staggered delay.',
    markup: svg(
      'bar-chart',
      `<line x1="3" y1="21" x2="21" y2="21"/>
<rect class="bar bar-1" x="5" y="14" width="3" height="7"/>
<rect class="bar bar-2" x="10.5" y="10" width="3" height="11"/>
<rect class="bar bar-3" x="16" y="6" width="3" height="15"/>`,
    ),
  },
  {
    id: 'line-chart',
    name: 'Line Chart',
    category: 'data',
    technique: 'css',
    description: 'A trend line draws itself in via animated stroke-dashoffset.',
    markup: svg(
      'line-chart',
      `<polyline class="trend-line" points="3,17 8,11 12,14 16,7 21,9" pathLength="100"/>
<circle cx="21" cy="9" r="1.4" fill="currentColor" stroke="none"/>`,
    ),
  },
  {
    id: 'donut-chart',
    name: 'Donut Chart',
    category: 'data',
    technique: 'css',
    description: 'The two data segments spin a quarter turn around the track.',
    markup: svg(
      'donut-chart',
      `<circle cx="12" cy="12" r="8" stroke-opacity="0.25"/>
<g class="donut-segments">
<circle cx="12" cy="12" r="8" pathLength="100" stroke-dasharray="30 70"/>
<circle cx="12" cy="12" r="8" pathLength="100" stroke-dasharray="20 80" stroke-dashoffset="-30"/>
</g>`,
    ),
  },
  {
    id: 'scatter-plot',
    name: 'Scatter Plot',
    category: 'data',
    technique: 'css',
    description: 'Four data points pulse in sequence.',
    markup: svg(
      'scatter-plot',
      `<line x1="3" y1="3" x2="3" y2="21"/>
<line x1="3" y1="21" x2="21" y2="21"/>
<circle class="dot dot-1" cx="7" cy="15" r="1.3" fill="currentColor" stroke="none"/>
<circle class="dot dot-2" cx="11" cy="9" r="1.3" fill="currentColor" stroke="none"/>
<circle class="dot dot-3" cx="14" cy="13" r="1.3" fill="currentColor" stroke="none"/>
<circle class="dot dot-4" cx="18" cy="6" r="1.3" fill="currentColor" stroke="none"/>`,
    ),
  },
  {
    id: 'dataframe-table',
    name: 'DataFrame',
    category: 'data',
    technique: 'css',
    description: 'Table rows highlight one after another.',
    markup: svg(
      'dataframe-table',
      `<rect x="3" y="4" width="18" height="16" rx="1"/>
<line x1="3" y1="9" x2="21" y2="9"/>
<line x1="3" y1="14" x2="21" y2="14"/>
<line x1="9" y1="4" x2="9" y2="20"/>
<rect class="row row-1" x="3.5" y="9.3" width="17" height="4.4" fill="currentColor" stroke="none" opacity="0"/>
<rect class="row row-2" x="3.5" y="14.3" width="17" height="5.4" fill="currentColor" stroke="none" opacity="0"/>`,
    ),
  },
  {
    id: 'magnifier-insight',
    name: 'Insight Search',
    category: 'data',
    technique: 'css',
    description: 'The magnifier handle wiggles side to side.',
    markup: svg(
      'magnifier-insight',
      `<circle cx="10.5" cy="10.5" r="6.5"/>
<path d="M9 13l1.5-3 1.5 2 1.5-4" stroke-width="1.4"/>
<line class="handle" x1="15.5" y1="15.5" x2="21" y2="21"/>`,
    ),
  },
  {
    id: 'funnel',
    name: 'ETL Funnel',
    category: 'workflow',
    technique: 'css',
    description: 'A droplet repeatedly falls through the funnel while hovered.',
    markup: svg(
      'funnel',
      `<path d="M3 4h18l-7 9v6l-4 2v-8z"/>
<circle class="drop" cx="12" cy="4" r="1.1" fill="currentColor" stroke="none"/>`,
    ),
  },
  {
    id: 'api-plug',
    name: 'API Connector',
    category: 'workflow',
    technique: 'css',
    description: 'Two connector halves slide together to "plug in".',
    markup: svg(
      'api-plug',
      `<g class="plug plug-left">
<path d="M3 12h5M4 9v6"/>
<rect x="7" y="9" width="4" height="6" rx="1"/>
</g>
<g class="plug plug-right">
<rect x="13" y="9" width="4" height="6" rx="1"/>
<path d="M21 12h-5m3-3v6"/>
</g>`,
    ),
  },
  {
    id: 'notebook-pen',
    name: 'Storyteller Notebook',
    category: 'workflow',
    technique: 'css',
    description: 'A second line of "ink" draws itself in under a pen.',
    markup: svg(
      'notebook-pen',
      `<rect x="4" y="3" width="13" height="18" rx="1.5"/>
<line x1="7" y1="8" x2="14" y2="8"/>
<line class="ink-line" x1="7" y1="12" x2="14" y2="12" pathLength="100"/>
<path d="M15 15l5-5 2 2-5 5-3 1z"/>`,
    ),
  },
  {
    id: 'terminal',
    name: 'Terminal',
    category: 'infra',
    technique: 'css',
    description: 'The command-line cursor blinks while hovered.',
    markup: svg(
      'terminal',
      `<rect x="3" y="4" width="18" height="16" rx="1.5"/>
<path d="M6.5 9l3 3-3 3"/>
<line class="cursor" x1="11.5" y1="15" x2="15.5" y2="15"/>`,
    ),
  },
  {
    id: 'cpu-chip',
    name: 'CPU',
    category: 'infra',
    technique: 'css',
    description: 'The chip core pulses to suggest active compute.',
    markup: svg(
      'cpu-chip',
      `<rect x="7" y="7" width="10" height="10" rx="1"/>
<rect class="core" x="10" y="10" width="4" height="4" fill="currentColor" stroke="none" opacity="0.35"/>
<line x1="9" y1="3" x2="9" y2="7"/><line x1="12" y1="3" x2="12" y2="7"/><line x1="15" y1="3" x2="15" y2="7"/>
<line x1="9" y1="17" x2="9" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/><line x1="15" y1="17" x2="15" y2="21"/>
<line x1="3" y1="9" x2="7" y2="9"/><line x1="3" y1="12" x2="7" y2="12"/><line x1="3" y1="15" x2="7" y2="15"/>
<line x1="17" y1="9" x2="21" y2="9"/><line x1="17" y1="12" x2="21" y2="12"/><line x1="17" y1="15" x2="21" y2="15"/>`,
    ),
  },
  {
    id: 'cloud-aws',
    name: 'Cloud',
    category: 'infra',
    technique: 'css',
    description: 'The cloud gently floats up and down while hovered.',
    markup: svg(
      'cloud-aws',
      `<path class="cloud" d="M7 17a4 4 0 010-8 5 5 0 019.6-1.5A3.5 3.5 0 0117 17H7z"/>`,
    ),
  },
  {
    id: 'settings-gear',
    name: 'Config',
    category: 'infra',
    technique: 'css',
    description: 'The gear spins continuously while hovered.',
    markup: svg(
      'settings-gear',
      `<circle cx="12" cy="12" r="3"/>
<path class="gear" d="M12 3v3m0 12v3m9-9h-3M6 12H3m14.5-6.5l-2 2M9.5 14.5l-2 2m11-2l-2-2M8.5 8.5l-2-2"/>`,
    ),
  },
  {
    id: 'neural-network',
    name: 'Neural Network',
    category: 'ml',
    technique: 'smil',
    description: 'Hidden and output nodes pulse in sequence as a signal "fires" through the network.',
    markup: svg(
      'neural-network',
      `${HIT_AREA}
<line x1="4" y1="6" x2="12" y2="9"/>
<line x1="4" y1="12" x2="12" y2="9"/>
<line x1="4" y1="18" x2="12" y2="15"/>
<line x1="12" y1="9" x2="20" y2="12"/>
<line x1="12" y1="15" x2="20" y2="12"/>
<circle cx="4" cy="6" r="1.6" fill="currentColor" stroke="none"/>
<circle cx="4" cy="12" r="1.6" fill="currentColor" stroke="none"/>
<circle cx="4" cy="18" r="1.6" fill="currentColor" stroke="none"/>
<circle cx="12" cy="9" r="1.6" fill="currentColor" stroke="none">
<animate attributeName="r" values="1.6;2.4;1.6" dur="0.8s" begin="neural-network.mouseover" end="neural-network.mouseout" repeatCount="indefinite"/>
</circle>
<circle cx="12" cy="15" r="1.6" fill="currentColor" stroke="none">
<animate attributeName="r" values="1.6;2.4;1.6" dur="0.8s" begin="neural-network.mouseover+0.2s" end="neural-network.mouseout" repeatCount="indefinite"/>
</circle>
<circle cx="20" cy="12" r="1.8" fill="currentColor" stroke="none">
<animate attributeName="r" values="1.8;2.6;1.8" dur="0.8s" begin="neural-network.mouseover+0.4s" end="neural-network.mouseout" repeatCount="indefinite"/>
</circle>`,
    ),
  },
  {
    id: 'dashboard-gauge',
    name: 'Dashboard Gauge',
    category: 'ml',
    technique: 'smil',
    description: 'The needle sweeps back and forth across the gauge via animateTransform.',
    markup: svg(
      'dashboard-gauge',
      `${HIT_AREA}
<path d="M4 17a8 8 0 0116 0"/>
<line x1="12" y1="17" x2="12" y2="9" class="needle">
<animateTransform attributeName="transform" type="rotate" values="-50 12 17;50 12 17;-50 12 17" dur="1.4s" begin="dashboard-gauge.mouseover" end="dashboard-gauge.mouseout" repeatCount="indefinite"/>
</line>
<circle cx="12" cy="17" r="1.3" fill="currentColor" stroke="none"/>`,
    ),
  },
  {
    id: 'globe-locale',
    name: 'Globe',
    category: 'ml',
    technique: 'smil',
    description: 'The meridian lines spin around the globe via animateTransform.',
    markup: svg(
      'globe-locale',
      `${HIT_AREA}
<circle cx="12" cy="12" r="9"/>
<line x1="3" y1="12" x2="21" y2="12"/>
<g class="meridians">
<ellipse cx="12" cy="12" rx="3.5" ry="9"/>
<ellipse cx="12" cy="12" rx="9" ry="3.5"/>
<animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="2.2s" begin="globe-locale.mouseover" end="globe-locale.mouseout" repeatCount="indefinite"/>
</g>`,
    ),
  },
  {
    id: 'flask-experiment',
    name: 'Experiment',
    category: 'ml',
    technique: 'smil',
    description: 'Bubbles rise and fade inside the flask, staggered, via animate.',
    markup: svg(
      'flask-experiment',
      `${HIT_AREA}
<path d="M10 3h4v5l5 10a2 2 0 01-1.8 3H6.8A2 2 0 015 18l5-10z"/>
<line x1="8.5" y1="3" x2="15.5" y2="3"/>
<circle cx="11" cy="17" r="1" fill="currentColor" stroke="none">
<animate attributeName="cy" values="17;10" dur="1.2s" begin="flask-experiment.mouseover" end="flask-experiment.mouseout" repeatCount="indefinite"/>
<animate attributeName="opacity" values="1;0" dur="1.2s" begin="flask-experiment.mouseover" end="flask-experiment.mouseout" repeatCount="indefinite"/>
</circle>
<circle cx="13.5" cy="17" r="0.8" fill="currentColor" stroke="none">
<animate attributeName="cy" values="17;9" dur="1.4s" begin="flask-experiment.mouseover+0.3s" end="flask-experiment.mouseout" repeatCount="indefinite"/>
<animate attributeName="opacity" values="1;0" dur="1.4s" begin="flask-experiment.mouseover+0.3s" end="flask-experiment.mouseout" repeatCount="indefinite"/>
</circle>`,
    ),
  },
  {
    id: 'git-branch',
    name: 'Git Branch',
    category: 'workflow',
    technique: 'smil',
    description: 'A commit dot travels from the trunk out to the feature branch via animateMotion.',
    markup: svg(
      'git-branch',
      `${HIT_AREA}
<circle cx="6" cy="5" r="1.6"/>
<circle cx="6" cy="19" r="1.6"/>
<circle cx="18" cy="12" r="1.6"/>
<path d="M6 6.6V19"/>
<path id="git-branch-path" d="M6 10c0 4 3 2 12 2" fill="none" stroke="none"/>
<circle class="commit-dot" r="1.1" fill="currentColor" stroke="none">
<animateMotion dur="1.6s" begin="git-branch.mouseover" end="git-branch.mouseout" repeatCount="indefinite">
<mpath href="#git-branch-path"/>
</animateMotion>
</circle>`,
    ),
  },
];

export const SMIL_TECHNIQUE_MIN = 3;
export const SMIL_TECHNIQUE_MAX = 5;

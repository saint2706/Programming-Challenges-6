/**
 * A template says which regions a page has and what goes in each. `header` is `'full'` (name,
 * label and contact line), `'name'` (name and label only) or `null`; `contact` puts a standalone
 * contact block at the top of the region; `sections` is `'all'` or a list of section ids.
 * Typography and spacing live in the template's stylesheet (`src/styles/templates.css`).
 */
export default {
  id: 'classic',
  name: 'Classic',
  description: 'One column, ruled section headings. The safest layout for applicant tracking systems.',
  regions: [{ id: 'main', header: 'full', contact: false, sections: 'all' }],
};

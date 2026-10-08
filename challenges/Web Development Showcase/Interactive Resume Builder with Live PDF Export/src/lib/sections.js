/**
 * What each resume section is called and which fields its entries have. The side panel builds its
 * forms from `fields`, the paginator's block builder takes headings from `title`, and `newEntry`
 * makes blank entries that are valid against the schema.
 *
 * Field types: text, textarea, date, url, bullets (array of strings), csv (array edited as one
 * comma-separated string).
 */

export const SECTION_META = {
  summary: { title: 'Summary' },
  work: {
    title: 'Experience',
    singular: 'position',
    labelKeys: ['position', 'name'],
    fields: [
      { key: 'position', label: 'Position', type: 'text' },
      { key: 'name', label: 'Company', type: 'text' },
      { key: 'url', label: 'Link', type: 'url' },
      { key: 'startDate', label: 'Start', type: 'date' },
      { key: 'endDate', label: 'End (blank means present)', type: 'date' },
      { key: 'summary', label: 'Summary', type: 'textarea' },
      { key: 'highlights', label: 'Bullets', type: 'bullets' },
    ],
  },
  education: {
    title: 'Education',
    singular: 'school',
    labelKeys: ['institution'],
    fields: [
      { key: 'institution', label: 'Institution', type: 'text' },
      { key: 'studyType', label: 'Degree', type: 'text' },
      { key: 'area', label: 'Field of study', type: 'text' },
      { key: 'url', label: 'Link', type: 'url' },
      { key: 'startDate', label: 'Start', type: 'date' },
      { key: 'endDate', label: 'End (blank means present)', type: 'date' },
      { key: 'score', label: 'Score', type: 'text' },
    ],
  },
  projects: {
    title: 'Projects',
    singular: 'project',
    labelKeys: ['name'],
    fields: [
      { key: 'name', label: 'Name', type: 'text' },
      { key: 'url', label: 'Link', type: 'url' },
      { key: 'startDate', label: 'Start', type: 'date' },
      { key: 'endDate', label: 'End', type: 'date' },
      { key: 'description', label: 'Description', type: 'textarea' },
      { key: 'highlights', label: 'Bullets', type: 'bullets' },
    ],
  },
  skills: {
    title: 'Skills',
    singular: 'skill group',
    labelKeys: ['name'],
    fields: [
      { key: 'name', label: 'Group name', type: 'text' },
      { key: 'keywords', label: 'Skills (comma separated)', type: 'csv' },
      { key: 'level', label: 'Level', type: 'text' },
    ],
  },
  certificates: {
    title: 'Certifications',
    singular: 'certificate',
    labelKeys: ['name'],
    fields: [
      { key: 'name', label: 'Name', type: 'text' },
      { key: 'issuer', label: 'Issuer', type: 'text' },
      { key: 'date', label: 'Date', type: 'date' },
      { key: 'url', label: 'Link', type: 'url' },
    ],
  },
  languages: {
    title: 'Languages',
    singular: 'language',
    labelKeys: ['language'],
    fields: [
      { key: 'language', label: 'Language', type: 'text' },
      { key: 'fluency', label: 'Fluency', type: 'text' },
    ],
  },
  awards: {
    title: 'Awards',
    singular: 'award',
    labelKeys: ['title'],
    fields: [
      { key: 'title', label: 'Title', type: 'text' },
      { key: 'awarder', label: 'Awarded by', type: 'text' },
      { key: 'date', label: 'Date', type: 'date' },
      { key: 'summary', label: 'Summary', type: 'textarea' },
    ],
  },
};

/** A blank entry for `section`, valid against the schema. */
export function newEntry(section) {
  const fields = SECTION_META[section]?.fields;
  if (!fields) throw new Error(`section "${section}" has no entries`);
  return Object.fromEntries(
    fields.map((f) => [f.key, f.type === 'bullets' || f.type === 'csv' ? [] : '']),
  );
}

/** A short label for an entry in the side panel list. */
export function entryLabel(section, entry, index) {
  const meta = SECTION_META[section];
  const parts = meta.labelKeys.map((key) => entry[key]).filter(Boolean);
  return parts.length ? parts.join(', ') : `New ${meta.singular} ${index + 1}`;
}

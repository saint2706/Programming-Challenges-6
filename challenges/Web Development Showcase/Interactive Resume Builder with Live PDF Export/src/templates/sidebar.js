/** Two regions that flow independently: experience on the right, facts about the person on the left. */
export default {
  id: 'sidebar',
  name: 'Sidebar',
  description: 'Two columns: a tinted sidebar for contact, skills and certifications beside the main story.',
  regions: [
    {
      id: 'main',
      header: 'name',
      contact: false,
      sections: ['summary', 'work', 'education', 'projects', 'awards'],
    },
    {
      id: 'sidebar',
      header: null,
      contact: true,
      sections: ['skills', 'certificates', 'languages'],
    },
  ],
};

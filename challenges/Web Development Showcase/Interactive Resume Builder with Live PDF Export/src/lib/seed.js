import { emptyResume, parseResume } from './schema.js';

/**
 * The starting document: the author's real education, experience, skills, projects and
 * certifications, taken from their public portfolio (saint2706.github.io). Contact details (email,
 * phone) are left empty; the header links to GitHub and LinkedIn instead.
 */
function seedData() {
  return {
    // the Sidebar template shows this particular resume best, and fits it on one page
    'x-layout': { template: 'sidebar' },
    basics: {
      name: 'Rishabh Agrawal',
      label: 'Data Storyteller & Analytics Strategist',
      email: '',
      phone: '',
      url: 'https://saint2706.github.io',
      summary:
        'Big Data Analytics postgraduate at Goa Institute of Management with a Computer Science background. I thrive at the intersection of data, technology, and creativity—using analytics, AI, and software development to turn complex challenges into actionable insights.',
      location: { city: 'Goa', region: '', countryCode: 'IN' },
      profiles: [
        { network: 'GitHub', username: 'saint2706', url: 'https://github.com/saint2706' },
        {
          network: 'LinkedIn',
          username: 'rishabh-agrawal',
          url: 'https://www.linkedin.com/in/rishabh-agrawal-1807321b9',
        },
      ],
    },
    work: [
      {
        name: 'TheSmartBridge',
        position: 'Data Intern',
        url: '',
        startDate: '2023-05',
        endDate: '2023-07',
        summary:
          'Evaluated SaaS market opportunities and aligned stakeholders on a roadmap for a new analytics product.',
        highlights: [
          'Researched product gaps and competitive benchmarks to shape the go-to-market plan.',
          'Developed personas, positioning, and phased delivery milestones with a five-member intern cohort.',
          'Presented recommendations to senior management.',
        ],
      },
      {
        name: 'Mood Indigo, IIT Bombay',
        position: 'Indigo Squad Member',
        url: '',
        startDate: '2022-07',
        endDate: '2022-12',
        summary:
          "Drove experiential marketing, partnerships, and community engagement for Asia's largest college cultural festival.",
        highlights: [
          'Amplified outreach to 1,700+ colleges through storytelling-led campaigns.',
          'Boosted social engagement by 25% via multi-platform content strategies.',
          'Secured three new sponsorships.',
        ],
      },
    ],
    education: [
      {
        institution: 'Goa Institute of Management (GIM)',
        url: '',
        area: 'Big Data Analytics',
        studyType: 'PGDM',
        startDate: '2025-06',
        endDate: '2027-07',
        score: '',
      },
      {
        institution: 'Vellore Institute of Technology',
        url: '',
        area: 'Computer Science',
        studyType: 'B.Tech',
        startDate: '2020',
        endDate: '2024',
        score: '',
      },
    ],
    projects: [
      {
        name: 'Coding-For-MBA',
        url: 'https://github.com/saint2706/Coding-For-MBA',
        startDate: '',
        endDate: '',
        description:
          'A comprehensive Python coding curriculum designed for MBA students, bridging business strategy and technical implementation. Features 15 structured lessons from basics to data analysis.',
        highlights: [],
      },
      {
        name: 'AI Attendance Management System',
        url: 'https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition',
        startDate: '',
        endDate: '',
        description:
          'Full-stack facial recognition attendance platform with real-time face detection, automated check-ins, and a comprehensive analytics dashboard.',
        highlights: [],
      },
      {
        name: 'Client Modding Guide',
        url: 'https://github.com/saint2706/Client-Modding-Guide',
        startDate: '',
        endDate: '',
        description:
          'Comprehensive guide for Discord client modding with step-by-step tutorials. My most starred personal project, with an active community.',
        highlights: [],
      },
    ],
    skills: [
      {
        name: 'Programming',
        level: '',
        keywords: ['Python', 'SQL', 'JavaScript', 'Java', 'C++', 'R', 'HTML/CSS'],
      },
      {
        name: 'Data Science & AI',
        level: '',
        keywords: [
          'Data Analytics',
          'Machine Learning',
          'Pandas',
          'NumPy',
          'Scikit-learn',
          'Tableau',
          'Power BI',
        ],
      },
      {
        name: 'Frameworks & Cloud',
        level: '',
        keywords: ['AWS', 'Azure', 'Google Cloud', 'TensorFlow', 'PyTorch', 'React', 'Node.js'],
      },
    ],
    certificates: [
      {
        name: 'AWS Certified Cloud Practitioner',
        issuer: 'Amazon Web Services',
        date: '2023-07',
        url: '',
      },
      { name: 'Data Analytics powered by IBM', issuer: 'SmartInternz', date: '2023-07', url: '' },
      {
        name: 'AWS Academy Graduate - Cloud Foundations',
        issuer: 'Amazon Web Services',
        date: '2023-06',
        url: '',
      },
      {
        name: 'JavaScript Algorithms and Data Structures',
        issuer: 'freeCodeCamp',
        date: '2023-01',
        url: '',
      },
    ],
    languages: [
      { language: 'English', fluency: 'Native / Bilingual' },
      { language: 'Hindi', fluency: 'Native / Bilingual' },
      { language: 'Marathi', fluency: 'Native / Bilingual' },
      { language: 'French', fluency: 'Professional working' },
    ],
  };
}

/** A fresh copy of the seed resume (validated, so the defaults are filled in). */
export function seedResume() {
  const result = parseResume(seedData());
  if (!result.ok) throw new Error(`seed resume is invalid: ${JSON.stringify(result.errors)}`);
  return result.data;
}

/** An empty resume with the default layout. */
export function blankResume() {
  return emptyResume();
}

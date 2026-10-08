import { parseResume } from './schema.js';

/**
 * A long, entirely fictional resume ("Avery Morgan") used for the "Load example" button, for
 * pagination demos and as the end-to-end test fixture. It is deterministic: the same arguments
 * always produce the same document. `copies` repeats the work history to make it longer.
 */

const COMPANIES = [
  ['Brightwave Systems', 'Staff Software Engineer'],
  ['Harbor & Pine Software', 'Senior Software Engineer'],
  ['Quillstone Data', 'Software Engineer II'],
  ['Lumen Ridge', 'Software Engineer'],
  ['Fernhill Robotics', 'Platform Engineer'],
  ['Oakline Health', 'Junior Developer'],
];

const BULLETS = [
  (n) => `Led a team of ${n + 3} engineers to redesign the ingestion pipeline, cutting p95 latency from ${900 + n * 40} ms to ${180 + n * 10} ms.`,
  (n) => `Introduced contract tests across ${n + 4} services, which caught ${12 + n} breaking changes before they reached production.`,
  (n) => `Migrated a ${n + 2}-terabyte reporting database to a partitioned schema with zero downtime, saving about $${(n + 2) * 3},000 a month.`,
  (n) => `Mentored ${n + 2} engineers through promotion, and wrote the team's onboarding guide that new hires still use.`,
  (n) => `Built an internal feature-flag service used by ${n + 5} product teams, shortening the average release cycle by ${20 + n} percent.`,
  (n) => `Owned the on-call rotation and reduced pages by ${30 + n * 2} percent by fixing the ${n + 3} noisiest alerts and writing runbooks for the rest.`,
  (n) => `Partnered with design and research to ship an accessibility overhaul that raised the audited score from ${61 + n} to ${92 - n}.`,
  (n) => `Prototyped a recommendation service in a ${n + 2}-week spike, then handed it to a team of ${n + 2} for production hardening.`,
];

function work(copies) {
  const entries = [];
  for (let c = 0; c < copies; c++) {
    COMPANIES.forEach(([name, position], i) => {
      const n = c * COMPANIES.length + i;
      const start = 2024 - i * 2 - 2;
      entries.push({
        name,
        position,
        url: '',
        startDate: `${start}-0${(i % 9) + 1}`,
        endDate: i === 0 ? '' : `${start + 2}-0${(i % 9) + 1}`,
        summary:
          i % 2 === 0
            ? `Joined to help scale the core product from a handful of customers to thousands, working across the stack with a focus on reliability.`
            : '',
        highlights: [0, 1, 2, 3, 4].map((j) => BULLETS[(i + j) % BULLETS.length](n % 7)),
      });
    });
  }
  return entries;
}

function data(copies) {
  return {
    basics: {
      name: 'Avery Morgan',
      label: 'Senior Software Engineer',
      email: 'avery.morgan@example.com',
      phone: '+1 555 0100',
      url: 'https://example.com/avery',
      summary:
        'Software engineer with twelve years of experience building reliable data products. I like small teams, clear writing and boring technology that works, and I enjoy turning a vague problem into a plan that other people can pick up and run with.',
      location: { city: 'Portland', region: 'OR', countryCode: 'US' },
      profiles: [
        { network: 'GitHub', username: 'avery-morgan', url: 'https://github.com/avery-morgan' },
        {
          network: 'LinkedIn',
          username: 'avery-morgan',
          url: 'https://www.linkedin.com/in/avery-morgan',
        },
      ],
    },
    work: work(copies),
    education: [
      {
        institution: 'Cascade State University',
        url: '',
        area: 'Computer Science',
        studyType: 'M.S.',
        startDate: '2010',
        endDate: '2012',
        score: '',
      },
      {
        institution: 'Rainier College',
        url: '',
        area: 'Mathematics',
        studyType: 'B.S.',
        startDate: '2006',
        endDate: '2010',
        score: '',
      },
    ],
    projects: [
      ['Tidepool', 'A tiny open-source dashboard for tracking service level objectives.'],
      ['Ledgerline', 'A command-line tool that reconciles bank exports against a plain-text ledger.'],
      ['Fieldnotes', 'A local-first notes app with end-to-end encrypted sync.'],
      ['Mapwright', 'A static-site generator for hand-drawn walking maps.'],
      ['Quietmail', 'A digest service that batches newsletters into a single morning email.'],
      ['Shelfmark', 'A reading tracker that exports to plain Markdown.'],
    ].map(([name, description], i) => ({
      name,
      url: `https://example.com/${name.toLowerCase()}`,
      startDate: `${2016 + i}`,
      endDate: '',
      description,
      highlights: [
        `Reached ${(i + 2) * 150} GitHub stars and ${(i + 3) * 4} outside contributors.`,
        'Wrote the documentation, the test suite and the release automation.',
      ],
    })),
    skills: [
      ['Languages', ['TypeScript', 'Go', 'Python', 'SQL', 'Rust']],
      ['Data', ['PostgreSQL', 'Kafka', 'ClickHouse', 'dbt', 'Airflow']],
      ['Cloud', ['AWS', 'Terraform', 'Kubernetes', 'Docker']],
      ['Practices', ['Code review', 'Incident response', 'Technical writing', 'Mentoring']],
      ['Frontend', ['React', 'Svelte', 'Accessibility', 'Web performance']],
    ].map(([name, keywords]) => ({ name, level: '', keywords })),
    certificates: [
      ['AWS Certified Solutions Architect', 'Amazon Web Services', '2022-03'],
      ['Certified Kubernetes Administrator', 'CNCF', '2021-09'],
      ['Professional Scrum Master I', 'Scrum.org', '2019-05'],
      ['Google Data Engineer', 'Google Cloud', '2020-11'],
      ['HashiCorp Terraform Associate', 'HashiCorp', '2021-02'],
      ['Accessibility Fundamentals', 'Deque University', '2023-04'],
    ].map(([name, issuer, date]) => ({ name, issuer, date, url: '' })),
    languages: [
      { language: 'English', fluency: 'Native' },
      { language: 'Spanish', fluency: 'Professional working' },
      { language: 'Japanese', fluency: 'Elementary' },
    ],
    awards: [
      ['Engineering Excellence Award', 'Brightwave Systems', '2023-12', 'Recognised for leading the ingestion rewrite.'],
      ['Open Source Contributor of the Year', 'Harbor & Pine Software', '2020-12', 'Awarded for maintaining three internal libraries.'],
      ['Best Paper, Regional Data Systems Workshop', 'DSW', '2013-06', 'For work on incremental view maintenance.'],
    ].map(([title, awarder, date, summary]) => ({ title, awarder, date, summary })),
  };
}

/**
 * @param {number} copies how many times to repeat the six-job work history
 * @param {object} layout optional layout overrides (template, pageSize, ...)
 */
export function longResume(copies = 1, layout = {}) {
  const result = parseResume({ ...data(copies), 'x-layout': layout });
  if (!result.ok) throw new Error(`long resume is invalid: ${JSON.stringify(result.errors)}`);
  return result.data;
}

import { describe, it, expect } from 'vitest';
import { renderTimeline, renderTimelineItem } from '../src/render.js';

const sampleEntries = [
  {
    type: 'education',
    date: '2020 – 2024',
    title: 'B.Tech – Computer Science',
    org: 'Vellore Institute of Technology',
    description: 'Core foundation in algorithms.',
  },
  {
    type: 'experience',
    date: 'May 2023 – Jul 2023',
    title: 'Data Intern',
    org: 'TheSmartBridge',
    description: 'Evaluated SaaS market opportunities.',
    highlights: ['Researched product gaps.', 'Presented recommendations.'],
  },
  {
    type: 'certification',
    date: 'Jul 2023',
    title: 'AWS Certified Cloud Practitioner',
    org: 'Amazon Web Services',
  },
];

describe('renderTimelineItem', () => {
  it('renders the category badge, date, title, and org', () => {
    const li = renderTimelineItem(sampleEntries[0], document);
    expect(li.tagName).toBe('LI');
    expect(li.className).toContain('timeline-item--education');
    expect(li.querySelector('.timeline-item__badge').textContent).toBe('Education');
    expect(li.querySelector('.timeline-item__date').textContent).toBe('2020 – 2024');
    expect(li.querySelector('.timeline-item__title').textContent).toBe('B.Tech – Computer Science');
    expect(li.querySelector('.timeline-item__org').textContent).toBe('Vellore Institute of Technology');
  });

  it('renders a description when present', () => {
    const li = renderTimelineItem(sampleEntries[0], document);
    expect(li.querySelector('.timeline-item__description').textContent).toBe(
      'Core foundation in algorithms.',
    );
  });

  it('omits the description element when not present', () => {
    const li = renderTimelineItem(sampleEntries[2], document);
    expect(li.querySelector('.timeline-item__description')).toBeNull();
  });

  it('renders highlights as a list when present', () => {
    const li = renderTimelineItem(sampleEntries[1], document);
    const items = li.querySelectorAll('.timeline-item__highlights li');
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toBe('Researched product gaps.');
  });

  it('omits the highlights list when not present', () => {
    const li = renderTimelineItem(sampleEntries[0], document);
    expect(li.querySelector('.timeline-item__highlights')).toBeNull();
  });

  it('falls back to the raw type string for an unknown category', () => {
    const li = renderTimelineItem({ ...sampleEntries[0], type: 'award' }, document);
    expect(li.querySelector('.timeline-item__badge').textContent).toBe('award');
  });
});

describe('renderTimeline', () => {
  it('renders one <li> per entry inside an <ol class="timeline">', () => {
    const ol = renderTimeline(sampleEntries, document);
    expect(ol.tagName).toBe('OL');
    expect(ol.className).toBe('timeline');
    expect(ol.querySelectorAll('.timeline-item')).toHaveLength(sampleEntries.length);
  });

  it('preserves entry order', () => {
    const ol = renderTimeline(sampleEntries, document);
    const titles = Array.from(ol.querySelectorAll('.timeline-item__title')).map((el) => el.textContent);
    expect(titles).toEqual([
      'B.Tech – Computer Science',
      'Data Intern',
      'AWS Certified Cloud Practitioner',
    ]);
  });

  it('renders an empty list for an empty entries array', () => {
    const ol = renderTimeline([], document);
    expect(ol.querySelectorAll('.timeline-item')).toHaveLength(0);
  });
});

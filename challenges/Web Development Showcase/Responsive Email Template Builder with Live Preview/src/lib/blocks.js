let idCounter = 0;

export function nextId() {
  idCounter += 1;
  return `block-${idCounter}-${Date.now().toString(36)}`;
}

export const BLOCK_TYPES = [
  { type: 'heading', label: 'Heading' },
  { type: 'paragraph', label: 'Paragraph' },
  { type: 'image', label: 'Image' },
  { type: 'button', label: 'Button' },
  { type: 'divider', label: 'Divider' },
  { type: 'spacer', label: 'Spacer' },
  { type: 'columns', label: 'Two columns' },
];

export function createBlock(type) {
  const id = nextId();
  switch (type) {
    case 'heading':
      return {
        id,
        type,
        props: { text: 'Your heading here', align: 'left', color: '#111827', fontSize: 24 },
      };
    case 'paragraph':
      return {
        id,
        type,
        props: {
          text: 'Write your message here. Keep paragraphs short for email readability.',
          align: 'left',
          color: '#374151',
          fontSize: 14,
        },
      };
    case 'image':
      return {
        id,
        type,
        props: { src: 'https://placehold.co/600x200', alt: 'Descriptive alt text', width: 600 },
      };
    case 'button':
      return {
        id,
        type,
        props: {
          label: 'Call to action',
          href: 'https://example.com',
          bgColor: '#2563eb',
          textColor: '#ffffff',
          align: 'center',
        },
      };
    case 'divider':
      return { id, type, props: { color: '#e5e7eb', height: 1 } };
    case 'spacer':
      return { id, type, props: { height: 24 } };
    case 'columns':
      return {
        id,
        type,
        props: { leftText: 'Left column content.', rightText: 'Right column content.' },
      };
    default:
      throw new Error(`Unknown block type: ${type}`);
  }
}

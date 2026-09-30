// Photos are from Unsplash (Unsplash License), downsized to <= 800px WebP.
// `id` is the Unsplash photo id; full credits are in the README.
const photos = [
  ['whangarei-falls-footbridge', 'eOpewngf68w', 'Whangarei Falls footbridge', 'Tim Swaan', 800, 534, 'A blue and brown steel footbridge at Whangarei Falls'],
  ['golden-flower-petals-in-macro', 'wR9VG-W8nU4', 'Golden petals', 'Volodymyr Lymariev', 533, 800, 'Macro shot of golden flower petals'],
  ['reeds-in-silhouette-on-lake', 'hLYbJB-D5Gg', 'Reeds at dusk', 'Mark Dixon', 800, 600, 'Reeds in silhouette on a lake'],
  ['orange-dragonfly-on-thin-twig', 'IG1yO9YDkqU', 'Dragonfly', 'CR', 533, 800, 'Orange dragonfly perched on a thin twig'],
  ['lightning-over-dark-mountain-peaks', 'd-p06WttJJE', 'Lightning storm', 'Marek Piwnicki', 800, 450, 'Lightning over dark mountain peaks'],
  ['bumblebee-hovering-near-purple-salvia', 'a3HmolpGW3s', 'Bumblebee', 'Dmytro Koplyk', 800, 600, 'Bumblebee hovering near purple salvia'],
  ['sea-stack-at-shark-fin-cove', 'shn9z-172sM', 'Shark Fin Cove', 'Karla Hernandez', 640, 800, 'Sea stack rising from the water at Shark Fin Cove'],
  ['black-cat-resting-on-table', '8CsDIpCytF0', 'Cat nap', 'Bastian Alexander-Coleman', 800, 486, 'Black cat resting on a table'],
  ['orange-hibiscus-flower-in-dark-garden', 'hBfY_uyLwAE', 'Hibiscus', 'Iván Díaz', 533, 800, 'Orange hibiscus flower in a dark garden'],
  ['north-america-nebula-with-gas-clouds', '3lSdgBnv9ag', 'North America Nebula', 'Wallace Henry', 800, 551, 'North America Nebula with glowing gas clouds'],
  ['tortoise-with-textured-brown-shell', 'tI7TOjJOFqI', 'Tortoise', 'Adrian Botica', 584, 800, 'Tortoise with a textured brown shell'],
  ['painted-lady-butterfly-on-purple-salvia', 'R0PjjzRWWf8', 'Painted lady', 'Dmytro Koplyk', 800, 600, 'Painted lady butterfly on purple salvia'],
  ['bald-eagle-in-profile', 'DTY3cKv0pvc', 'Bald eagle', 'Venti Views', 800, 584, 'Bald eagle in profile'],
  ['crescent-moon-behind-blurred-tree-branches', '1o5hFRQ77l0', 'Crescent moon', 'Xx M', 800, 534, 'Crescent moon behind blurred tree branches'],
  ['female-northern-cardinal-on-branch', 'TXmLf1NSTVs', 'Northern cardinal', 'Dmytro Koplyk', 800, 534, 'Female northern cardinal on a branch'],
];

export const PHOTOS = photos.map(([file, unsplashId, title, photographer, width, height, alt]) => ({
  id: unsplashId,
  src: `photos/${file}.webp`,
  title,
  photographer,
  width,
  height,
  alt,
  url: `https://unsplash.com/photos/${unsplashId}`,
}));

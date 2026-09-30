import { createAlbum } from './album.js';
import { enableDragAndDrop } from './dnd.js';
import { PHOTOS } from './photos.js';

const status = document.getElementById('status');
const reset = document.getElementById('reset');

function announce(message) {
  // Clearing first makes a repeated identical message announce again.
  status.textContent = '';
  requestAnimationFrame(() => {
    status.textContent = message;
  });
}

function getStorage() {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

const album = createAlbum({
  grid: document.getElementById('album'),
  photos: PHOTOS,
  storage: getStorage(),
  announce,
});
enableDragAndDrop(album);

reset.addEventListener('click', () => album.reset());

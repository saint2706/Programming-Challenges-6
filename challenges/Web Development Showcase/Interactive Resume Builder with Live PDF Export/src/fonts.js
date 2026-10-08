// Self-hosted static fonts (Fontsource): four weights of three families. These only declare
// @font-face rules; the browser fetches a file the first time text actually uses it.
//
// Static (not variable) files on purpose: Chrome can subset a static TrueType font into the PDF as
// a real TrueType font, but it embeds a *variable* font as a Type 3 glyph-drawing font, which some
// applicant tracking systems cannot extract text from. The e2e test checks the PDF fonts.
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/source-serif-4/400.css';
import '@fontsource/source-serif-4/500.css';
import '@fontsource/source-serif-4/600.css';
import '@fontsource/source-serif-4/700.css';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans/700.css';
